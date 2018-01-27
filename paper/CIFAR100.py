import numpy as np
from models.vgg import VGG
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from dynnet.utils import shrinknet_penalty
from dynnet.filters import SimpleFilter
from dynnet.graph import Graph
import torchsample as ts
import torch

SEED = 0
CRITERION = CrossEntropyLoss()
NORMALIZATIONS = {
    MNIST: [(0.13066047740239478,), (0.30810780876661253,)],
    FashionMNIST: [(0.2860406021898328,), (0.353024253432129,)],
    CIFAR10: [(0.49139968, 0.48215845, 0.44653094),
              (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)],
    CIFAR100: [(0.50707513, 0.48654887, 0.44091785),
               (0.26733428, 0.25643846, 0.27615047)]
}
SIZES = {
    MNIST: (1, 28, 28),
    FashionMNIST: (1, 28, 28),
    CIFAR10: (3, 32, 32),
    CIFAR100: (3, 32, 32)
}
REGULARIZATIONS = {
    MNIST: [2, 5],
    FashionMNIST: [2, 5],
    CIFAR10: [2, 6],
    CIFAR100: [2, 6]
}

DATA_AUGMENTATIONS = {
    MNIST: [],
    FashionMNIST: [],
    CIFAR100: [
        ts.transforms.RandomFlip(True, False, 0.5),
        ts.transforms.RandomShear(25),
        ts.transforms.RandomBrightness(-0.3, 0.4),
        ts.transforms.RandomRotate(35),
        ts.transforms.RandomSaturation(-0.8, 1),
    ],
    CIFAR10: [
        ts.transforms.RandomFlip(True, False, 0.5),
        ts.transforms.RandomShear(25),
        ts.transforms.RandomBrightness(-0.3, 0.4),
        ts.transforms.RandomRotate(35),
        ts.transforms.RandomSaturation(-0.8, 1),
    ]
}

class IndexDataset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        super(IndexDataset, self).__init__()
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, ix):
        result = self.dataset[self.indices[ix]]
        if self.transform is not None:
            result = (self.transform(result[0]), result[1])
        return result

def create_model():
    model = VGG('VGG19').cuda()



def preload_dataset(ds, batch_size):
    dummy_dl = DataLoader(ds, batch_size=10000000, shuffle=False)
    inputs, labels = next(iter(dummy_dl))
    return DataLoader(TensorDataset(*[
        x.cuda() for x in (inputs, labels)]),
        batch_size=batch_size, shuffle=False)

def split_dataset(inputs, labels):
    X_train, X_test, y_train, y_test = train_test_split(
         inputs, labels, test_size=0.20, random_state=SEED)
    return (X_train, y_train), (X_test, y_test)

def prepare_loaders(dataset, batch_sizes, split=0.8):
    basic_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*NORMALIZATIONS[dataset])
    ])
    data_augmentations = transforms.Compose([
        transforms.ToTensor()] + DATA_AUGMENTATIONS[dataset] + \
        [transforms.Normalize(*NORMALIZATIONS[dataset])])

    full_train_dataset_simple = dataset(
            '../datasets/%s/' % dataset.__name__,
            train=True, download=True,
            transform=basic_transforms)
    full_train_dataset_augmented = dataset(
            '../datasets/%s/' % dataset.__name__,
            train=True, download=True,
            transform=data_augmentations)
    testing_dataset = dataset(
            '../datasets/%s/' % dataset.__name__,
            train=False, download=True,
            transform=basic_transforms)
    indices = np.arange(0, len(full_train_dataset_simple))
    np.random.shuffle(indices)
    boundary = int(len(indices) * split)
    train_indices = indices[:boundary]
    validation_indices = indices[boundary:]
    training_dataset = IndexDataset(full_train_dataset_augmented, train_indices,
                                transform=None)
    validation_dataset = IndexDataset(full_train_dataset_simple, validation_indices)

    training_dataloader = DataLoader(training_dataset,
                                     batch_size=batch_sizes[0],
                                     pin_memory=True,
                                     shuffle=True,
                                     num_workers=40)
    validation_dataloader = preload_dataset(validation_dataset, batch_sizes[1])
    testing_dataloader = preload_dataset(testing_dataset, batch_sizes[2])
    return training_dataloader, validation_dataloader, testing_dataloader

def forward(graph, dataloader, lamb=0, optimizer=None):
    accs = []
    sizes = []

    for inputs, labels in dataloader:
        inputs = Variable(inputs.cuda(async=True), volatile=(optimizer is None))
        labels = Variable(labels.cuda(async=True), volatile=(optimizer is None))
        prediction = graph(inputs)
        loss = CRITERION(prediction, labels) + lamb * shrinknet_penalty(graph)
        discrete_prediction = prediction.max(1)[1]
        accuracy = (discrete_prediction == labels).float().data.mean()
        accs.append(accuracy)
        sizes.append(compute_size(graph))
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return accs, sizes

def compute_size(model):
    layers = []
    for module in model:
        if isinstance(module, SimpleFilter):
            layers.append(int(module.get_alive_features().float().sum()))
    return layers

def init_model(model):
    for parameter in model.parameters():
        if len(parameter.size()) > 1:
            torch.nn.init.kaiming_uniform(parameter.data)

def split_params(model):
    filter_parameters = set()
    for module in model.modules():
        if isinstance(module, SimpleFilter):
            for p in module.parameters():
                filter_parameters.add(p)
    other_parameters = set(model.parameters()) - filter_parameters
    return other_parameters, filter_parameters

def optimize(graph, all_optimizers, optimizer, train_accs, val_accs, test_accs, all_sizes, train_dl, val_dl, test_dl, lamb):
    graph.train()
    tracc, cs = forward(graph, train_dl, lamb, optimizer)
    # Do garbage collection ASAP
    if isinstance(graph, Graph):
        log = graph.garbage_collect()
        for op in all_optimizers:
            log.update_optimizer(op)
    graph.eval()
    vaccs, _ = forward(graph, val_dl, 0, None)
    taccs, _ = forward(graph, test_dl, 0, None)
    train_accs.append(np.array(tracc).mean())
    val_accs.append(np.array(vaccs).mean())
    test_accs.append(np.array(taccs).mean())
    all_sizes.append(np.array(cs).mean(0))
    print(train_accs[-1], val_accs[-1], test_accs[-1], list(all_sizes[-1]))

def train(graph, dataset=MNIST, wd=0, batch_size=256,
          lamb=0, epochs=200):
    init_model(graph)
    other_parameters, filter_parameters = split_params(graph)
    optimizer = SGD([{
        'params': other_parameters,
        'lr': 0.01,
        'momentum': 0.9,
        'nesterov': True
    }, {
        'params': filter_parameters,
        'lr': 0,
        'momentum': 0.3,
        'nesterov': True
    }])
    train_dl, val_dl, test_dl = prepare_loaders(dataset, (batch_size, 1000, 1000))
    train_accs = []
    val_accs = []
    test_accs = []
    all_sizes = []

    for i in range(epochs):
        _, filter_parameters = split_params(graph)
        opt2 = Adam(filter_parameters)
        optimize(graph, [optimizer, opt2], optimizer, train_accs, val_accs, test_accs, all_sizes, train_dl, val_dl, test_dl, lamb)
        optimize(graph, [optimizer, opt2], opt2, train_accs, val_accs, test_accs, all_sizes, train_dl, val_dl, test_dl, lamb)

    return np.array([train_accs, val_accs, test_accs, all_sizes])

