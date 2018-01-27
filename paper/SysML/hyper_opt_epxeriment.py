import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import ReLU, CrossEntropyLoss, MaxPool2d
from torch.nn.functional import relu
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from dynnet.graph import Graph
from dynnet.filters import SimpleFilter
from sklearn.model_selection import train_test_split
from uuid import uuid4
import sys
from dynnet.layers import (
    Input, Linear, Conv2d, Flatten
)

SEED = 0
COUNT = 50
CRITERION = CrossEntropyLoss()
NORMALIZATIONS = {
    MNIST: [(0.13066047740239478,), (0.30810780876661253,)],
    FashionMNIST: [(0.2860406021898328,), (0.353024253432129,)],
    CIFAR10: [(0.49139968, 0.48215845, 0.44653094),
              (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)]
}
SIZES = {
    MNIST: (1, 28, 28),
    FashionMNIST: (1, 28, 28),
    CIFAR10: (3, 32, 32)
}
REGULARIZATIONS = {
    MNIST: [2, 5],
    FashionMNIST: [2, 5],
    CIFAR10: [2, 6]
}

DATASETS = {
    'MNIST': MNIST,
    'FashionMNIST' : FashionMNIST,
    'CIFAR10': CIFAR10
}

def preload_dataset(dataset, train):
    trans = [
        transforms.ToTensor(),
        transforms.Normalize(*NORMALIZATIONS[dataset])
    ]
    ds = dataset(
            '../datasets/%s/' % dataset.__name__,
            train=train,
            download=True,
            transform=transforms.Compose(trans))
    # load everything
    dummy_dl = DataLoader(ds, batch_size=10000000, shuffle=False)
    inputs, labels = next(iter(dummy_dl))
    return inputs.numpy(), labels.numpy()

def split_dataset(inputs, labels):
    X_train, X_test, y_train, y_test = train_test_split(
         inputs, labels, test_size=0.20, random_state=SEED)
    return (X_train, y_train), (X_test, y_test)

def prepare_loaders(dataset, batch_sizes):
    training_tuple = preload_dataset(dataset, True)
    testing_tuple = preload_dataset(dataset, False)
    training_tuple, validation_tuple = split_dataset(*training_tuple)
    all_tuples = [training_tuple, validation_tuple, testing_tuple]
    return [DataLoader(TensorDataset(*[
        torch.from_numpy(x).cuda() for x in ds
    ]), batch_size=b, shuffle=True)
            for ds, b
            in zip(all_tuples, batch_sizes)]


def init_model(model):
    for parameter in model.parameters():
        if len(parameter.size()) > 1:
            torch.nn.init.xavier_normal(parameter.data, gain=np.sqrt(2))
    for l in model:
        if isinstance(l, SimpleFilter):
            l.weight.data.uniform_(0, 1)


def create_dynamic_model(start_size=5000, size=(1, 28, 28), conv=False):
    graph = Graph()
    if not conv:
        l = graph.add(Input, size[1] * size[2] * size[0])()
        for i in range(3):
            l = graph.add(Linear, out_features=start_size)(l)
            l = graph.add(SimpleFilter)(l)
            l = graph.add(ReLU)(l)
        graph.add(Linear, out_features=10)(l)
    else:
        l = graph.add(Input, *size)()
        l = graph.add(Conv2d, out_channels=start_size[0], kernel_size=5)(l)
        l = graph.add(SimpleFilter)(l)
        l = graph.add(ReLU)(l)
        l = graph.add(MaxPool2d, kernel_size=2)(l)
        l = graph.add(Conv2d, out_channels=start_size[1], kernel_size=5)(l)
        l = graph.add(SimpleFilter)(l)
        l = graph.add(ReLU)(l)
        l = graph.add(MaxPool2d, kernel_size=2)(l)
        l = graph.add(Flatten)(l)
        l = graph.add(Linear, out_features=start_size[2])(l)
        l = graph.add(SimpleFilter)(l)
        l = graph.add(ReLU)(l)
        l = graph.add(Linear, out_features=10)(l)
    init_model(graph)
    return graph.cuda()

class TorchFlatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    def __repr__(self):
        return "Flatten()"

def create_static_model(capacities=(200, 200, 200), size=(1, 28, 28), conv=False):
    layers = []
    if conv:
        layers.append(torch.nn.Conv2d(size[0], capacities[0], 5))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.MaxPool2d(2))
        layers.append(torch.nn.Conv2d(capacities[0], capacities[1], 5))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.MaxPool2d(2))
        layers.append(TorchFlatten())
        input_count = int(capacities[1] * (((size[2] - 4) / 2) - 4) / 2 * (((size[1] - 4) / 2) - 4) / 2)
        layers.append(torch.nn.Linear(input_count, capacities[2]))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(capacities[2], 10))
    else:
        capacities = (int(np.array(size).prod()),) + capacities
        for i in range(3):
            layers.append(torch.nn.Linear(capacities[i], capacities[i + 1]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(capacities[-1], 10))
    model = torch.nn.Sequential(*layers)
    init_model(model)
    return model.cuda()


def regularized_loss(graph, prediction, labels, lamb):
    loss = CRITERION(prediction, labels)
    filter_weights = []
    for m in graph:
        if isinstance(m, SimpleFilter):
            filter_weights.append(m.weight)
    if not filter_weights:
        return loss, loss
    return loss, loss + lamb * relu(torch.cat(filter_weights)).sum()

def compute_network_size(graph, conv=False, parts=False):
    if not conv:
        if isinstance(graph, Graph):
            t = ([m.get_alive_features().float().sum() for m in graph if isinstance(m, SimpleFilter)])
            return sum(t)
        else:
            total = 0
            sizes = []
            for layer in graph.modules():
                if isinstance(layer, torch.nn.Linear):
                    sizes.append(layer.in_features)
                    sizes.append(layer.out_features)
            total += sum(sizes[1:-1]) / 2
            return total
    else:
        res = 0
        for p in graph.parameters():
            res += p.data.view(-1).size(0)
        return res


def forward(graph, dataloader, lamb=0, conv=False, optimizer=None):
    accs = []
    sizes = []
    for inputs, labels in dataloader:
        if not conv:
            inputs = inputs.view(inputs.size(0), -1)
        inputs = Variable(inputs, volatile=(optimizer is None))
        labels = Variable(labels.cuda(), volatile=(optimizer is None))
        if isinstance(graph, Graph):
            prediction,  = graph({graph[0]: inputs}, graph[-1])
        else:
            prediction = graph(inputs)
        _, loss = regularized_loss(graph, prediction, labels, lamb)
        discrete_prediction = prediction.max(1)[1]
        accuracy = (discrete_prediction == labels).float().data.mean()
        size = compute_network_size(graph, conv)
        accs.append(accuracy)
        sizes.append(size)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return accs, sizes

def train(graph, dataset=MNIST, wd=0, batch_size=256,
          lamb=0,conv=False):
    if conv:
        epochs = 200
    else:
        epochs = 100
    optimizer = Adam(graph.parameters(), weight_decay=wd)
    train_dl, val_dl, test_dl = prepare_loaders(dataset, (batch_size, 1000, 1000))
    train_accs = []
    val_accs = []
    test_accs = []
    all_sizes = []

    for i in range(epochs):
        graph.train()
        tracc, cs = forward(graph, train_dl, lamb, conv, optimizer)
        graph.eval()
        vaccs, _ = forward(graph, val_dl, 0, conv, None)
        taccs, _ = forward(graph, test_dl, 0, conv, None)
        train_accs.append(np.array(tracc).mean())
        val_accs.append(np.array(vaccs).mean())
        test_accs.append(np.array(taccs).mean())
        all_sizes.append(np.array(cs).mean())
        print(train_accs[-1], val_accs[-1], test_accs[-1], all_sizes[-1])
        if isinstance(graph, Graph):
            log = graph.garbage_collect()
            log.update_optimizer(optimizer)
            pass
    return np.array([train_accs, val_accs, test_accs, all_sizes])

def sample_static_sizes(conv=False):
    r = []
    if conv:
        for i in range(2):
            r.append(int(np.random.uniform(1, 50)))
        r.append(int(2.0 ** np.random.uniform(4, 12)))
    else:
        for i in range(3):
            r.append(int(2.0 ** np.random.uniform(4, 12)))
    return tuple(r)

def store_training(prefix, params, logs):
    with open('./experiments_results/hyper_opt_%s_%s' % (prefix, uuid4()), 'wb') as f:
        torch.save([params, logs], f)

def sample_regularization(dataset):
    range = REGULARIZATIONS[dataset]
    return 10**(-1 * np.random.uniform(*range))

def hyper_opt_static(dataset, bs, conv=False):
    for i in range(COUNT):
        sizes = sample_static_sizes(conv)
        params = {
            'sizes': sizes
        }
        print(params)
        model = create_static_model(sizes, SIZES[dataset], conv)
        log = train(model, dataset, batch_size=bs, conv=conv)
        p = dataset.__name__
        if conv:
            p = "conv_" + p
        store_training("%s_static" % p, params, log)

def hyper_opt_shrink(dataset, bs, conv=False):
    for i in range(COUNT):
        lamb = sample_regularization(dataset)
        params = {
            'lamb': lamb
        }
        print(params)
        start_sizes = 500
        if conv:
            start_sizes = (50, 50, 5000)
        model = create_dynamic_model(start_sizes, SIZES[dataset], conv)
        log = train(model, dataset, batch_size=bs, conv=conv, lamb=lamb)
        p = dataset.__name__
        if conv:
            p = "conv_" + p
        store_training("%s_shrink" % p, params, log)

def hyper_opt_decay(dataset, bs, conv=False):
    for i in range(COUNT):
        sizes = sample_static_sizes(conv)
        decay = sample_regularization(dataset)
        params = {
            'sizes': sizes,
            'decay': decay
        }
        print(params)
        model = create_static_model(sizes, SIZES[dataset], conv)
        log = train(model, dataset, batch_size=bs, conv=conv, wd=decay)
        p = dataset.__name__
        if conv:
            p = "conv_" + p
        store_training("%s_decay" % p, params, log)

dataset = DATASETS.get(sys.argv[-1])
mode =sys.argv[-2]
conv = sys.argv[-3] == 'conv'
if mode == 'static':
    hyper_opt_static(dataset, 256, conv=conv)
elif mode == 'decay':
    hyper_opt_decay(dataset, 256, conv=conv)
elif mode == 'shrink':
    hyper_opt_shrink(dataset, 256, conv=conv)
