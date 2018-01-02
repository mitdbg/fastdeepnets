import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import ReLU, CrossEntropyLoss
from torch.nn.functional import relu
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
from dynnet.graph import Graph
from dynnet.layers import Input, Linear
from dynnet.filters import SimpleFilter

SEED = 0
CRITERION = CrossEntropyLoss()
MNIST_TRANSFORM = [
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]
FashionMNIST_TRANSFORM= [
    transforms.ToTensor(),
    transforms.Normalize((0.2860), (0.3530,))
]


def preload_dataset(dataset, train, batch_size):
    if dataset == MNIST:
        trans = MNIST_TRANSFORM
    elif dataset == FashionMNIST:
        trans = FashionMNIST_TRANSFORM
    else:
        assert False, 'Unknown datset'
    ds = dataset(
            '../datasets/%s/' % dataset.__name__,
            train=train,
            download=True,
            transform=transforms.Compose(trans))
    # load everything
    dummy_dl = DataLoader(ds, batch_size=10000000, shuffle=False)
    inputs, labels = next(iter(dummy_dl))
    inputs = inputs.cuda()
    labels = labels.cuda()
    final_dataset = TensorDataset(inputs, labels)
    dl= DataLoader(final_dataset, shuffle=train, batch_size=batch_size)
    return dl


def create_model(start_size=10000):
    graph = Graph()
    inp = graph.add(Input, 28*28)()
    last_layer = graph.add(Linear, out_features=start_size)(inp)
    last_layer = graph.add(SimpleFilter)(last_layer)
    last_layer = graph.add(ReLU)(last_layer)
    graph.add(Linear, out_features=10)(last_layer)
    for parameter in graph.parameters():
        if len(parameter.size()) == 2:
            torch.nn.init.xavier_normal(parameter.data, gain=np.sqrt(2))
    graph[2].weight.data.uniform_(0, 1)
    return graph.cuda()


def regularized_loss(graph, prediction, labels, lamb):
    loss = CRITERION(prediction, labels)
    return loss, loss + lamb * relu(graph[2].weight).sum()


def forward(graph, dataloader, lamb=0, optimizer=None):
    losses = []
    sizes = []
    for inputs, labels in dataloader:
        inputs = Variable(inputs.view(-1, 28*28))
        labels = Variable(labels.cuda())
        prediction,  = graph({graph[0]: inputs}, graph[-1])
        original_loss, loss = regularized_loss(graph, prediction, labels, lamb)
        discrete_prediction = prediction.max(1)[1]
        accuracy = (discrete_prediction == labels).float().data.mean()
        size = graph[2].get_alive_features().float().sum()
        losses.append(original_loss.data.cpu().numpy()[0])
        sizes.append(size)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return losses, sizes

def train(dataset=MNIST, batch_size=256, lamb=0, epochs=100):
    graph = create_model()
    optimizer = Adam(graph.parameters())
    training_set = preload_dataset(dataset, True, batch_size)
    all_losses = []
    all_sizes = []
    with open('./experiments_results/conv_%s_bs%s_lamb%s.dat' %
              (dataset.__name__, batch_size, lamb), 'w') as f:

        for i in range(epochs):
            cl, cs = forward(graph, training_set, lamb, optimizer)
            for tup in zip(cl, cs):
                f.write("%s %s\n" % (tup))
            f.flush()
            log = graph.garbage_collect()
            log.update_optimizer(optimizer)

for ds in [MNIST, FashionMNIST]:
    for bs in [128, 256, 512]:
        for lamb in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
            train(ds, bs, lamb)


