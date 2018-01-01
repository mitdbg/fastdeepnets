import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import ReLU, CrossEntropyLoss
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import transforms
from torchvision.datasets import MNIST
from dynnet.graph import Graph
from dynnet.layers import Input, Linear
from dynnet.filters import SimpleFilter

SEED = 0
CRITERION = CrossEntropyLoss()
MNIST_TRANSFORM = [
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]

def preload_dataset(dataset, train, batch_size):
    ds = dataset(
            '../datasets/%s/' % dataset.__name__,
            train=train,
            download=True,
            transform=transforms.Compose(MNIST_TRANSFORM))
    # load everything
    dummy_dl = DataLoader(ds, batch_size=10000000, shuffle=False)
    final_dataset = TensorDataset(*next(iter(dummy_dl)))
    return DataLoader(final_dataset, shuffle=train, batch_size=batch_size)


def create_model(start_size=10000):
    graph = Graph()
    inp = graph.add(Input, 28*28)()
    last_layer = graph.add(Linear, out_features=start_size)(inp)
    last_layer = graph.add(SimpleFilter)(last_layer)
    last_layer = graph.add(ReLU)(last_layer)
    graph.add(Linear, out_features=10)(last_layer)
    return graph


def regularized_loss(graph, prediction, labels, lamb):
    loss = CRITERION(prediction, labels)
    return loss + lamb * graph[2].weight.sum()


def forward(graph, dataloader, lamb=0, optimizer=None):
    for inputs, labels in dataloader:
        inputs = Variable(inputs.view(-1, 28*28))
        labels = Variable(labels)
        prediction,  = graph({graph[0]: inputs}, graph[-1])
        loss = regularized_loss(graph, prediction, labels, lamb)
        discrete_prediction = prediction.max(1)[1]
        accuracy = (discrete_prediction == labels).float().data.mean()
        graph.garbage_collect()
        size = graph[2].get_alive_features().float().sum()
        # print(accuracy, size)
        d = graph[2].weight.data
        print(d.mean(), d.min(), d.max())
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train(dataset=MNIST, batch_size=256, lamb=0, epochs=100):
    graph = create_model()
    optimizer = Adam(graph.parameters())
    training_set = preload_dataset(dataset, True, batch_size)
    testing_set = preload_dataset(dataset, False, 10000)
    forward(graph, training_set, lamb, optimizer)
    forward(graph, training_set, lamb, optimizer)
    forward(graph, training_set, lamb, optimizer)
train(lamb=10)


