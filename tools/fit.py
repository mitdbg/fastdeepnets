import torch
import argparse
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import relu
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
from dynnet.graph import Sequential
from dynnet.layers import Input, Linear, BatchNorm
from dynnet.filters import SimpleFilter
from dynnet.utils import shrinknet_penalty

PARSER = argparse.ArgumentParser(
    'ShrinkNet example with a MLP for a regression')
PARSER.add_argument('--layers', default=3, type=int,
                    help='Number of layers in the Perceptron (default: 3)')
PARSER.add_argument('--dropout', default=0, type=int,
                    help='Dropout value (default: 0, no dropout)')
PARSER.add_argument('--batch_norm', help='Enable batch_norm, (default: disabled)',
                    action='store_true')
PARSER.add_argument('--validation_split', type=float, default=0.2,
                    help='Ratio of the training data allocated for validation (default: 0.8)')
PARSER.add_argument('--lambda', type=float, default=0.001,
                    help='The regularization factor that control the final size of the network (default: 0.001)')
PARSER.add_argument('--batch_size', type=int, default=32,
                    help='The Mini-Batch size for optimization')
PARSER.add_argument('--max_neurons', type=int, default=5000,
                    help='The max(ie. starting) size of each layer of the MLP (default: 5000)')
PARSER.add_argument('--input_features', type=int, default=32,
                    help='The number of input for the model (default: 32)')
PARSER.add_argument('--output_features', type=int, default=1,
                    help='The number of output for the model (default: 1)')
PARSER.add_argument('--epochs', type=int, default=100,
                    help='The number of epochs to train the model (default: 100)')
PARSER.add_argument('input', type=str,
                    help='The file name containing the training data')
PARSER.add_argument('output', type=str,
                    help='Where to store your trained model')

class Model(torch.nn.Module):

    def __init__(self, params):
        super(Model, self).__init__()
        layer_count = params['layers']
        dropout = params['dropout']
        batch_norm = params['batch_norm']
        max_neurons = params['max_neurons']
        input_features = params['input_features']
        output_features = params['output_features']
        graph = Sequential()
        graph.add(Input, input_features)

        assert layer_count > 0, "Need at least one layer"
        for _ in range(layer_count):
            graph.add(Linear, out_features=max_neurons)
            graph.add(SimpleFilter)
            graph.add(torch.nn.ReLU)
            if dropout > 0:
                graph.add(torch.nn.Dropout, p=dropout)
            if batch_norm:
                graph.add(BatchNorm)
        graph.add(Linear, out_features=output_features)
        self.graph = graph

    def forward(self, x):
        return self.graph(x)


def load_data(params):
    file_name = params['input']
    input_features = params['input_features']
    output_features = params['output_features']
    validation_split = params['validation_split']
    inputs = []
    labels = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            line = [float(x) for x in line]
            assert len(line) == input_features + output_features, (
                'Invalid number of features, %s (expected %s + %s) ' % (
                    len(line), input_features, output_features
                )
            )
            inputs.append(line[:input_features])
            labels.append(line[input_features:])
    inputs = np.array(inputs)
    labels = np.array(labels)
    random_indices = np.arange(0, labels.shape[0])
    np.random.shuffle(random_indices)
    split_point = int((1 - validation_split) * labels.shape[0])
    inputs = torch.from_numpy(inputs).float()
    labels = torch.from_numpy(labels).float()
    random_indices = torch.from_numpy(random_indices)
    training_dataset = TensorDataset(inputs[random_indices[:split_point]],
                                     labels[random_indices[:split_point]])
    validation_dataset = TensorDataset(inputs[random_indices[split_point:]],
                                       labels[random_indices[split_point:]])
    return training_dataset, validation_dataset

def forward(model, dl, params, optimizer=None):
    criterion = torch.nn.MSELoss()
    lamb = params['lambda']
    total_loss = 0
    for inputs, labels in dl:
        inputs = Variable(inputs, requires_grad=False, volatile=optimizer==None)
        y_true = Variable(labels, requires_grad=False,volatile=optimizer==None)
        y_pred = model(inputs)
        loss = criterion(y_pred, y_true)
        total_loss += loss.data.cpu().numpy()[0]
        if optimizer is not None:
            loss += lamb * shrinknet_penalty(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    total_loss /= len(dl)
    return total_loss

def compute_network_size(graph):
        t = ([m.get_alive_features().float().sum() for m in graph if isinstance(m, SimpleFilter)])
        return t

def train(model, training_dataset, validation_dataset, params):
    epochs = params['epochs']
    batch_size = params['batch_size']
    training_dl = DataLoader(training_dataset,
                             batch_size=batch_size,
                             shuffle=True)
    validation_dl = DataLoader(validation_dataset,
                               batch_size=100000000,
                               shuffle=False)
    optimizer = Adam(model.parameters())
    try:
        for i in range(epochs):
            model.train()
            t_loss = forward(model, training_dl, params, optimizer)
            model.eval()
            v_loss = forward(model, validation_dl, params)
            size = compute_network_size(model.graph)
            print(i, t_loss, v_loss, size)
            log = model.graph.garbage_collect()
            log.update_optimizer(optimizer)
    except:
        raise


def init_model(model):
    for parameter in model.parameters():
        if len(parameter.size()) > 1:
            torch.nn.init.xavier_normal(parameter.data, gain=np.sqrt(2))


args = PARSER.parse_args().__dict__
model = Model(args)
init_model(model.graph)
print('Your Model:')
print(model)
print('Start training')
print('Epoch -- Training Loss -- Validation loss -- Hidden sizes')
training_dataset, validation_dataset = load_data(args)
train(model, training_dataset, validation_dataset, args)
with open(args['output'], 'wb') as f:
    torch.save(model, f)
