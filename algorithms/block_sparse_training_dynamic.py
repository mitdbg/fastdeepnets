import torch
from copy import deepcopy
import numpy as np
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torchvision.datasets import MNIST, FashionMNIST
from utils.misc import tn
from modules.dynamic import DynamicModule
from utils.wrapping import wrap

from models.MultiLayerDynamicPerceptron import MultiLayerDynamicPerceptron

EPOCHS = 15

def get_dynamic(model):
    for m in model.modules():
        if isinstance(m, DynamicModule):
            yield m

def garbage_collect(model):
    for m in get_dynamic(model):
        m.garbage_collect()

def get_block_paramteters(model, block=-1):
    for m in get_dynamic(model):
        for p in m.block_parameters(block):
            yield p

def get_capacities(model):
    c = []
    for m in get_dynamic(model):
        if m.out_features is None:
            c.append(m.num_output_features)
    return c

def disable_last_block(model):
    for m in get_dynamic(model):
        if m.out_features is None:
            m.filters_blocks[-1].data.zero_()
    garbage_collect(model)

def get_capacity(model):
    return sum(get_capacities(model), 0)

def get_l1_loss(model):
    losses = [x.l1_loss(True) for x in get_dynamic(model)]
    return torch.cat(losses).sum()

def has_collapsed(model):
    return tn(get_l1_loss(model).data) == 0

def grow(model, factor=2):
    for m in get_dynamic(model):
        if m.out_features is None:
            sizes = m.block_features
            next_size = max([x * factor for x in sizes] + [model.initial_size])
            m.grow(next_size)
        else:
            m.grow()
    m.garbage_collect()
    model() # fake pass

def forward(model, dl, lamb=0, optimizer=None):
    criterion = CrossEntropyLoss()
    acc_sum = 0
    tot = 0
    for i, (images, labels) in enumerate(dl):
        images = wrap(Variable(images, requires_grad=False))
        labels = wrap(Variable(labels, requires_grad=False))
        output = model(images)
        if optimizer is not None:
            original_loss = criterion(output, labels)
            penalized_loss = original_loss + float(lamb) * get_l1_loss(model)
            # print(tn(original_loss.data), tn(penalized_loss.data), tn(penalty.data))
            optimizer.zero_grad()
            penalized_loss.backward()
            optimizer.step()

        acc = (output.max(1)[1] == labels).float().sum()
        tot += len(labels)
        acc_sum += acc

    acc = tn(acc_sum.data / tot)
    return acc


def train_until_convergence(model, train_dl, val_dl, lamb, patience=3, min_epochs=2):
    best_model = None
    best_score = (-np.inf, 0)
    early_stop = 0

    print('lambda', lamb)

    optimizer = Adam(get_block_paramteters(model))

    while True:
        train_acc = forward(model, train_dl, lamb, optimizer)
        current_accuracy = forward(model, val_dl)
        capacity = get_capacity(model)
        current_score = (-capacity, current_accuracy)

        print(train_acc, current_accuracy, get_capacities(model))

        if has_collapsed(model):
            raise StopIteration('Model collapsed')
        elif min_epochs > 0:
            min_epochs -=1
        elif current_score > best_score:
            best_model = deepcopy(model)
            best_score = current_score
            early_stop = 0
        elif early_stop == patience - 1:
            return best_score[1], best_model
        else:
            early_stop += 1

def train(model, train_dl, val_dl, test_dl, start_lamb=1, lamb_decay=5, max_patience=4, min_lambda=1e-7):
    lamb = start_lamb
    val_accuracies = []
    test_accuracies = []
    best_acc = 0
    best_model = model
    while lamb >= min_lambda:
        current_model = deepcopy(best_model)
        grow(current_model)
        while True: # Find Grip
            try:
                current_accuracy, current_model = train_until_convergence(current_model, train_dl, val_dl, lamb, patience=max_patience)
                print(current_model)
                break
            except StopIteration:
                print('COLLAPSED')
                lamb /= lamb_decay
                if lamb < min_lambda:
                    return val_accuracies, test_accuracies, model
                current_model = deepcopy(best_model)
                grow(current_model)
        if current_accuracy > best_acc:  # We did improve the model
            print('did improve')
            best_model = current_model # We build on top of this block
            best_acc = current_accuracy
            val_accuracies.append(best_acc)
            test_acc = forward(best_model, test_dl)
            print('TEST ACC:', test_acc)
            test_accuracies.append(test_acc)
        else:
            print('did not improve')
        lamb /= lamb_decay
    return val_accuracies, test_accuracies, best_model

if __name__ == "__main__":
    model = MultiLayerDynamicPerceptron(1, initial_size=15).cuda()
    print('hello owrld')
