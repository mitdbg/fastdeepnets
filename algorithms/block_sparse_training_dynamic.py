import torch
from copy import deepcopy
import numpy as np
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss, MSELoss
from torch.autograd import Variable
from utils.misc import tn
from modules.dynamic import DynamicModule
from utils.wrapping import wrap
from optimizers.COCOB import COCOB
from utils.measures import TrainingStats
from time import time


def get_dynamic(model):
    for m in model.modules():
        if isinstance(m, DynamicModule):
            yield m

def garbage_collect(model):
    for m in get_dynamic(model):
        m.garbage_collect()

def set_optimizer(model, optimizer):
    for m in get_dynamic(model):
        m.set_optimizer(optimizer)

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
    if len(losses) == 0:
        return None
    return torch.cat(losses).sum()

def has_collapsed(model):
    l1 = get_l1_loss(model)
    if l1 is None:
        return False
    return tn(l1.data) == 0

def grow(model, factor=2):
    for m in get_dynamic(model):
        if hasattr(m, 'out_features') and m.out_features is None:
            sizes = m.block_features
            initial_size = m.initial_size if hasattr(m, 'initial_size') else model.initial_size
            next_size = max([x * factor for x in sizes] + [initial_size])
            m.grow(next_size)
        else:
            m.grow()
    m.garbage_collect()
    model() # fake pass

def forward(model, dl, lamb=0, optimizer=None, mode='classification', stats=None, weight=None):
    if mode == 'classification':
        criterion = CrossEntropyLoss(weight=weight, size_average=True)
    else:
        criterion = MSELoss(size_average=False)
    acc_sum = 0
    tot = 0
    for i, (images, labels) in enumerate(dl):
        if stats is not None:
            stats.next_batch()
        images = wrap(Variable(images, requires_grad=False))
        labels = wrap(Variable(labels, requires_grad=False))
        output = model(images)
        if optimizer is not None:
            original_loss = criterion(output, labels)
            penalized_loss = original_loss
            penalty = get_l1_loss(model)
            if penalty is not None:
                penalized_loss += float(lamb) * penalty
            # print(tn(original_loss.data), tn(penalized_loss.data), tn(penalty.data))
            optimizer.zero_grad()
            penalized_loss.backward()
            optimizer.step()
            if stats is not None:
                capacities = get_capacities(model)
                for layer, c in enumerate(capacities):
                    stats.log('capacity_l%s' % layer, c)
                stats.log('capacity', sum(capacities))
                stats.log('batch_original_loss', tn(original_loss.data))
                stats.log('batch_l1_penalty', tn(penalty.data))
                stats.log('batch_loss', tn(penalized_loss.data))
        if mode == 'classification':
            acc = (output.max(1)[1] == labels).float().sum()
        else:
            acc = -torch.nn.functional.mse_loss(output, labels, size_average=False)
        if stats is not None and optimizer is not None:
            stats.log('batch_acc', tn(acc.data) / images.size(0))
        tot += len(labels)
        acc_sum += acc

    try:
        acc = tn(acc_sum.data / tot)
    except:
        acc = acc_sum / tot
    return acc


def train_until_convergence(model, train_dl, val_dl, lamb, patience=3, min_epochs=2, mode='classification'):
    best_model = None
    best_score = (-np.inf, 0)
    early_stop = 0

    print('lambda', lamb)

    optimizer = Adam(model.parameters())
    set_optimizer(model, optimizer)

    while True:
        train_acc = forward(model, train_dl, lamb, optimizer, mode=mode)
        current_accuracy = forward(model, val_dl, mode=mode)
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
            test_acc = forward(best_model, test_dl, mode=mode)
            print('TEST ACC:', test_acc)
            test_accuracies.append(test_acc)
        else:
            print('did not improve')
        lamb /= lamb_decay
    return val_accuracies, test_accuracies, best_model

def generalization_gap(model, dl1, dl2):
    return forward(model, dl1) - forward(model, dl2)

def simple_train(model, lr, dl, dl2, o, m):
    accs = []
    taccs = []
    opt = SGD(get_block_paramteters(model), lr=lr, momentum=m, nesterov=True)
    try:
        while True:
            accs.append(forward(model, dl, 0, opt, mode='regression'))
            taccs.append(forward(model, dl, mode='regression'))
            print(accs[-1], taccs[-1])
    except:
        return accs, taccs

def compress_train(model, dl, dl2, dl3, lamb, lamb_decay=2**(1/10), weight_decay=1e-6, max_time=5, mode='classification', weight=None):
    opt = Adam(model.parameters(), weight_decay=weight_decay)
    stats = TrainingStats()
    set_optimizer(model, opt)
    try:
        start = time()
        while time() - start < max_time * 60:
            stats.next_epoch()
            stats.log('lambda', lamb)
            with stats.time('train_epoch'):
                a = forward(model, dl, lamb, opt, stats=stats, mode=mode, weight=weight)
                stats.log('mean_train_acc', a)
            if has_collapsed(model):
                break
            with stats.time('gc'):
                garbage_collect(model)
                model()
            with stats.time('eval_val'):
                b = forward(model, dl2, mode=mode, weight=weight)
                stats.log('val_acc', b)
            with stats.time('eval_test'):
                c = forward(model, dl3, mode=mode, weight=weight)
                stats.log('test_acc', c)
            lamb /= lamb_decay
            # print(a, b, c, get_capacities(model))
    except:
        raise
    return stats
