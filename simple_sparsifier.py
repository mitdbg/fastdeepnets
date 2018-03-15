import torch
import copy
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torchvision.datasets import MNIST, FashionMNIST
from collections import defaultdict
from io import BytesIO

import numpy as np
import  matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from models.MNIST_1h_sparsifier import MNIST_1h_sparsifier
from utils.wrapping import wrap, unwrap
from utils.MNIST import get_dl
from utils.misc import tn
from matplotlib import rc, use
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


EPOCHS = 15
fig_size = (10, 6)
ext = 'pdf'
with_title = False

def save_fig(file_name):
    file_name = file_name.replace(".png", "")
    plt.savefig(file_name + '.' + ext, bbox_inches='tight', pad_inches=0)

def train(models, dl, dl2, lamb=0.001, epochs=EPOCHS, l2_penalty=0.01, pre_out=None):
    try:
        lamb[len(models) - 1]
    except TypeError:
        lamb = [lamb] * len(models)
    criterion = CrossEntropyLoss()
    optimizers = []
    for model in models:
        optimizer = Adam([{
            'params': model.parameters(),
            'weight_decay': l2_penalty,
        }])
        optimizers.append(optimizer)
    sizes = []
    losses = []
    taccuracies = []
    accuracies = []
    stopped = [False] * len(models)
    best = [np.inf] * len(models)
    for e in range(0, epochs):
        print("Epoch %s" % e)
        gradient = np.zeros(len(models))
        los = np.zeros(len(models))
        accs = np.zeros(len(models))
        taccs = np.zeros(len(models))
        for i, (images, labels) in enumerate(dl):
            images = wrap(Variable(images, requires_grad=False))
            labels = wrap(Variable(labels, requires_grad=False))
            for mid, (model, optimizer) in enumerate(zip(models, optimizers)):
                output = model(images)
                if pre_out is not None:
                    output += Variable(pre_out(images).data, requires_grad=False)
                optimizer.zero_grad()
                l = criterion(output, labels)
                l2 = l + float(lamb[mid]) * model.loss()
                l2.backward()
                acc = (output.max(1)[1] == labels).float().sum()
                los[mid] += tn(l.data) # Save the loss without the penalty
                accs[mid] += tn(acc.data)
                optimizer.step()
        for i, (images, labels) in enumerate(dl2):
            images = wrap(Variable(images, requires_grad=False))
            labels = wrap(Variable(labels, requires_grad=False))
            for mid, (model, optimizer) in enumerate(zip(models, optimizers)):
                output = model(images)
                if pre_out is not None:
                    output += Variable(pre_out(images).data, requires_grad=False)
                acc = (output.max(1)[1] == labels).float().sum()
                taccs[mid] += tn(acc.data)
        losses.append(los)
        accuracies.append(accs)
        taccuracies.append(taccs)
        sizes.append([tn(m.l0_loss().data) for m in models])
    total_samples = len(dl.dataset)
    total_samples2 = len(dl2.dataset)
    return np.stack(sizes), np.stack(losses) / total_samples, np.stack(accuracies) / total_samples, np.stack(taccuracies) / total_samples2

def evaluate_neuron_importance(model, dl):
    def eval_loss():
        total_loss = 0
        criterion = CrossEntropyLoss()
        for images, labels in dl:
            images = wrap(Variable(images, requires_grad=False))
            labels = wrap(Variable(labels, requires_grad=False))
            output = model(images)
            total_loss += tn(criterion(output, labels).data)
        return total_loss

    current_loss = eval_loss()
    loss_differences = defaultdict(int)
    for i, value in enumerate(model.filter.data.cpu().numpy().tolist()):
        model.filter.data.index_fill_(0, (torch.ones(1) * i).long().cuda(), 0)
        loss_differences[i] = current_loss - eval_loss()
        model.filter.data.index_fill_(0, (torch.ones(1) * i).long().cuda(), value)
    return loss_differences


def plot_training(lambdas, training_accuracy, testing_accuracy, sizes, prefix, epochs):
    order = np.argsort(lambdas)
    fig, (a, b) = plt.subplots(1, 2)
    if with_title:
        fig.suptitle("Training process with different penalties", fontsize=14)
    b.set_xlabel('Epoch')
    a.set_xlabel('Epoch')
    a.set_ylabel('Accuracy (\\%)')
    b.set_ylabel('Network Size')
    a.minorticks_on()
    a.yaxis.set_minor_locator(MultipleLocator(1))
    a.yaxis.set_major_locator(MultipleLocator(5))
    a.yaxis.grid(b=True, which='major', linestyle='-')
    a.yaxis.grid(b=True, which='minor', alpha=0.4, linestyle='--')
    b.yaxis.set_minor_locator(MultipleLocator(10))
    b.yaxis.set_major_locator(MultipleLocator(100))
    b.yaxis.grid(b=True, which='major', linestyle='-')
    b.yaxis.grid(b=True, which='minor', alpha=0.4, linestyle='--')
    b.minorticks_on()
    a.set_xlim(xmin=0, xmax=epochs-1)
    b.set_xlim(xmin=0, xmax=epochs-1)
    fig.set_size_inches(fig_size)
    for index in order:
        c = 'C%s'%index
        a.plot(testing_accuracy[index]*100, color=c, linewidth=3)
        a.plot(training_accuracy[index]*100, color=c, linestyle=':', linewidth=3)
        b.plot(sizes[index], color=c, linewidth=3, label='%s penalty' % lambdas[index])
    b.legend(loc='center right')
    training_artist = plt.Line2D((0,1),(0,0), color='k', linestyle=':')
    testing_artist = plt.Line2D((0,1),(0,0), color='k', linestyle='-')
    a.legend([training_artist, testing_artist], ['Training', 'Testing'])
    save_fig('./plots/%s_1h_training_strict_sparsifier.png' % prefix)
    plt.close()

def plot_end(lambdas, training_accuracy, testing_accuracy, sizes, prefix, epochs):
    final_training_acc = training_accuracy[:, -1]
    final_testing_acc = testing_accuracy[:, -1]
    final_sizes = sizes[:, -1]
    bar_indices = [len(lambdas) - 1] + list(range(len(lambdas) - 1))
    plt.figure(figsize=fig_size)
    a = plt.gca()
    a.bar(bar_indices, 100 * final_training_acc, 0.65, color='C0', alpha=0.6, label='Training accuracy')
    a.bar(bar_indices, 100 * final_testing_acc, 0.85, color='C2', label='Testing accuracy')
    a.minorticks_on()
    labels_text = list(map(str, lambdas))
    labels_text = labels_text[1:] + labels_text[:1]
    plt.xticks(np.arange(len(lambdas)), labels_text)
    a.yaxis.set_minor_locator(MultipleLocator(1))
    a.yaxis.set_major_locator(MultipleLocator(5))
    a.yaxis.grid(b=True, which='major', linestyle='-')
    a.yaxis.grid(b=True, which='minor', alpha=0.4, linestyle='--')
    a.set_ylabel('Accuracy (%)')
    a.set_xlabel('Size penalty')
    a.set_axisbelow(True)
    if with_title:
        plt.title('%s - Statistics after training %s epochs' % (prefix, epochs))
    b = a.twinx()
    b.set_ylim(ymin=0, ymax=final_sizes.max()* 1.1)
    a.set_ylim(ymin=0, ymax=100)
    b.set_ylabel('Number of neurons used')
    order = np.argsort(bar_indices)
    b.plot(np.array(bar_indices)[order], final_sizes[order], color='C1', linewidth=4, marker='o', markerfacecolor='black', markersize=10, mew=2)
    a.legend(loc='lower right')
    plt.tight_layout()
    save_fig('./plots/%s_1h_stats_strict_sparsifier.png' % prefix)
    plt.close()
    return final_training_acc

def simple_train(model, dl, dl2, lamb=0.001, pre_out=None):
    print(lamb)
    total_samples = len(dl.dataset)
    total_samples2 = len(dl2.dataset)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    sizes = []
    losses = []
    taccuracies = []
    accuracies = []
    best = (-np.inf, -np.inf)
    bn = None
    patience = 1
    def go(images):
        output = model(images)
        if pre_out is not None:
            output += Variable(pre_out(images).data, requires_grad=False)
        return output

    while True:
        print('epoch')
        los = 0
        accs = 0
        taccs = 0
        for i, (images, labels) in enumerate(dl):
            images = wrap(Variable(images, requires_grad=False))
            labels = wrap(Variable(labels, requires_grad=False))
            output = go(images)
            optimizer.zero_grad()
            l = criterion(output, labels)
            l2 = l + float(lamb) * model.loss()
            l2.backward()
            accs += tn((output.max(1)[1] == labels).float().sum().data)
            los += tn(l.data) # Save the loss without the penalty
            optimizer.step()
        for i, (images, labels) in enumerate(dl2):
            images = wrap(Variable(images, requires_grad=False))
            labels = wrap(Variable(labels, requires_grad=False))
            output = go(images)
            taccs += tn((output.max(1)[1] == labels).float().sum().data)
        losses.append(los)
        accuracies.append(accs)
        taccuracies.append(taccs / total_samples2)
        sizes.append(tn(model.l0_loss().data))
        next_score = (-sizes[-1], taccuracies[-1])
        if best < next_score:
            best = next_score
            bm = copy.deepcopy(model)
            patience = 1
        else:
            patience += 1
            if patience >= 3:
                break
    return bm, best[1], np.stack(sizes), np.stack(losses) / total_samples, np.stack(accuracies) / total_samples, np.stack(taccuracies)

def train_algo(model_gen, ds, l=1, size=50, f=10):
    models = []
    dl1 = get_dl(ds, True)
    dl2 = get_dl(ds, False)
    gbs = 0
    l *= f
    def preout(x):
        values = [m(x) for m in models]
        return sum(values[1:], values[0])
    while l > 1e-9:
        l /= f
        model = model_gen()
        pr = preout if len(models) > 0 else None
        bm, bs, sizes, losses, accs, taccs = simple_train(model, dl1, dl2, lamb=l, pre_out=pr)
        if sizes[-1] == 0 or bs < gbs:
            continue
        else:
            print('temp - best score', bs)
            while True:
                l *= f
                cm, cs, ss, ll, aa, taa = simple_train(bm, dl1, dl2, lamb=l, pre_out=pr)
                if cs < bs:
                    break
                else:
                    bm = cm
                    bs = cs
                    print('temp - best score', bs)
            print('block score')
            if bs > gbs:
                models.append(bm)
                print('current size', sum([tn(m.l0_loss().data) for m in models]))
                gbs = bs
            else:
                return models
    return models

def simple_benchmark(ds, replicas=10, epochs=100, starting_neurons=1000):
    dl = get_dl(ds)
    dl2 = get_dl(ds, False)
    lambdas = np.power(10.0, -np.arange(1, 8))
    lambdas = np.insert(lambdas, 0, 0)
    l = lambdas
    lambdas = np.tile(lambdas, (replicas, 1)).reshape(-1)
    models = [MNIST_1h_sparsifier(starting_neurons).cuda() for _ in lambdas]
    result = train(models, dl, dl2, lamb=lambdas, epochs=epochs, l2_penalty=0)
    result = [x.reshape(epochs, replicas, -1).mean(1).T for x in result]
    plot_training(l, result[2], result[3], result[0], ds.__name__, epochs)
    plot_end(l, result[2], result[3], result[0], ds.__name__, epochs)
    return result

if __name__ == '__main__':
    # simple_benchmark(MNIST, replicas=30, epochs=60)
    simple_benchmark(FashionMNIST, replicas=30, epochs=60)

