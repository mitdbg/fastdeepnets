import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torchvision.datasets import MNIST, FashionMNIST

import numpy as np
import  matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from models.MNIST_1h_sparsifier import MNIST_1h_sparsifier
from utils.wrapping import wrap, unwrap
from utils.MNIST import get_dl
from utils.misc import tn

EPOCHS = 15

def train(models, dl, dl2, lamb=0.001, epochs=EPOCHS, l2_penalty=0.01):
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
                acc = (output.max(1)[1] == labels).float().sum()
                taccs[mid] += tn(acc.data)
        losses.append(los)
        accuracies.append(accs)
        taccuracies.append(taccs)
        sizes.append([tn(m.l0_loss().data) for m in models])
    total_samples = len(dl.dataset)
    total_samples2 = len(dl2.dataset)
    return np.stack(sizes), np.stack(losses) / total_samples, np.stack(accuracies) / total_samples, np.stack(taccuracies) / total_samples2

def plot_training(lambdas, training_accuracy, testing_accuracy, sizes, prefix, epochs):
    order = np.argsort(lambdas)
    fig, (a, b) = plt.subplots(1, 2)
    fig.suptitle("Training process with different penalties", fontsize=14)
    b.set_xlabel('Epoch')
    a.set_xlabel('Epoch')
    a.set_ylabel('Accuracy (%)')
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
    fig.set_size_inches((10, 5))
    for index in order:
        c = 'C%s'%index
        a.plot(testing_accuracy[index]*100, color=c, linewidth=3)
        a.plot(training_accuracy[index]*100, color=c, linestyle=':', linewidth=3)
        b.plot(sizes[index], color=c, linewidth=3, label='%s penalty' % lambdas[index])
    b.legend(loc='center right')
    training_artist = plt.Line2D((0,1),(0,0), color='k', linestyle=':')
    testing_artist = plt.Line2D((0,1),(0,0), color='k', linestyle='-')
    a.legend([training_artist, testing_artist], ['Training', 'Testing'])
    plt.savefig('./plots/%s_1h_training_strict_sparsifier.png' % prefix)
    plt.close()

def plot_end(lambdas, training_accuracy, testing_accuracy, sizes, prefix, epochs):
    final_training_acc = training_accuracy[:, -1]
    final_testing_acc = testing_accuracy[:, -1]
    final_sizes = sizes[:, -1]
    bar_indices = [len(lambdas) - 1] + list(range(len(lambdas) - 1))
    plt.figure(figsize=(10, 5))
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
    plt.title('%s - Statistics after training %s epochs' % (prefix, epochs))
    b = a.twinx()
    b.set_ylim(ymin=0, ymax=final_sizes.max()* 1.1)
    a.set_ylim(ymin=0, ymax=100)
    b.set_ylabel('Number of neurons used')
    order = np.argsort(bar_indices)
    b.plot(np.array(bar_indices)[order], final_sizes[order], color='C1', linewidth=4, marker='o', markerfacecolor='black', markersize=10, mew=2)
    a.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('./plots/%s_1h_stats_strict_sparsifier.png' % prefix)
    plt.close()
    return final_training_acc

def simple_benchmark(ds, replicas=10, epochs=100):
    dl = get_dl(ds)
    dl2 = get_dl(ds, False)
    lambdas = np.power(10.0, -np.arange(0, 5))
    lambdas = np.insert(lambdas, 0, 0)
    l = lambdas
    lambdas = np.tile(lambdas, (replicas, 1)).reshape(-1)
    models = [MNIST_1h_sparsifier(500).cuda() for _ in lambdas]
    result = train(models, dl, dl2, lamb=lambdas, epochs=epochs, l2_penalty=0)
    result = [x.reshape(epochs, replicas, -1).mean(1).T for x in result]
    plot_training(l, result[2], result[3], result[0], ds.__name__, epochs)
    plot_end(l, result[2], result[3], result[0], ds.__name__, epochs)
    return result

if __name__ == '__main__':
    print('hello how are you')
    simple_benchmark(MNIST, replicas=30, epochs=60)
    simple_benchmark(FashionMNIST, replicas=30, epochs=60)

