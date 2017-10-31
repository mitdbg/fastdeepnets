import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy
from algorithms.exp_norm_mixture_fit import fit as fit_exp_norm
from algorithms.digamma_mixture_fit import fit as fit_digamma

from models.MNIST_1h import MNIST_1h

REPLICATES = 11

transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
])


wrap = lambda x: x.cuda(async=True) if torch.is_tensor(x) and x.is_pinned() else x.cuda()
unwrap = lambda x: x.cpu()

def init_models(count=REPLICATES):
    return [wrap(MNIST_1h(int(2**(i / 2)))) for i in range(8, 28) for _ in range(count)]

def save_model(model, id):
    with open('/tmp/model-%s.data' % id, 'wb+') as f:
        torch.save(model, f)

def load_model(id):
    with open('/tmp/model-%s.data' % id, 'rb') as f:
        return torch.load(f)

def train(models, dl, epochs=30):
    criterion = nn.CrossEntropyLoss()
    optimizers = [Adam(model.parameters()) for model in models]

    for e in range(0, epochs):
        print("Epoch %s" % e)
        for i, (images, labels) in enumerate(dl):
            # print(round(i / len(dl) * 100))
            images = wrap(Variable(images, requires_grad=False))
            labels = wrap(Variable(labels, requires_grad=False))
            for model, optimizer in zip(models, optimizers):
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


# There is DEFINITELY a better online batched estimator for the variance
# But this one is quite efficient and quite stable (no square of sums)
# I will improve it for sure when I have more time
# At least this is good enough to get me started
def get_activations(models, loader, wrap=wrap):
    count = 0
    sums = [wrap(torch.zeros(m.hidden_layer.out_features)) for m in models]
    sums_diff = [wrap(torch.zeros(m.hidden_layer.out_features)) for m in models]
    for images, labels in loader:
        images = wrap(Variable(images, volatile=True))
        bs = images.size(0)
        count += bs
        for i, model in enumerate(models):
            b = model.partial_forward(images).data
            sums[i] += b.sum(0)
            running_mean = sums[i] / count
            diff = b - running_mean.unsqueeze(0).expand(bs, sums_diff[i].size()[0])
            sums_diff[i] += (diff * diff).sum(0)
    return [torch.sqrt(x / count).cpu().numpy() for x in sums_diff]


def plot_distributions(activations, prefix):
    to_plot = [1, 3,  5, 9, 11, 13, 15]
    plt.figure(figsize=(10, 5))
    for i in reversed(sorted(to_plot)):
        sns.distplot(activations[i], hist=False, label="%s neurons" % (len(activations[i]) / REPLICATES ))
    plt.xlim((0, 8.5))
    plt.xlabel('Standard deviation (unitless)')
    plt.ylabel('Density')
    plt.title(prefix +' - Distribution of standard deviation of activation after hidden layer')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plots/%s_1h_dist_activations.png' % prefix)
    plt.close()


def plot_sum_variance(activations, prefix):
    sizes = np.array([len(x) / REPLICATES for x in activations])
    sums = np.array([x.sum() for x in activations])
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, sums)
    plt.title(prefix +' - Sum of variance depending on the size of the layer')
    plt.xlabel('Number of neurons')
    plt.ylabel('Sum of variance')
    plt.tight_layout()
    plt.savefig('./plots/%s_1h_sum_variance.png' % prefix)
    plt.close()


def plot_shapiro(activations, prefix):
    sizes = np.array([len(x) / REPLICATES for x in activations])
    t_values = np.array([scipy.stats.shapiro(x)[0] for x in activations])
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, t_values)
    plt.title(prefix +' - Results of normality tests(Shapiro) for different layer size')
    plt.xlabel('Number of neurons')
    plt.ylabel('t_value')
    argmax = sizes[t_values.argmax()]
    plt.axvline(x=argmax, color='C1')
    plt.xticks(list(plt.xticks()[0]) + [argmax])
    plt.xlim(0, 10000)
    plt.ylim((0, 1))
    plt.tight_layout()
    plt.savefig('./plots/%s_1h_normality_test.png' % prefix)
    plt.close()


def plot_distributions_around_sweet(activations, prefix):
    t_values = np.array([scipy.stats.shapiro(x)[0] for x in activations])
    argmax = t_values.argmax()
    to_plot = [argmax - 1, argmax, argmax + 1]
    plt.figure(figsize=(10, 5))
    for i in reversed(sorted(to_plot)):
        sns.distplot(activations[i], hist=False, label="%s neurons" % (len(activations[i]) / REPLICATES))
    plt.xlim((0, 5))
    plt.ylim((0, 0.5))
    plt.xlabel('Standard deviation (unitless)')
    plt.ylabel('Density')
    plt.title(prefix +' - Distribution of standard deviation of activation after hidden layer around the most normal')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plots/%s_1h_dist_activations_around_sweet.png' % prefix)
    plt.close()


def plot_dead_neurons(activations, prefix):
    sizes = np.array([len(x) / REPLICATES for x in activations])
    deads = np.array([(x == 0).sum() for x in activations])
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, deads)
    plt.title(prefix +' - Evolution of the quantity of dead neurons')
    plt.xlabel('Number of neurons')
    plt.ylabel('Number of dead neurons')
    plt.tight_layout()
    plt.savefig('./plots/%s_1h_dead_neurons.png' % prefix)
    plt.close()

def get_accuracy(models, loader):
    models = [wrap(model) for model in models]
    accs = [0] * len(models)
    for images, labels in loader:
        images = wrap(Variable(images, volatile=True))
        labels = wrap(labels)
        for i, model in enumerate(models):
            predicted = model(images).data
            acc = (predicted.max(1)[1] == labels).float().mean()
            accs[i] += acc
    return np.array(accs) / len(loader)


def plot_compare_shapiro_accuracy(activations, accuracies, prefix):
    sizes = np.array([len(x) / REPLICATES for x in activations])
    t_values = np.array([scipy.stats.shapiro(x)[0] for x in activations])
    plt.figure(figsize=(10, 5))
    a = plt.gca()
    b = a.twinx()
    a.plot(sizes, t_values, color='C0')
    b.plot(sizes, accuracies, color='C1')
    a.set_ylabel('t_value (shapiro test)')
    b.set_ylabel('accuracy')
    plt.xscale('log', basex=2)
    a.set_xlabel('Number of neurons')
    plt.title(prefix +' - Comparison between normality test and measured accuracy')
    plt.tight_layout()
    plt.savefig('./plots/%s_1h_acc_vs_shapiro.png' % prefix)
    plt.close()

def plot_mixture_ratio(activations, accuracies, prefix):
    n_acc = (accuracies - accuracies.min()) / (accuracies.max() - accuracies.min())
    z_activations = [(a == 0).sum() / REPLICATES for a in activations]
    nz_activations = [a[a != 0] for a in activations]
    sizes = np.array([len(x) / REPLICATES for x in activations])
    digamma_ratio = np.array([fit_digamma(x)[1] for x in nz_activations])
    exp_norm_ratio = np.array([fit_exp_norm(x)[1] for x in nz_activations])
    t_values = np.array([scipy.stats.shapiro(x)[0] for x in nz_activations])
    plt.figure(figsize=(10, 5))
    a = plt.gca()
    b = a.twinx()
    a.plot(sizes, digamma_ratio, color='C0', label='Digamma fitting')
    a.plot(sizes, exp_norm_ratio, color='C1', label='Exp+Normal fitting')
    a.plot(sizes, t_values, color='C2', label='Shapiro normality test')
    a.plot(sizes, n_acc, color='C4', label='Normalized accuracy')
    a.axhline(y = 0.025, color='C5', label='Arbitrary suggested threshold (p=0.05)')
    a.legend(loc='lower right')
    b.plot(sizes, z_activations, color='C3', label='Dead neurons')
    a.set_ylabel('Proportions')
    b.set_ylabel('Number of dead neurons')
    b.legend(loc='upper right')
    plt.xscale('log', basex=2)
    b.set_ylim((0, 20))
    a.set_xlabel('Number of neurons')
    plt.title(prefix +' - Comparison between multiple metrics')
    plt.savefig('./plots/%s_1h_acc_vs_mixtures.png' % prefix)
    plt.close()

def benchmark(dataset, prefix):
    dl = DataLoader(
        dataset(
            './datasets/%s/' % prefix,
            train=True,
            download=True,
            transform=transform),
        batch_size=128,
        pin_memory=True,
        shuffle=True
    )
    models = init_models()
    train(models, dl)
    activations = np.array(get_activations(models, dl)).reshape(-1, REPLICATES)
    activations = [np.concatenate(a) for a in activations]
    accuracies = get_accuracy(models, dl).reshape(-1, REPLICATES).mean(axis=1) 
    plot_distributions(activations, prefix)
    plot_shapiro(activations, prefix)
    plot_sum_variance(activations, prefix)
    plot_distributions_around_sweet(activations, prefix)
    plot_dead_neurons(activations, prefix)
    plot_compare_shapiro_accuracy(activations, accuracies, prefix)
    plot_mixture_ratio(activations, accuracies, prefix)

if __name__ == '__main__':
    benchmark(MNIST, 'MNIST')
    benchmark(FashionMNIST, 'FashionMNIST')
