import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy

from models.MNIST_1h import MNIST_1h

REPLICATES = 11

transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
])

training_dataset = MNIST(
    './datasets/MNIST/',
    train=True,
    download=True,
    transform=transform
)
training_dataloader = DataLoader(
    training_dataset,
    batch_size=128,
    pin_memory=True,
    shuffle=True)

testing_dataset = MNIST(
    './datasets/MNIST/',
    train=False,
    download=True,
    transform=transform
)
testing_dataloader = DataLoader(
    testing_dataset,
    batch_size=128,
    pin_memory=True,
    shuffle=True)

wrap = lambda x: x.cuda(async=True) if torch.is_tensor(x) and x.is_pinned() else x.cuda()
unwrap = lambda x: x.cpu()

def init_models(count=REPLICATES):
    return [wrap(MNIST_1h(int(2**(i / 2)))) for i in range(7, 28) for _ in range(count)]

def save_model(model, id):
    with open('/tmp/model-%s.data' % id, 'wb+') as f:
        torch.save(model, f)

def load_model(id):
    with open('/tmp/model-%s.data' % id, 'rb') as f:
        return torch.load(f)

def train(models):
    criterion = nn.CrossEntropyLoss()
    optimizers = [Adam(model.parameters()) for model in models]

    for e in range(0, 15):
        print("Epoch %s" % e)
        for i, (images, labels) in enumerate(training_dataloader):
            print(round(i / len(training_dataloader) * 100))
            images = wrap(Variable(images, requires_grad=False))
            labels = wrap(Variable(labels, requires_grad=False))
            for model, optimizer in zip(models, optimizers):
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


def get_activations(models, loader=training_dataloader):
    results = []
    for i, model in enumerate(reversed(models)):
        model = wrap(model)
        print('Model ', i)
        outputs = None
        for images, labels in loader:
            images = wrap(Variable(images, volatile=True))
            b = model.partial_forward(images)
            outputs = b if outputs is None else torch.cat([outputs, b], 0)
        model = None
        results.append(unwrap(outputs.std(0).data).numpy())
    return results


def plot_distributions(activations):
    to_plot = [1, 3,  5, 9, 11, 13, 15]
    plt.figure(figsize=(10, 5))
    for i in reversed(sorted(to_plot)):
        sns.distplot(activations[i], hist=False, label="%s neurons" % (len(activations[i]) / REPLICATES ))
    plt.xlim((0, 8.5))
    plt.xlabel('Standard deviation (unitless)')
    plt.ylabel('Density')
    plt.title('Distribution of standard deviation of activation after hidden layer')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plots/MNIST_1h_dist_activations.png')
    plt.close()


def plot_sum_variance(activations):
    sizes = np.array([len(x) / REPLICATES for x in activations])
    sums = np.array([x.sum() for x in activations])
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, sums)
    plt.title('Sum of variance depending on the size of the layer')
    plt.xlabel('Number of neurons')
    plt.ylabel('Sum of variance')
    plt.tight_layout()
    plt.savefig('./plots/MNIST_1h_sum_variance.png')
    plt.close()


def plot_shapiro(activations):
    sizes = np.array([len(x) / REPLICATES for x in activations])
    t_values = np.array([scipy.stats.shapiro(x)[0] for x in activations])
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, t_values)
    plt.title('Results of normality tests(Shapiro) for different layer size')
    plt.xlabel('Number of neurons')
    plt.ylabel('t_value')
    argmax = sizes[t_values.argmax()]
    plt.axvline(x=argmax, color='C1')
    plt.xticks(list(plt.xticks()[0]) + [argmax])
    plt.xlim(0, 10000)
    plt.ylim((0, 1))
    plt.tight_layout()
    plt.savefig('./plots/MNIST_1h_normality_test.png')
    plt.close()


def plot_distributions_around_sweet(activations):
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
    plt.title('Distribution of standard deviation of activation after hidden layer around the most normal')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plots/MNIST_1h_dist_activations_around_sweet.png')
    plt.close()


def plot_dead_neurons(activations):
    sizes = np.array([len(x) / REPLICATES for x in activations])
    deads = np.array([(x == 0).sum() for x in activations])
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, deads)
    plt.title('Evolution of the quantity of dead neurons')
    plt.xlabel('Number of neurons')
    plt.ylabel('Number of dead neurons')
    plt.tight_layout()
    plt.savefig('./plots/MNIST_1h_dead_neurons.png')
    plt.close()

def get_accuracy(models, loader=training_dataloader):
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


def plot_compare_shapiro_accuracy(activations, accuracies):
    sizes = np.array([len(x) / REPLICATES for x in activations])
    t_values = np.array([scipy.stats.shapiro(x)[0] for x in activations])
    plt.figure(figsize=(10, 5))
    a = plt.gca()
    b = a.twinx()
    a.plot(sizes, t_values, color='C0')
    b.plot(sizes, accuracies, color='C1')
    a.set_ylabel('t_value (shapiro test)')
    b.set_ylabel('accuracy')
    plt.xlabel('Number of neurons')
    plt.title('Comparison between normality test and measured accuracy')
    plt.savefig('./plots/MNIST_1h_acc_vs_shapiro.png')
    plt.close()


if False and __name__ == '__main__':
    models = init_models()
    train(models)
    for i, model in enumerate(models):
        save_model(models, i)
    models = [unwrap(model) for model in models] # Freeing some GPU memory
    activations = np.array(get_activations(models)).reshape(-1, REPLICATES)
    activations = [np.concatenate(a) for a in activations]
    accuracies = get_accuracy(models).reshape(-1, REPLICATES).mean(axis=1) 
    plot_distributions(activations)
    plot_shapiro(activations)
    plot_sum_variance(activations)
    plot_distributions_around_sweet(activations)
    plot_dead_neurons(activations)
    plot_compare_shapiro_accuracy(activations, accuracies)
