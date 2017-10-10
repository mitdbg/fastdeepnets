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
    batch_size=64,
    shuffle=True)

testing_dataset = MNIST(
    './datasets/MNIST/',
    train=False,
    download=True,
    transform=transform
)
testing_dataloader = DataLoader(
    testing_dataset,
    batch_size=64,
    shuffle=True)

def init_models():
    return [MNIST_1h(int(2**(i / 2))) for i in range(7, 29)]


def train(models):
    criterion = nn.CrossEntropyLoss()
    optimizers = [Adam(model.parameters()) for model in models]

    for e in range(0, 15):
        print("Epoch %s" % e)
        for i, (images, labels) in enumerate(training_dataloader):
            print(round(i / len(training_dataloader) * 100))
            images = Variable(images, requires_grad=False)
            labels = Variable(labels, requires_grad=False)
            for model, optimizer in zip(models, optimizers):
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


def get_activations(model, loader=training_dataloader):
    outputs = []
    for images, labels in loader:
        images = Variable(images, volatile=True)
        outputs.append(model.partial_forward(images))
    return torch.cat(outputs, 0).std(0).data.numpy()


def plot_distributions(activations):
    to_plot = [1, 3,  5, 9, 11, 13]
    plt.figure(figsize=(10, 5))
    for i in reversed(sorted(to_plot)):
        sns.distplot(activations[i], hist=False, label="%s neurons" % len(activations[i]))
    plt.xlim((0, 8.5))
    plt.xlabel('Standard deviation (unitless)')
    plt.ylabel('Density')
    plt.title('Distribution of standard deviation of activation after hidden layer')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plots/MNIST_1h_dist_activations.png')
    plt.close()


def plot_sum_variance(activations):
    sizes = np.array([len(x) for x in activations])
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
    sizes = np.array([len(x) for x in activations])
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


def plot_distributions_arround_sweet(activations):
    t_values = np.array([scipy.stats.shapiro(x)[0] for x in activations])
    argmax = t_values.argmax()
    to_plot = [argmax - 1, argmax, argmax + 1]
    plt.figure(figsize=(10, 5))
    for i in reversed(sorted(to_plot)):
        sns.distplot(activations[i], hist=False, label="%s neurons" % len(activations[i]))
    plt.xlim((0, 5))
    plt.ylim((0, 0.5))
    plt.xlabel('Standard deviation (unitless)')
    plt.ylabel('Density')
    plt.title('Distribution of standard deviation of activation after hidden layer arround the most normal')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plots/MNIST_1h_dist_activations_arround_sweet.png')
    plt.close()


def plot_dead_neurons(activations):
    sizes = np.array([len(x) for x in activations])
    deads = np.array([(x == 0).sum() for x in activations])
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, deads)
    plt.title('Evolution of the quantity of dead neurons')
    plt.xlabel('Number of neurons')
    plt.ylabel('Number of dead neurons')
    plt.tight_layout()
    plt.savefig('./plots/MNIST_1h_dead_neurons.png')
    plt.close()

def get_accuracy(model, loader=training_dataloader):
    accs = []
    for images, labels in loader:
        images = Variable(images, volatile=True)
        predicted = model(images).data
        accs.append((predicted.max(1)[1] == labels).float().mean())
    return np.array(accs).mean()


def plot_compare_shapiro_accuracy(activations, accuracies):
    sizes = np.array([len(x) for x in activations])
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
    plot.tight_layout()
    plt.savefig('./plots/MNIST_1h_acc_vs_shapiro.png')
    plt.close()


if __name__ == '__main__':
    models = init_models()
    train(models)
    activations = [get_activations(model) for model in models]
    accuracies = np.array([get_accuracy(x) for x in models])
    plot_distributions(activations)
    plot_shapiro(activations)
    plot_sum_variance(activations)
    plot_distributions_arround_sweet(activations)
    plot_dead_neurons(activations)
    plot_compare_shapiro_accuracy(activations, accuracies)
