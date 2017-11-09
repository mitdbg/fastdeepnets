from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.multiprocessing import Pool
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy
from algorithms.exp_norm_mixture_fit import fit as fit_exp_norm
from algorithms.digamma_mixture_fit import fit as fit_digamma

from models.MNIST_1h_flexible import MNIST_1h_flexible
from models.MNIST_1h_flexible_sorted import MNIST_1h_flexible_sorted
from models.MNIST_1h_flexible_scaled import MNIST_1h_flexible_scaled
from models.MNIST_1h_flexible_random import MNIST_1h_flexible_random
from models.MNIST_1h import MNIST_1h
from variance_metric import get_activations, train as simple_train


EPOCHS = 15

transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
])

if torch.cuda.device_count() > 0:
    wrap = lambda x: x.cuda(async=True) if torch.is_tensor(x) and x.is_pinned() else x.cuda()
    unwrap = lambda x: x.cpu()
else:
    wrap = lambda x: x
    unwrap = wrap

def tn(x):
    return x.cpu().numpy()[0]

def train(models, dl, lamb=0.001, epochs=EPOCHS, l2_penalty=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizers = []
    for model in models:
        normal_params = set(model.parameters())
        normal_params.remove(model.x_0)

        optimizer = Adam([{
            'params': normal_params,
            'weight_decay': l2_penalty,
        }, {
            'params': [model.x_0],
            'lr': 1,
        }])
        optimizers.append(optimizer)
    gradients = []
    sizes = []
    losses = []
    for e in range(0, epochs):
        print("Epoch %s" % e)
        gradient = np.zeros(len(models))
        los = np.zeros(len(models))
        for i, (images, labels) in enumerate(dl):
            images = wrap(Variable(images, requires_grad=False))
            labels = wrap(Variable(labels, requires_grad=False))
            for mid, (model, optimizer) in enumerate(zip(models, optimizers)):
                output = model(images)
                optimizer.zero_grad()
                l = (criterion(output, labels) + lamb * model.loss())
                l.backward()
                acc = (output.max(1)[1] == labels).float().mean()
                # a = tn(model.x_0.grad.data)
                # if a != a:
                #     return images, labels
                gradient[mid] += tn(model.x_0.grad.data)
                los[mid] += tn(l.data)
                # print(tn(acc.data), tn(model.x_0.data), tn(model.x_0.grad.data))
                optimizer.step()
                if isinstance(model, MNIST_1h_flexible_scaled):
                    model.reorder()
        gradients.append(gradient)
        losses.append(los)
        sizes.append([tn(m.x_0.data) for m in models])
    total_samples = len(dl.dataset)
    return np.stack(sizes), -np.stack(gradients) / total_samples, np.stack(losses) / total_samples

def get_accuracy(models, loader):
    accs = [0] * len(models)
    for images, labels in loader:
        images = wrap(Variable(images, volatile=True))
        labels = wrap(labels)
        for i, model in enumerate(models):
            predicted = model(images).data
            accs[i] += (predicted.max(1)[1] == labels).float().mean()
    return np.array(accs) / len(loader)

def get_dl(dataset, train=True):
    return DataLoader(
        dataset(
            './datasets/%s/' % dataset.__name__,
            train=train,
            download=True,
            transform=transform),
        batch_size=128,
        pin_memory=torch.cuda.device_count() > 0,
        shuffle=True
    )

def plot_convergence(models, sizes, prefix, suffix):
    convergences = np.array([m.x_0.data.cpu().numpy()[0] for m in models])
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, convergences)
    plt.title(prefix +' - Network size after training for different starting sizes')
    plt.xlabel('Number of neurons at the beginning')
    plt.ylabel('Number of neurons at the end')
    plt.tight_layout()
    plt.savefig('./plots/%s_1h_simple_flexible_convergence%s.png' % (prefix, suffix))
    plt.close()

def plot_accuracies(accuracies, sizes, prefix, suffix):
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, accuracies)
    plt.title(prefix +' - Accuracies for different starting sizes')
    plt.xlabel('Number of neurons at the beginning')
    plt.ylabel('Accuracy after training')
    plt.tight_layout()
    plt.savefig('./plots/%s_1h_simple_flexible_accuracies%s.png' % (prefix, suffix))
    plt.close()

def plot_frontier(powers, data, best_acc, prefix, suffix):
    plt.figure(figsize=(10, 5))
    a = plt.gca()
    b = a.twinx()
    valid = data[:, 0] > 0
    data = data[valid, :]
    powers = powers[valid]
    xes = np.arange(0, len(powers)) + 1
    ratios = (data[:, 1] - data[:, 2]) * 100
    ratios2 = (data[:, 1] - best_acc) * 100
    w = 0.35
    a.set_xticks(xes)
    a.set_xticklabels(['1e%s' % x for x in powers] , rotation=70) 
    a.bar(xes - w / 2, ratios, 0.35, label="Loss in accuracy vs flexible of same size", color='C0')
    a.bar(xes + w / 2, ratios2, 0.35,  label="Loss in accuracy vs best model", color='C1')
    a.axhline(y=0, color='k')
    a.grid()
    b.plot(xes, data[:, 0], label="Neuron used", color='C2')
    plt.title(prefix +' - Accuracies for the simple flexible model')
    a.set_xlabel('Network size penalty')
    a.set_ylabel('Loss in accurcy (%)')
    b.set_ylabel('Converged netowrk size')
    b.legend(loc="upper right")
    a.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig('./plots/%s_1h_simple_flexible_frontier%s.png' % (prefix, suffix))
    plt.close()

def get_data(params):
    dl, dl2, w, l2_penalty = params
    models = [wrap(MNIST_1h_flexible(500, wrap, k)) for k in [250] for _ in range(7)]
    train(models, dl, w, EPOCHS, l2_penalty=l2_penalty)
    neurons = np.percentile([m.x_0.data.cpu().numpy()[0] for m in models], 50)
    accuracy = np.percentile(get_accuracy(models, dl2), 50)
    if neurons > 0:
        models2 = [wrap(MNIST_1h(int(neurons))) for _ in range(7)]
        simple_train(models2, dl, EPOCHS)
        accuracy2 = np.percentile(get_accuracy(models2, dl2), 50)
    else:
        accuracy2 = 0
    res = neurons, accuracy, accuracy2
    print(res)
    return res


def validate_plateau_hypothesis2(ds):
    dl = get_dl(ds, False) # Testing because it is smaller, does not change anything
    model = wrap(MNIST_1h_flexible(500, wrap, 250))
    train([model], dl, 0, l2_penalty=0)
    total_weights = unwrap(torch.abs(model.output_layer.weight).sum(0).data).numpy()
    scaler = unwrap(model.get_scaler().data).numpy()
    plt.figure(figsize=(10, 5))
    plt.title('Proof that the l2 penalty is responsible')
    a = plt.gca()
    b = a.twinx()
    b.plot(total_weights, label='Sum of absolute weights associated', color='C0')
    a.plot(scaler, label='Neuron used (Smoothe Indicator function)', color='C1')
    a.legend(loc='upper right')
    b.legend(loc='lower right')
    plt.xlabel('neuron')
    a.set_ylabel('Neuron liveness')
    b.set_ylabel('Sum of abs. weights')
    plt.tight_layout()
    plt.savefig('./plots/%s_1h_plateau_explanation_no_pen.png' % (ds.__name__))
    plt.close()

def validate_plateau_hypothesis(ds):
    dl = get_dl(ds, False) # Testing because it is smaller, does not change anything
    model = wrap(MNIST_1h_flexible(500, wrap, 250))
    train([model], dl, 0, l2_penalty=0.001)
    total_weights = unwrap(torch.abs(model.output_layer.weight).sum(0).data).numpy()
    scaler = unwrap(model.get_scaler().data).numpy()
    plt.figure(figsize=(10, 5))
    plt.title('Proof that the regularization is responsible for the size plateau')
    a = plt.gca()
    b = a.twinx()
    b.plot(total_weights, label='Sum of absolute weights associated', color='C0')
    a.plot(scaler, label='Neuron used (Smoothe Indicator function)', color='C1')
    a.legend(loc='upper right')
    b.legend(loc='lower right')
    plt.xlabel('neuron')
    a.set_ylabel('Neuron liveness')
    b.set_ylabel('Sum of abs. weights')
    plt.tight_layout()
    plt.savefig('./plots/%s_1h_plateau_explanation.png' % (ds.__name__))
    plt.close()

def benchmark_dataset(ds, l2_penalty=0.001, suffix='', test_train=False):
    subpool = Pool(40)
    dl = get_dl(ds, True)
    dl2 = get_dl(ds, test_train)
    sizes = np.array(range(0, 500, 25))
    models = [wrap(MNIST_1h_flexible(500, wrap, k)) for k in range(0, 500, 25)]
    train(models, dl, 1e-5, l2_penalty=l2_penalty)
    accuracies = np.array(get_accuracy(models, dl))
    plot_accuracies(accuracies, sizes, ds.__name__, suffix)
    plot_convergence(models, sizes, ds.__name__, suffix)
    powers = -np.arange(2.5, 8, 0.5)
    weights = 10**powers
    data = np.array([get_data((dl, dl2, x, l2_penalty)) for x in weights.tolist()])
    best_model = wrap(MNIST_1h(1000))
    simple_train([best_model], dl, EPOCHS)
    plot_frontier(powers, data, get_accuracy([best_model], dl2)[0], ds.__name__, suffix)

def plot_convergence_comparison(sizes, gradients, losses, prefix, labels, filename):
    sizes = np.insert(sizes, 0, 0, axis=1)
    epochs = list(range(1, sizes.shape[1]))
    figure = plt.figure(figsize=(10, 15))
    plot_sizes = figure.add_subplot(3, 1, 1)
    plot_sizes.set_title('Evolution of sizes')
    plot_sizes.set_xlabel('Epoch')
    plot_sizes.set_ylabel('Size of the network')
    for i, label in enumerate(labels):
        plot_sizes.plot([0] + epochs, sizes[i], label=label)
    plot_sizes.grid()
    plot_sizes.legend()
    plot_gradients = figure.add_subplot(3, 1, 2)
    plot_gradients.set_title('Evolution of gradients')
    plot_gradients.set_xlabel('Epoch')
    plot_gradients.set_ylabel('Gradient on size')
    for i, label in enumerate(labels):
        plot_gradients.plot(epochs, gradients[i], label=label)
    plot_gradients.grid()
    plot_gradients.legend()
    plot_gradients.set_yscale('log')
    plot_losses = figure.add_subplot(3, 1, 3)
    plot_losses.set_title('Evolution of loss')
    plot_losses.set_xlabel('Epoch')
    plot_losses.set_ylabel('Batch loss')
    for i, label in enumerate(labels):
        plot_losses.plot(epochs, losses[i], label=label)
    plot_losses.grid()
    plot_losses.legend()
    plot_losses.set_yscale('log')
    plt.tight_layout()
    plt.savefig('./plots/%s_1h_%s.png' % (prefix, filename))

def compare_convergence(ds):
    dl = get_dl(ds)
    replicas = 30
    r = range(replicas)
    simple_models = [MNIST_1h_flexible(500, wrap, 0).cuda() for _ in r]
    random_models = [MNIST_1h_flexible_random(500, wrap, 0).cuda() for _ in r]
    all_models = simple_models + random_models
    result = train(all_models, dl, lamb=0, epochs=EPOCHS * 4, l2_penalty=0)
    sizes, gradients, losses = [x.reshape(-1, 2, replicas).mean(axis=2).T for x in result]
    labels = ['Deterministic Model', 'Random Model']
    plot_convergence_comparison(sizes, gradients, losses, ds.__name__, labels, 'deterministic_random_comparison')

def behavior_on_pretrained(ds):
    dl = get_dl(ds)
    replicas = 30
    r = range(replicas)
    pretraining_epochs = 10
    evaluation_epochs = EPOCHS * 4
    half_trained_models = [MNIST_1h_flexible(500, wrap, 500).cuda() for _ in r]
    fully_trained_models = [MNIST_1h_flexible(500, wrap, 500).cuda() for _ in r]
    fresh_models = [MNIST_1h_flexible(500, wrap, 0).cuda() for _ in r]
    train(fully_trained_models, dl, lamb=0, epochs=pretraining_epochs, l2_penalty=0)
    train(half_trained_models, dl, lamb=0, epochs=int(pretraining_epochs / 2), l2_penalty=0)
    for m in half_trained_models + fully_trained_models:
        m.x_0.data.zero_() # We reset everyone to zero neurons
    all_models = fully_trained_models + half_trained_models + fresh_models
    result = train(all_models, dl, lamb=0, epochs=evaluation_epochs, l2_penalty=0)
    sizes, gradients, losses = [x.reshape(-1, 3, replicas).mean(axis=2).T for x in result]
    labels = ['Pretrained %s epochs' % pretraining_epochs,
              'Pretrained %s epochs' % int(pretraining_epochs / 2),
              'Fresh Model']
    plot_convergence_comparison(sizes, gradients, losses, ds.__name__, labels, 'flexible_behavior_on_pretrained')

if __name__ == '__main__':
    # benchmark_dataset(MNIST)
    # benchmark_dataset(FashionMNIST)
    # validate_plateau_hypothesis(MNIST)
    # validate_plateau_hypothesis2(MNIST)
    # benchmark_dataset(MNIST, 0, '_without_penalty')
    # benchmark_dataset(FashionMNIST, 0, '_without_penalty')
    # benchmark_dataset(MNIST, 0, '_without_penalty_training', True)
    # benchmark_dataset(FashionMNIST, 0, '_without_penalty_training', True)
    # compare_convergence(MNIST)
    # compare_convergence(FashionMNIST)
    behavior_on_pretrained(MNIST)
    behavior_on_pretrained(FashionMNIST)
    pass
