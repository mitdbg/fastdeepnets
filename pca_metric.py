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
from sklearn.decomposition import PCA
from scipy.linalg import inv, norm
from algorithms.exp_norm_mixture_fit import fit as fit_exp_norm
from algorithms.digamma_mixture_fit import fit as fit_digamma

from models.MNIST_1h import MNIST_1h

REPLICATES = 11

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

def init_models(count=REPLICATES):
    return [wrap(MNIST_1h(s)) for s in [32, 64, 128, 200, 350, 568, 1024, 2048]]

def save_model(model, id):
    with open('/tmp/model-%s.data' % id, 'wb+') as f:
        torch.save(model, f)

def load_model(id):
    with open('/tmp/model-%s.data' % id, 'rb') as f:
        return torch.load(f)

def train(models, dl):
    criterion = nn.CrossEntropyLoss()
    optimizers = [Adam(model.parameters()) for model in models]

    for e in range(0, 10):
        print("Epoch %s" % e)
        for i, (images, labels) in enumerate(dl):
            print(round(i / len(dl) * 100))
            images = wrap(Variable(images, requires_grad=False))
            labels = wrap(Variable(labels, requires_grad=False))
            for model, optimizer in zip(models, optimizers):
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

def get_activations(model, loader):
    outputs = []
    for images, labels in loader:
        images = wrap(Variable(images, volatile=True))
        outputs.append(model.partial_forward(images, False).data)
    result = torch.cat(outputs, 0)
    return result - result.mean(0)

def simplify_all(values, basis):
    D = torch.inverse(basis).mm(values.transpose(0, 1))
    result = D.clone().zero_()
    yield result.transpose(0, 1)
    for i in range(basis.size(0)):
        basis_col = basis[:, i].unsqueeze(1)
        D_lines = D[i].unsqueeze(0)
        result += basis_col.mm(D_lines)
        yield result.transpose(0, 1)

def score_simplification(values, simplified_values):
    return (values - simplified_values).pow(2).sum(1).mean()

def get_distances(activations, pcas):
    results = []
    for a, pca in zip(activations, pcas):
        errs = []
        for reconstructed in simplify_all(a, wrap(torch.from_numpy(pca.components_))):
            errs.append(score_simplification(a, reconstructed))
        results.append(np.array(errs))
    return results

def get_dl(dataset, prefix):
    return DataLoader(
        dataset(
            './datasets/%s/' % prefix,
            train=True,
            download=True,
            transform=transform),
        batch_size=128,
        pin_memory=torch.cuda.device_count() > 0,
        shuffle=True
    )

def get_pca(activations):
    pca = PCA()
    pca.fit(unwrap(activations).numpy())
    return pca

def plot_variance(variances, prefix):
    plt.figure(figsize=(10, 5))
    plt.title('%s - Explained variance per component for multiple models' % prefix)
    for x in variances:
        plt.plot(x, label=('%s neurons' % len(x)))
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.xlabel('xth component (sorted)')
    plt.ylabel('Explained variance')
    plt.ylim((1e-13, 1e4))
    plt.savefig('./plots/%s_1h_pca_explained_variance.png' % prefix)
    plt.close()

def plot_distances(distances, prefix):
    plt.figure(figsize=(10, 5))
    plt.title('%s - Square distance of PCA reconstruction' % prefix)
    for x in distances:
        plt.plot(x, label=('%s neurons' % len(x)))
    plt.legend()
    plt.xlabel('Number of components kept')
    plt.ylabel('Mean distance (L2)')
    plt.savefig('./plots/%s_1h_pca_reconstruction_distance.png' % prefix)
    plt.close()


def pipeline(ds, prefix):
    dl = get_dl(ds, prefix)
    models = init_models()
    train(models, dl)
    activations = [get_activations(model, dl) for model in models]
    pcas = [get_pca(x) for x in activations]
    variances = [x.explained_variance_ for x in pcas]
    distances = get_distances(activations, pcas)
    plot_distances(distances, prefix)
    plot_variance(variances, prefix)
    return distances

if __name__ == '__main__':
    pipeline(MNIST, 'MNIST')
    pipeline(FashionMNIST, 'FashionMNIST')
