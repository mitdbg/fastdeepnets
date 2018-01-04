from matplotlib import rc, use
use('agg')
import torch
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
# for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

nice_boundaries = {
    'MNIST': (94.5, 99),
    'FashionMNIST': (80, 91),
    'CIFAR10': (45, 56)
}

modes = ['static', 'decay', 'shrink']
hatches = ['OO', '...', '////']
tex_labels = ['Fixed', 'Fixed $+ L_2$', 'ShrinkNets']
colors = [(1, 127/255, 14/255, 0.5),
          (31/255, 119/255, 180/255, 0.5),
          (44/255, 160/255, 44/255, 0.5)]

def select_best_experiment(log):
    best_val = log[1].argmax()
    return log[:, best_val]

def load_single_file(filename):
    with open(filename, 'rb') as f:
        params, logs = torch.load(f)
    be = select_best_experiment(logs)
    return be

def load_files(dataset, mode, max_samples=50):
    all_files = glob('./experiments_results/hyper_opt_%s_%s*' % (dataset, mode))
    all_files = all_files[:max_samples]
    assert len(all_files) == max_samples, "Not enough experiments"
    return np.stack([load_single_file(f) for f in all_files])

def sub_plot(ax, dataset, first):
    infos = [load_files(dataset, mode)[:, 2] * 100 for mode in modes]
    result = ax.boxplot(infos, patch_artist=True, widths=0.7)
    boxes = result['boxes']
    for b, h, c in zip(boxes, hatches, colors):
        b.set_hatch(h)
        b.set_facecolor(c)
        b.set_edgecolor('black')
    ax.set_title('\\texttt{%s}' % dataset)
    ax.set_xticks([])
    ax.yaxis.grid(b=True, which='major', alpha=0.4, linestyle='--')
    ax.set_ylim(nice_boundaries[dataset])
    if first:
        ax.legend(handles=boxes , labels=tex_labels ,loc='lower center', bbox_to_anchor=(1.75, 1.09), ncol=3, columnspacing=0.5)
        ax.set_ylabel('Accuracy (\\%)')

f1 = load_files('FashionMNIST', 'static')
f2 = load_files('FashionMNIST', 'shrink')
f3 = load_files('FashionMNIST', 'decay')
f, axes = plt.subplots(1, 3, sharex=True)
plt.subplots_adjust(wspace=0.3, hspace=0)
first = True
for ax, ds in zip(axes, ['MNIST', 'FashionMNIST', 'CIFAR10']):
    sub_plot(ax, ds, first)
    first = False

f.set_size_inches(5, 3)
plt.show()
plt.savefig('hyper_opt.pdf', bbox_inches='tight')

