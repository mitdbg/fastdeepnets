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
    'MNIST': (97, 98.75),
    'FashionMNIST': (86, 90.2),
    'CIFAR10': (46, 56)
}

conv_nice_boundaries = {
    'MNIST': (98.5, 99.5),
    'FashionMNIST': (87.75, 92.5),
    'CIFAR10': (55, 75)
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

def load_files(dataset, mode, max_samples=50, conv=False):
    p = ""
    if conv:
        p = "conv_"
    pat = './experiments_results/hyper_opt_%s%s_%s*' % (p, dataset, mode)
    all_files = glob(pat)
    all_files = all_files[:max_samples]
    assert len(all_files) == max_samples, "Not enough experiments"
    return np.stack([load_single_file(f) for f in all_files])

for conv in [True, False]:
    if conv:
        print("Convolutional:")
    else:
        print("Fully Connected")
    for dataset in nice_boundaries.keys():
        print(dataset)
        for mode in modes:
            infos = load_files(dataset, mode, 50, conv)[:, [1, 3]]
            bests_indices = infos[:, 0].argsort()[::-1][:10]
            res = np.percentile(infos[bests_indices], 50, axis=0)
            print('\t'.join(map(str, [mode] + list(res))))


def sub_plot(ax, dataset, first, conv=False, ff=False):
    try:
        infos = [load_files(dataset, mode, 50, conv)[:, 2] * 100 for mode in modes]
        result = ax.boxplot(infos, patch_artist=True, widths=0.7,
                            medianprops={'color': (250/255, 49/255,0, 1), 'linewidth': 2}
                            )
    except:
        result = {'boxes': []}
        pass
    boxes = result['boxes']
    for b, h, c in zip(boxes, hatches, colors):
        b.set_hatch(h)
        b.set_facecolor(c)
        b.set_edgecolor('black')
    ax.set_xticks([])
    ax.yaxis.grid(b=True, which='major', alpha=0.4, linestyle='--')
    if conv:
        ax.set_ylim(conv_nice_boundaries[dataset])
    else:
        ax.set_title('\\textbf{%s}' % dataset, size=10)
        ax.set_ylim(nice_boundaries[dataset])
    if dataset == 'CIFAR10':
        ax.yaxis.set_label_position("right")
        ax.yaxis.label.set_size(10)
        ax.set_ylabel('\\textbf{%s}' % ('LeNet-5' if conv else '3 Layers Feed Forward'))
    if first:
        ax.set_ylabel('Accuracy (\\%)')
    if ff:
        ax.legend(handles=boxes , labels=tex_labels ,loc='lower center', bbox_to_anchor=(1.75, - 0.23), ncol=3, columnspacing=0.5)

f, [axes_fc, axes_conv] = plt.subplots(2, 3, sharex=True)
plt.subplots_adjust(wspace=0.3, hspace=0.1)
first = True
ff = False
for ax, ds in zip(axes_fc, ['MNIST', 'FashionMNIST', 'CIFAR10']):
    sub_plot(ax, ds, first, False, ff)
    first = False
    ff = False
first = True
ff = True
for ax, ds in zip(axes_conv, ['MNIST', 'FashionMNIST', 'CIFAR10']):
    sub_plot(ax, ds, first, True, ff)
    first = False
    ff = False

f.set_size_inches(5, 5)
plt.show()
plt.savefig('hyper_opt.pdf', bbox_inches='tight', pad_inches=0)

