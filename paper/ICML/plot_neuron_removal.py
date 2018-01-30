from matplotlib import rc, use
use('agg')
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from glob import glob
# for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# def last(experiments):
#     return np.stack([x[-1] for x in experiments])
# 
# def load_file(filename):
#     with open(filename, 'rb') as f:
#         [gs_params, gs_logs, s_params, s_logs] = torch.load(f)
#     return last(gs_logs), last(s_logs)
# 
# def plot_file(ax, fn, max_val, right=False):
#     gs, s = load_file(fn)
#     ax.plot(gs[:, 2], gs[:, 0] / max_val, 'x-', label="Group Sparsity")
#     ax.plot(s[:, 2], s[:, 0] / max_val, '.-', label="ShrinkNet")
#     ax.set_ylim((-0.1, 1.1))
#     if right:
#         ax.yaxis.set_ticklabels([])
#     ax.yaxis.grid(b=True, which='major', alpha=0.4, linestyle='--')
#     ax.xaxis.grid(b=True, which='major', alpha=0.4, linestyle='--')
# 
# f, (axs_linear, axs_logistic) = plt.subplots(2, 2, sharex=True)
# plt.subplots_adjust(wspace=0.05, hspace=0.05)
# plot_file(axs_linear[0], './experiments_results/linear_reg_scm1d.arff.pkl', 1)
# plot_file(axs_linear[1], './experiments_results/linear_reg_oes97.arff.pkl', 1, True)
# plot_file(axs_logistic[0], './experiments_results/logistic_reg_1041.pkl', np.log(10))
# plot_file(axs_logistic[1], './experiments_results/logistic_reg_1477.pkl', np.log(6), True)
# axs_linear[0].legend()
# f.set_size_inches((5, 5))
# 
# f.text(0.52, 0.04, 'Sparsity', ha='center')
# f.text(0.02, 0.5, 'Normalized loss', va='center', rotation='vertical')
# plt.savefig('regressions.pdf', bbox_inches='tight', pad_inches=0)

def load_file(ds, lamb):
    fn = './experiments_results/neuron_removal_reg_%s.arff_l_%s.pkl' % (ds, lamb)
    with open(fn, 'rb') as f:
        data = torch.load(f)
    frame = pd.DataFrame(data, columns=[
        'lambda', 'lr', 'iterations', 'gamma', 'gc', 'randomized',
        'loss', 'penalty', 'sparsity', 'neurons_alive', 'cost'])
    deterministic = frame[np.bitwise_and(frame.randomized == False, frame.gc == True)]
    randomized = frame[frame.randomized == True]
    baseline = frame[np.bitwise_and(frame.randomized == False, frame.gc == False)]
    return deterministic.groupby('gamma').describe(), randomized.groupby('gamma').describe(), baseline.describe()

def t(xes):
    return 1 - xes

def plot_curve(target, f, k, label):
    if label == 'Deterministic':
        ls = 'x-'
    else:
        ls = '.-'
    target.plot(t(f.index), f[k]['mean'], ls, label=label)
    target.fill_between(t(f.index), f[k]['mean'] - f[k]['std'], f[k]['mean'] + f[k]['std'], alpha=0.2)


def plot(ax_left, ax_right, det, ran, bl):
    ff = 'loss'
    plot_curve(ax_left, det, ff, 'Deterministic')
    plot_curve(ax_left, ran, ff, 'Randomized')
    ax_left.set_xscale('log')
    bl_artist = ax_left.axhline(bl[ff]['mean'], color='black', label='Baseline')
    bl_boundary_artis = ax_left.axhline(bl[ff]['mean'] + bl[ff]['std'], ls=':', color='black')
    ax_left.axhline(bl[ff]['mean'] - bl[ff]['std'], ls=':', color='black', label='Baseline $\pm$ std dev')
    ax_left.set_ylabel('Loss')
    ax_right.set_ylabel('Total Cost')
    ax_right.yaxis.set_label_position("right")
    ax_left.set_xlim(ax_left.get_xlim()[::-1])
    ax_right.yaxis.tick_right()
    # plt.plt(det.index, det.loss['mean'], yerr=det.loss['std'], label='deterministic')
    plot_curve(ax_right, det, 'cost', 'deterministic')
    plot_curve(ax_right, ran, 'cost', 'randomized')
    ax_right.set_yscale('log')
    ax_right.yaxis.grid(b=True, which='major', alpha=0.4, linestyle='--')
    ax_right.xaxis.grid(b=True, which='major', alpha=0.4, linestyle='--')
    ax_left.yaxis.grid(b=True, which='major', alpha=0.4, linestyle='--')
    ax_left.xaxis.grid(b=True, which='major', alpha=0.4, linestyle='--')

def plot_configs(configs):
    fig, lines = plt.subplots(len(configs), 2, sharex=True)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for line, config in zip(lines, configs):
        f = load_file(*config)
        plot(*(tuple(line) + tuple(f)))
    lines[0][0].legend(bbox_to_anchor=(1.9, 1.3), ncol=2, loc='upper right')
    fig.set_size_inches((5, 10))
    fig.text(0.515, 0.04, '$\gamma$', ha='center')

plot_configs([('scm1d', 0.1), ('scm1d', 0.001), ('oes97', 0.1), ('oes97', 0.0036)])
plt.savefig('neuron_removal.pdf', bbox_inches='tight', pad_inches=0)
