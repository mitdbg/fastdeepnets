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

def last(experiments):
    return np.stack([x[-1] for x in experiments])

def load_file(filename):
    with open(filename, 'rb') as f:
        [gs_params, gs_logs, s_params, s_logs] = torch.load(f)
    return last(gs_logs), last(s_logs)

def plot_file(ax, fn, max_val, right=False):
    gs, s = load_file(fn)
    ax.plot(gs[:, 2], gs[:, 0] / max_val, 'x-', label="Group Sparsity")
    ax.plot(s[:, 2], s[:, 0] / max_val, '.-', label="ShrinkNet")
    ax.set_ylim((-0.1, 1.1))
    if right:
        ax.yaxis.set_ticklabels([])
    ax.yaxis.grid(b=True, which='major', alpha=0.4, linestyle='--')
    ax.xaxis.grid(b=True, which='major', alpha=0.4, linestyle='--')

f, (axs_linear, axs_logistic) = plt.subplots(2, 2, sharex=True)
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plot_file(axs_linear[0], './experiments_results/linear_reg_scm1d.arff.pkl', 1)
plot_file(axs_linear[1], './experiments_results/linear_reg_oes97.arff.pkl', 1, True)
plot_file(axs_logistic[0], './experiments_results/logistic_reg_1041.pkl', np.log(10))
plot_file(axs_logistic[1], './experiments_results/logistic_reg_1477.pkl', np.log(6), True)
axs_linear[0].legend()
f.set_size_inches((5, 5))

f.text(0.52, 0.04, 'Sparsity', ha='center')
f.text(0.02, 0.5, 'Normalized loss', va='center', rotation='vertical')
plt.savefig('regressions.pdf', bbox_inches='tight', pad_inches=0)
