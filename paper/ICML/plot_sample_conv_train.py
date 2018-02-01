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

def read_file(mode, id):
    fn = './experiments_results/simple_%s_CIFAR10_%s.pkl' % (mode, id)
    with open(fn, 'rb') as f:
        params, logs = torch.load(f)
        logs = logs.logs
    return params, logs

def extrat_timeline(logs, key):
    return logs[logs.measure==key]

def plot_timings(logs):
    f, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4)
    plt.subplots_adjust(wspace=0.25, hspace=0.5)
    gc_weights = extrat_timeline(logs, 'time_garbage_collect')
    gc_opt = extrat_timeline(logs, 'time_optimizer_update')
    train_time = extrat_timeline(logs, 'time_training')
    inference_time = extrat_timeline(logs, 'time_inference_val')
    ax1.plot(train_time.epoch, train_time.value, color='C0', label='Training time')
    ax2.plot(inference_time.epoch, inference_time.value, color='C1', label='Inference time')
    ax3.plot(gc_weights.epoch, gc_weights.value, color='C2', label='Garbage collection (weights)')
    ax3.plot(gc_opt.epoch, gc_opt.value, color='C3', label='Garbage collection (optimizer state)')
    for ax in [ax1, ax2, ax3, ax4]:
        ax.yaxis.grid(b=True, which='major', alpha=0.4, linestyle='--')
        ax.xaxis.grid(b=True, which='major', alpha=0.4, linestyle='--')
    for ax in [ax1, ax2, ax3]:
        ax.set_ylabel('Time (s)')
        ax.legend()
    ax4.xaxis.set_ticklabels([])
    ax4.set_ylabel('Total time (min, log-scale)')
    ax1.set_title('Time spent per epoch \\textbf{Training}')
    ax2.set_title('Time spent per epoch \\textbf{Inference}')
    ax3.set_title('Time spent per epoch \\textbf{Garbage collection}')
    sums = [x.value.sum() / 60 for x in [train_time, inference_time, gc_weights, gc_opt]]
    print(sums)
    ax4.bar(list(range(0, 4)), sums, color=['C0', 'C1', 'C2', 'C3'])
    ax4.set_yscale('log')
    f.set_size_inches((20, 5))
    plt.savefig('dynamic_timings.pdf', bbox_inches='tight', pad_inches=0)

plot_timings(read_file('dynamic', 0)[1])
