from matplotlib import rc, use
# use('agg')
import numpy as np
import pandas as pd
import torch
from glob import glob
import matplotlib.pyplot as plt
from glob import glob
from paper.ICML.models.VGG import cfg
# for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
vgg_configs = {k: [x for x in v if x != 'M'] for (k, v) in cfg.items()}

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

def process_file(fn):
    with open(fn, 'rb') as f:
        params, logs = torch.load(f)
        logs = logs.logs
    # Mistake while writing
    logs.measure.replace('test_', 'test_acc', inplace=True)
    return params, logs

def read_file(mode, id):
    fn = './experiments_results/simple_%s_CIFAR10_%s.pkl' % (mode, id)
    return process_file(fn)

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

def get_best_entry(logs):
    vac = logs[logs.measure == 'val_acc']
    return logs[logs.epoch == vac.ix[vac['value'].argmax()].epoch]

def plot_shape(logs, segments=10, vgg='VGG16', factor=2):
    cf = vgg_configs[vgg]
    best = get_best_entry(logs)
    best_epoch = int(best.epoch.iloc[0])
    parts = np.linspace(0, best_epoch, segments).astype(int)
    last_value = None
    for epoch in parts:
        if epoch == 0:
            values = np.array(cf) * factor
        else:
            at_that_time = logs[logs.epoch == epoch]
            at_that_time = at_that_time[at_that_time['measure'].str.startswith('size')]
            values = at_that_time.value.as_matrix()[:-2]
        last_value = values
        plt.fill_between(range(len(values)), 0*values, values, color='C0', alpha=0.1)
    plt.xlabel('layer')
    plt.ylabel('size')
    plt.plot(range(len(last_value)), last_value, color='C0', label='Converted Shrinknet size')
    plt.plot(range(len(last_value)), cf, color='C1', label='Original VGG16 size')
    ax = plt.gca()
    ax.yaxis.grid(b=True, which='major', alpha=0.4, linestyle='--')
    ax.xaxis.grid(b=True, which='major', alpha=0.4, linestyle='--')
    plt.legend()


def compare_static_dynamic():
    results = []
    for i in range(2):
        cur = get_best_entry(read_file('static', i)[1])
        results.append(cur)
    return results

def cost(seq):
    prev = 3
    total = 0
    for e in seq:
        total += prev * e * 9
        prev = e
    return total

def get_final_size(logs,):
    best = get_best_entry(logs)
    epoch = int(best.epoch.iloc[0])
    at_that_time = logs[logs.epoch == epoch]
    at_that_time = at_that_time[at_that_time['measure'].str.startswith('size')]
    values = at_that_time.value.as_matrix()[:-2]
    return cost(values)

def get_all_files(mode):
    pattern = './experiments_results/simple_%s_CIFAR10_*.pkl' % (mode)
    files = glob(pattern)
    return [process_file(x) for x in files]


def get_compression(vgg='VGG16'):
    compressions = []
    cf = vgg_configs[vgg]
    original_cost = cost(cf)
    result = []
    for _, logs in get_all_files('dynamic'):
        b = get_best_entry(logs)
        result.append(cost(cf) / get_final_size(logs))
    return np.array(result)

def get_final_measure(mode, measure='test_acc'):
    best_results = [get_best_entry(x[1]) for x in get_all_files(mode)]
    if measure == 'time':
        return np.array([float(x.max().time) for x in best_results])
    else:
        return np.array([float(x[x.measure == measure].value) for x in best_results])
