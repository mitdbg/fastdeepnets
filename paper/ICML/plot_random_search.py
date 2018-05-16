from matplotlib import rc, use
# use('agg')
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob
from paper.ICML.models.VGG import cfg
from algorithms.expectation_random_permutation import max_expectation

theirs = np.array([x for x  in[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'] if x is not 'M'])
configs = {
    'COVER_FC': {
        'ymin': 0.5,
        's_range': (0.7, 10),
        'ref': (3440, 0.83),
        'cpu': {
            'warmup': 50,
            'rep': 500,
            'large_batches': 512,
        },
        'gpu': {
            'warmup': 1,
            'rep': 100,
            'large_batches': 4096,
        }
    },
    'CIFAR10_VGG': {
        'ymin': 0.4,
        's_range': (0.8, 6),
        'cpu': {
            'warmup': 1,
            'rep': 10,
            'large_batches': 64,
        },
        'gpu': {
            'warmup': 1,
            'rep': 100,
            'large_batches': 1024,
        }
    }
}

def is_pareto_efficient(costs):
     is_efficient = np.ones(costs.shape[0], dtype= bool)
     for i, c in enumerate(costs):
         if is_efficient[i]:
             is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)
     return is_efficient


def load_files(prefix, mode):
    with open('./experiments_results/%s_%s_benchmark.pkl' % (prefix, mode), 'rb') as f:
        benchmark = torch.load(f)
        benchmark = {k[0]: k[1:] for k in benchmark}
    files = glob('./experiments_results/random_search/%s_%s/*.pkl' % (prefix, mode))
    result = []
    errors = 0
    for file_name in files:
        id = file_name.split('/')[-1].replace('.pkl', '')
        try:
            with open(file_name, 'rb') as f:
                result.append(torch.load(f) + (benchmark[id],))
        except:
            errors += 1
            pass
    print(errors)
    return result

def get_best_entry(logs):
    vac = logs[logs.measure == 'val_acc'].ix[int(len(logs) / 2):]
    return logs[logs.epoch == vac.ix[vac['value'].argmax()].epoch]

def get_final_measure(log, measure='test_acc'):
    x = get_best_entry(log)
    if measure == 'epoch':
        return int(x.epoch.iloc[0])
    if measure == 'time':
        return float(x.max().time)
    else:
        return float(x[x.measure == measure].value)

def generate_stats(infos):
    logs = infos[2].logs
    params = infos[1]['params']
    result = [
        get_final_measure(logs, 'test_acc'),
        compute_size(infos),
        get_final_measure(logs, 'val_acc'),
        get_final_measure(logs, 'train_acc'),
        get_final_measure(logs, 'time'),
        get_final_measure(logs, 'epoch'),
        params['lambda'],
        params['batch_size'],
        params['learning_rate'],
        params['gamma'],
        params['weight_decay'],
        *infos[3]
    ]
    try:
        result.append(params['factor'])
    except:
        pass
    try:
        result.append(params['layers'])
    except:
        pass
    return result

def compute_cost(seq, cl1, cl2):
    prev = 3
    total = 0
    for e in seq:
        total += prev * e * 9
        prev = e
    total += 1024 * cl1
    total += cl1 * cl2
    total += 10 * cl2
    return total

def compute_size(infos):
    i = 0
    logs = infos[2].logs
    sizes = []
    channel_count = infos[1]['params']['input_features'][0]
    model_kind = infos[1]['model'].__name__
    cost = 0
    if model_kind == 'FullyConnected':
        for i in range(infos[1]['params']['layers']):
            try:
                cc = get_final_measure(logs, 'size_%s' % (i + 1))
            except:
                cc = infos[1]['params']['size_layer_%s' % (i + 1)]
            cost += cc * channel_count
            channel_count = cc
        return cost
    else:
        try:
            while True:
                i += 1
                sizes.append(get_final_measure(logs, 'size_%s' % i))
        except:
            pass
        if sizes:
            cl1 = sizes[-2]
            cl2 = sizes[-1]
            sizes = sizes[:-2]
        else:
            model = infos[1]['params']['name']
            original_sizes = [x for x in cfg[model] if x is not 'M']
            sizes = infos[1]['params']['factor'] * np.array(original_sizes)
            sizes = sizes.astype(int)
            cl1 = infos[1]['params']['classifier_layer_1']
            cl2 = infos[1]['params']['classifier_layer_2']
        return compute_cost(sizes, cl1, cl2)


columns = ['test_acc', 'size', 'val_acc', 'train_acc', 'time', 'epochs', 'lambda', 'batch_size', 'learning_rate', 'gamma', 'weight_decay', 'bench_cpu_small', 'bench_cpu_big', 'bench_gpu_small', 'bench_gpu_big', 'factor']

def summarize_experiment(prefix):
    r1 = np.array([generate_stats(x) for x in load_files(prefix, 'DYNAMIC')])
    dynamics = pd.DataFrame(r1, columns=columns[:r1.shape[1]])
    statics = pd.DataFrame([generate_stats(x) for x in load_files(prefix, 'STATIC')], columns=columns[:r1.shape[1]])
    if prefix == 'COVER_FC':
        dynamics = dynamics[dynamics.factor == 3]
    return dynamics, statics

def plot_comparable(ax, dyn, dynamic_pareto, static, metric='size'):
    result = []
    for acc, size in dynamic_pareto:
        line = dyn[np.bitwise_and(dyn.test_acc == acc, dyn['size'] == size)]
        better = static[static.test_acc >= acc].min()[metric]
        result.append([acc, better / line[metric]])
    r = np.array(result)
    ax.axhline(1, ls=':', label="1x reference", color='black')
    ax.plot(r[:, 0], r[:, 1], label='%s improvement' % metric, color='black')


def plot_size_accuracy_component(ax, dataset, color, label, marker=None):
    M = dataset[['test_acc', 'size']].as_matrix()
    pareto = M[is_pareto_efficient(M * np.array([-1, 1]))]
    pareto.sort(axis=0)
    ax.plot(pareto[:, 1], pareto[:, 0], c=color, label="%s Pareto front" % label)
    ax.scatter(dataset['size'], dataset['test_acc'], c=color, label="%s model" % label, alpha=0.3, marker=marker)
    return pareto

def plot_all(dyn, sta, dataset):
    conf = configs[dataset]

    ax1 = plt.subplot2grid((5, 2), (0, 0), colspan=2, rowspan=2) 
    ax1.set_xlabel('Size (floating point parameters)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Distribution of models by size and accuracy')
    ax2 = plt.subplot2grid((5, 2), (2, 0), colspan=2)
    ax2.set_title('Size benefits')
    ax2.set_ylabel('Compression factor')
    ax3 = plt.subplot2grid((5, 2), (3, 0))
    ax3.set_title('CPU speedup ($bs=1$)')
    ax4 = plt.subplot2grid((5, 2), (3, 1))
    ax4.set_title('CPU speedup ($bs=%s$)' % conf['cpu']['large_batches'])
    ax5 = plt.subplot2grid((5, 2), (4, 0))
    ax5.set_title('GPU speedup ($bs=1$)')
    ax6 = plt.subplot2grid((5, 2), (4, 1))
    ax6.set_title('GPU speedup ($bs=%s$)' % conf['gpu']['large_batches'])
    ax1.set_xscale('log')
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    dyn_pareto = plot_size_accuracy_component(ax1, dyn, 'C0', 'ShrinkNets')
    plot_size_accuracy_component(ax1, sta, 'C1', 'Static', marker='^')
    if 'ref' in conf:
        ax1.scatter([conf['ref'][0]], [conf['ref'][1]], marker='*', label='Result from Scardapane', color='C4')
    ax1.legend(loc='lower right', framealpha=0.5)
    plot_comparable(ax2, dyn, dyn_pareto, sta, 'size')
    plot_comparable(ax3, dyn, dyn_pareto, sta, 'bench_cpu_small')
    plot_comparable(ax4, dyn, dyn_pareto, sta, 'bench_cpu_big')
    plot_comparable(ax5, dyn, dyn_pareto, sta, 'bench_gpu_small')
    plot_comparable(ax6, dyn, dyn_pareto, sta, 'bench_gpu_big')
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.yaxis.grid(b=True, which='major', alpha=0.4, linestyle='--')
        ax.xaxis.grid(b=True, which='major', alpha=0.4, linestyle='--')

    for ax in [ax2, ax3, ax4, ax5, ax6]:
        ax.set_xlim(0.8)
    ax1.set_ylim(ymin=conf['ymin'])
    ax2.set_ylim(ymax=40)
    ax2.set_xlabel('Desired accuracy')

    for ax in [ax3, ax4, ax5, ax6]:
        ax.set_ylabel('Speedup')
        ax.set_ylim(conf['s_range'])

    f = plt.gcf()
    f.set_size_inches((5, 10))
    plt.tight_layout()
    plt.savefig('%s_summary.pdf' % dataset, bbox_inches='tight', pad_inches=0)

def select_models(dynamic):
    results = [(i, get_final_measure(x[2].logs, 'test_acc')) for i, x in enumerate(dynamic)]
    best = max(results, key=lambda x: x[1])[0]
    best= dynamic[best][2].logs
    rr = np.array([x[1] for x in results])
    good = np.abs(rr - 0.905).argmin()
    print(rr[good])
    return best,dynamic[good][2].logs

def get_size_at_epoch(log, epoch):
    l = log[log.epoch == epoch]
    sizes = []
    i = 1
    while True:
        try:
            t = int(l[l.measure == 'size_%s' % i].value)
            sizes.append(t)
            i += 1
        except:
            break
    return sizes[:-2]

def plot_shape(ax, log, steps=8):
    max_epoch = get_final_measure(log, 'epoch')
    selected = np.linspace(1, max_epoch, steps).astype(int)
    for epoch in selected:
        if epoch == 1:
            sizes = theirs * 2
        else:
            sizes = get_size_at_epoch(log, epoch)
        r = list(range(1, len(sizes) + 1))
        ax.fill_between(r, sizes, alpha=0.1, color='C0')
    last_sizes = get_size_at_epoch(log, max_epoch)
    ax.plot(r, last_sizes, label='Final size')
    ax.plot(r, theirs, ':', color='black', label='original VGG size')

def plot_shapes(dynamic):
    best, good = select_models(dynamic)
    fig, lines = plt.subplots(1, 2, sharex=True)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plot_shape(lines[0], best)
    plot_shape(lines[1], good)
    for ax in lines:
        ax.set_ylabel('\\#Convolution kernels')
    lines[1].set_xlabel('Layer')
    lines[1].legend(loc='upper left')
    fig.set_size_inches((5, 7))
    plt.savefig('size_evolution.pdf', bbox_inches='tight', pad_inches=0)


