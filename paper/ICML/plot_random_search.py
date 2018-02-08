import torch
import pandas as pd
import numpy as np
from glob import glob
from paper.ICML.models.VGG import cfg
from algorithms.expectation_random_permutation import max_expectation

def is_pareto_efficient(costs):
     is_efficient = np.ones(costs.shape[0], dtype= bool)
     for i, c in enumerate(costs):
         if is_efficient[i]:
             is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)
     return is_efficient


def load_files(prefix, mode):
    files = glob('./experiments_results/random_search/%s_%s/*.pkl' % (prefix, mode))
    print(len(files))
    result = []
    for file_name in files:
        try:
            with open(file_name, 'rb') as f:
                result.append(torch.load(f))
        except Exception as e:
            print('error', file_name)
            pass
    return result

def get_best_entry(logs):
    vac = logs[logs.measure == 'val_acc']
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


columns = ['test_acc', 'size', 'val_acc', 'train_acc', 'time', 'epochs', 'lambda', 'batch_size', 'learning_rate', 'gamma', 'weight_decay', 'factor']

def summarize_experiment(prefix):
    r1 = np.array([generate_stats(x) for x in load_files(prefix, 'DYNAMIC')])
    dynamics = pd.DataFrame(r1, columns=columns[:r1.shape[1]])
    statics = pd.DataFrame([generate_stats(x) for x in load_files(prefix, 'STATIC')], columns=columns[:r1.shape[1]])
    return dynamics, statics

def plot_comparable(ax, dynamic_pareto, static, metric='size'):
    result = []
    for acc, size in dynamic_pareto:
        better = static[static.test_acc >= acc].min()[metric]
        result.append([acc, better / size])
    r = np.array(result)
    ax.axhline(1, ls=':', label="1x reference", color='black')
    ax.plot(r[:, 0], r[:, 1], label='%s improvement' % metric, color='black')
    ax.set_xlabel('Desired accuracy')
    ax.set_ylabel('Improvement factor')


def plot_size_accuracy_component(ax, dataset, color, label):
    M = dataset[['test_acc', 'size']].as_matrix()
    pareto = M[is_pareto_efficient(M * np.array([-1, 1]))]
    pareto.sort(axis=0)
    ax.plot(pareto[:, 1], pareto[:, 0], c=color, label="%s Pareto front" % label)
    ax.scatter(dataset['size'], dataset['test_acc'], c=color, label="%s models" % label)
    return pareto
