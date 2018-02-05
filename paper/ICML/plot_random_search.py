import torch
import pandas as pd
import numpy as np
from glob import glob
from paper.ICML.models.VGG import cfg
from algorithms.expectation_random_permutation import max_expectation

def load_files(prefix, mode):
    files = glob('./experiments_results/random_search/%s_%s/*' % (prefix, mode))
    result = []
    for file_name in files:
        try:
            with open(file_name, 'rb') as f:
                result.append(torch.load(f))
        except:
            pass
    return result

def get_best_entry(logs):
    vac = logs[logs.measure == 'val_acc']
    return logs[logs.epoch == vac.ix[vac['value'].argmax()].epoch]

def get_final_measure(log, measure='test_acc'):
    x = get_best_entry(log)
    if measure == 'time':
        return float(x.max().time)
    else:
        return float(x[x.measure == measure].value)

def generate_stats(infos):
    logs = infos[2].logs
    params = infos[1]['params']
    return [
        get_final_measure(logs, 'test_acc'),
        compute_size(infos),
        get_final_measure(logs, 'val_acc'),
        get_final_measure(logs, 'train_acc'),
        get_final_measure(logs, 'time'),
        params['lambda'],
        params['batch_size'],
        params['learning_rate'],
        params['gamma'],
        params['factor'],
        params['weight_decay']
    ]

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

def compute_size(infos, model='VGG16'):
    i = 0
    logs = infos[2].logs
    sizes = []
    channel_count = 3
    cost = 0
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
        original_sizes = [x for x in cfg[model] if x is not 'M']
        sizes = infos[1]['params']['factor'] * np.array(original_sizes)
        sizes = sizes.astype(int)
        cl1 = infos[1]['params']['classifier_layer_1']
        cl2 = infos[1]['params']['classifier_layer_2']
    return compute_cost(sizes, cl1, cl2)



def summarize_experiment(prefix):
    dynamics = pd.DataFrame([generate_stats(x) for x in load_files(prefix, 'DYNAMIC')], columns=['test_acc', 'size', 'val_acc', 'train_acc', 'time', 'lambda', 'batch_size', 'learning_rate', 'gamma', 'factor', 'weight_decay'])
    statics = pd.DataFrame([generate_stats(x) for x in load_files(prefix, 'STATIC')], columns=['test_acc', 'size', 'val_acc', 'train_acc', 'time', 'lambda', 'batch_size', 'learning_rate', 'gamma', 'factor', 'weight_decay'])
    return dynamics, statics

# plot_timings(read_file('dynamic', 0)[1])
# plot_shape(read_file('dynamic', 0)[1])
