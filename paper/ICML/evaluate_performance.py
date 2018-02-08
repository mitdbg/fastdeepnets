import torch
import numpy as np
from torch.autograd import Variable
from copy import deepcopy
from time import perf_counter
import paper
from glob import glob

# Enable performance optimizations
torch.backends.cudnn.benchmark = True

configs = {
    'COVER_FC': {
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

def load_model(filename, dataset):
    with open(filename, 'rb') as f:
        data = torch.load(f)
    best_entry =  paper.ICML.plot_random_search.get_best_entry(data[2].logs)
    model = data[1]['model']
    cfg = deepcopy(data[1])
    if dataset == 'COVER_FC':
        for i in range(cfg['params']['layers']):
            try:
                cs = float(best_entry[best_entry.measure == 'size_%s' % (i + 1)].value)
                cfg['params']['size_layer_%s' % (i + 1)] = int(cs)
            except:
                pass # Keep the original size
    elif dataset == 'CIFAR10_VGG':
        sizes = []
        i = 0
        while True:
            try:
                cs = int(best_entry[best_entry.measure == 'size_%s' % (i + 1)].value)
                sizes.append(cs)
                i += 1
            except:
                break
        classifier_sizes = sizes[-2:]
        vgg_sizes = sizes[:-2]
        for i, s in enumerate(vgg_sizes):
            cfg['params']['size_%s' % (i + 1)] = s

        try:
            cfg['params']['classifier_layer_1'] = classifier_sizes[0]
            cfg['params']['classifier_layer_2'] = classifier_sizes[1]
            del cfg['params']['factor']
        except:
            pass # Keep the original values



    cfg['params']['dynamic'] = False # remove filters
    model = model(cfg['params'])
    return cfg, model

def evaluate_model(cfg, model, hardware, size, dataset):
    cd = configs[dataset][hardware]
    batch_size = 1 if size == 'small' else cd['large_batches']
    size = cfg['params']['input_features']
    off = cd['warmup']
    rep = cd['rep']
    data = Variable(torch.rand(*((batch_size,) + size)))
    if hardware == 'gpu':
        model = model.cuda()
        data = data.cuda()
    timings = []
    for i in range(off + rep):
        s = perf_counter()
        model(data)
        if hardware == 'gpu':
            torch.cuda.synchronize()
        e = perf_counter()
        timings.append(e - s)

    timings = timings[off:]
    return np.array(timings).mean() / batch_size
def full_evaluation(fn, dataset):
    idx = fn.split('/')[-1].replace('.pkl', '')
    cfg, model = load_model(fn, dataset)
    cpu_small = evaluate_model(cfg, model, 'cpu', 'small', dataset)
    cpu_big = evaluate_model(cfg, model, 'cpu', 'large', dataset)
    gpu_small = evaluate_model(cfg, model, 'gpu', 'small', dataset)
    gpu_big = evaluate_model(cfg, model, 'gpu', 'large', dataset)
    result = (idx, cpu_small, cpu_big, gpu_small, gpu_big)
    print(result)
    return result

def get_all_models(dataset, mode):
    print(dataset, mode)
    fns = glob('./experiments_results/random_search/%s_%s/*.pkl' % (dataset, mode))
    result = [full_evaluation(x, dataset) for x in fns]
    with open('./experiments_results/%s_%s_benchmark.pkl' % (dataset, mode), 'wb') as f:
        torch.save(result, f)

if __name__ == '__main__':
    # get_all_models('COVER_FC', 'DYNAMIC')
    get_all_models('COVER_FC', 'STATIC')
    # get_all_models('CIFAR10_VGG', 'DYNAMIC')
    get_all_models('CIFAR10_VGG', 'STATIC')
