import torch
import numpy as np
from torch.autograd import Variable
from copy import deepcopy
from time import perf_counter
from paper.ICML.plot_random_search import get_best_entry
from glob import glob

# Enable performance optimizations
torch.backends.cudnn.benchmark = True

def load_model(filename, dataset='cover'):
    with open(filename, 'rb') as f:
        data = torch.load(f)
    best_entry =  get_best_entry(data[2].logs)
    model = data[1]['model']
    if dataset == 'cover':
        cfg = deepcopy(data[1])
        for i in range(cfg['params']['layers']):
            cs = float(best_entry[best_entry.measure == 'size_%s' % (i + 1)].value)
            cfg['params']['size_layer_%s' % (i + 1)] = int(cs)
    cfg['params']['dynamic'] = False # remove filters
    model = model(cfg['params'])
    return cfg, model

def evaluate_model(cfg, model, hardware, batch_size):
    size = cfg['params']['input_features']
    data = Variable(torch.rand(*((batch_size,) + size)))
    if hardware == 'gpu':
        model = model.cuda()
        data = data.cuda()
    timings = []
    off = 20
    for i in range(off + 500):
        s = perf_counter()
        model(data)
        if hardware == 'gpu':
            torch.cuda.synchronize()
        e = perf_counter()
        timings.append(e - s)

    timings = timings[off:]
    return np.array(timings).mean() / batch_size
def full_evaluation(fn):
    idx = fn.split('/')[-1].replace('.pkl', '')
    cfg, model = load_model(fn)
    cpu_small = evaluate_model(cfg, model, 'cpu', 1)
    cpu_big = evaluate_model(cfg, model, 'cpu', 512)
    gpu_small = evaluate_model(cfg, model, 'gpu', 1)
    gpu_big = evaluate_model(cfg, model, 'gpu', 512)
    result = (idx, cpu_small, cpu_big, gpu_small, gpu_big)
    print(result)
    return result

def get_all_models(dataset, mode):
    print(dataset, mode)
    fns = glob('./experiments_results/random_search/%s_%s/*.pkl' % (dataset, mode))
    result = [full_evaluation(x) for x in fns]
    with open('./experiments_results/%s_%s_benchmark.pkl' % (dataset, mode), 'wb') as f:
        torch.save(result, f)

if __name__ == '__main__':
    get_all_models('COVER_FC', 'DYNAMIC')
    get_all_models('COVER_FC', 'STATIC')
