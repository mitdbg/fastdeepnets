import torch
import numpy as np
from torch.optim import Adagrad
from torch.autograd import Variable
from torch.multiprocessing import Pool
from linear_regression_experiment import (
    ShrinkModel, load_dataset, tn, get_sparsity, SHRINK_PARAMS
)

THRESHOLD = 0.5
FACTOR = 3

def train(inputs, outputs, model, l1, l2, lr=1e-5, epochs=10000, gamma=0.9, remove=False, randomized=False):
    criterion = torch.nn.MSELoss()
    optimizer = Adagrad(model.parameters(), lr=lr)

    # Everyone starts the same way
    for p in model.parameters():
        # p.data.fill_(1.0)
        pass

    log = []
    sum_beta = 0

    if gamma is not None:
        mask = model.beta.data.new(model.beta.size()).fill_(1).byte()
        exp_avg = model.beta.data.new(model.beta.size()).fill_(1.0)
        exp_std= model.beta.data.new(model.beta.size()).fill_(0)
    if randomized:
        noise = inputs.data.new(inputs.data.size())

    for _ in range(FACTOR * epochs):
        sum_beta += tn(mask.float().sum())
        if randomized:
            noise.uniform_(0, THRESHOLD)
            current_inputs = inputs * Variable((noise > exp_std).float())
        else:
            current_inputs = inputs
        original_loss = criterion(model(current_inputs), outputs)
        penalty = model.penalty(l1, l2)
        error = original_loss + penalty
        optimizer.zero_grad()
        error.backward()
        log.append((tn(original_loss),
                    tn(penalty),
                    tn(get_sparsity(model))))
        optimizer.step()
        if gamma is not None:
            bs = model.beta.data.sign()
            diff = bs - exp_avg
            exp_std.mul_(gamma).addcmul_(1 - gamma, diff, diff)
            exp_avg.mul_(gamma).add_(1 - gamma, bs)
            mask.mul_(exp_std <= THRESHOLD)
            if remove:
                model.beta.data.mul_(mask.float())
    return tuple(log[-1]) + (tn(mask.float().sum()), sum_beta)

def run_experiment(exp):
    model = exp[0](exp[1].size(1), exp[2].size(1)).cuda()
    return train(*((exp[1], exp[2] , model,) +exp[3:]))


REPEATS = 30
GRANULARITY = 15

running = True
first = True

if first:
    ds = 'scm1d.arff'
    selected_lambas = [4, 8]
else:
    ds = 'oes97.arff'
    selected_lambas = [4, 7]

if __name__ == '__main__':
    if running:
        i, o = load_dataset(ds)
        all_params = SHRINK_PARAMS[ds]
        with Pool(5) as p:
            for lamb_index in selected_lambas:
                base_params = all_params[lamb_index]
                results = []
                for randomized in [True, False]:
                    gs_params = [base_params + (1 - x, True, randomized) for x in 10**(-np.linspace(0, 5, GRANULARITY))] + [(base_params + (1, False, False))]
                    gs_params = gs_params * REPEATS
                    logs = p.map(run_experiment, [(ShrinkModel, i, o, l, 0, 10**-lr, it, gamma, remove, rnd) for (l, lr, it, gamma, remove, rnd) in gs_params])
                    results += [a + b for a , b in zip(gs_params, logs)]
                with open('./experiments_results/neuron_removal_reg_%s_l_%s.pkl' % (ds, base_params[0]), 'wb') as f:
                    torch.save(results, f)

