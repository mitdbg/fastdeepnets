from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass
import numpy as np
from torch.autograd import Variable
from torch.optim import Adam, Adagrad, SGD
from scipy.io.arff import loadarff
from torch.multiprocessing import Pool
import torch

def tn(o):
    try:
        return o.data.cpu().numpy()[0]
    except:
        return o

def normalize(tensor):
    m = tensor.mean(0)
    tensor -= m
    s = tensor.std(0)
    return tensor / s

def load_dataset(ds):
    file = loadarff('../../datasets/mtr-datasets/%s' % ds)[0]
    data = np.array([[w for w in y] for y in file])
    inputs = torch.from_numpy(data[:, :-16]).float().cuda()
    outputs = torch.from_numpy(data[:, -16:]).float().cuda()
    return Variable(normalize(inputs)), Variable(normalize(outputs))

def get_sparse_columns(model, threshold=1e-3):
    weights = model.get_weights()
    return (weights.abs() < threshold).float().sum(0) == weights.size(0)

def get_sparsity(model):
    return get_sparse_columns(model).float().mean()

# This class is just a wrapper so that they all have the same
# interface
class LinearModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=True)

    def penalty(self, l1, l2):
        return 0.0

    def forward(self, x):
        return self.linear(x)

    def get_weights(self):
        return self.linear.weight

class GroupSparseModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(GroupSparseModel, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=True)

    def penalty(self, l1, l2):
        return l1 * self.linear.weight.pow(2).sum(0).sqrt().sum()

    def get_weights(self):
        return self.linear.weight

    def forward(self, x):
        return self.linear(x)

class ShrinkModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ShrinkModel, self,).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=True)
        self.beta = torch.nn.Parameter(torch.randn(in_features))

    def penalty(self, l1, l2):
        return l1 * torch.abs(self.beta).sum() + l2 * self.linear.weight.pow(2).sum()

    def get_weights(self):
        return self.linear.weight * self.beta

    def forward(self, x):
        return self.linear(self.beta * x)

def create_models(inputs, outputs):
    models = []
    for Ctr in [LinearModel, GroupSparseModel, ShrinkModel]:
        models.append(Ctr(inputs.size(1), outputs.size(1)).cuda())
    return models

def train(inputs, outputs, model, l1, l2, lr=1e-5, epochs=10000):
    criterion = torch.nn.MSELoss()
    optimizer = Adagrad(model.parameters(), lr=lr)

    # Everyone starts the same way
    for p in model.parameters():
        # p.data.fill_(1.0)
        pass

    log = []
    for _ in range(epochs):
        original_loss = criterion(model(inputs), outputs)
        penalty = model.penalty(l1, l2)
        error = original_loss + penalty
        optimizer.zero_grad()
        error.backward()
        log.append((tn(original_loss), tn(penalty), tn(get_sparsity(model))))
        optimizer.step()

    return np.array(log)

def run_experiment(exp):
    Ctr, inputs, outputs, l1, l2, lr, epochs = exp
    model = Ctr(inputs.size(1), outputs.size(1)).cuda()
    return train(inputs, outputs,model, l1, l2, lr, epochs)

GROUP_SPARSE_PARAMS = {
    'scm1d.arff': [
        (1, 2, 2000),
        (0.36, 2.1, 4000),
        (0.1, 2.2, 4000),
        (0.036, 2.15, 8000),
        (0.01, 2.1, 8000),
        (0.0036, 2.1, 15000),
        (0.001, 2.1, 15000),
        (0.00036, 2.1, 15000),
        (0.0001, 2.1, 15000),
    ],
    'oes97.arff': [
        (1, 3, 2000),
        (0.36, 2.6, 6000),
        (0.1, 2.2, 6000),
        (0.036, 2.2, 10000),
        (0.01, 2.2, 10000),
        (0.0036, 2.15, 100000),
        (0.001, 2.1, 100000),
    ]
}

SHRINK_PARAMS = {
    'scm1d.arff': [
        (10, 1.2, 1000),
        (3.6, 1.2, 2000),
        (1, 1.2, 2000),
        (0.36, 1.2, 2000),
        (0.1, 1.2, 2000),
        (0.036, 1.2, 2500),
        (0.01, 1.3, 2500),
        (0.0036, 1.5, 10000),
        (0.001, 1.75, 10000),
        (0.00036, 1.85, 100000),
        (0.0001, 1.95, 100000)
    ],
    'oes97.arff': [
        (10, 1.2, 1000),
        (3.6, 1.275, 7000),
        (1, 1.35, 7000),
        (0.36, 1.35, 7000),
        (0.1, 1.35, 7000),
        (0.036, 1.325, 10000),
        (0.01, 1.3, 10000),
        (0.0036, 1.5, 15000),
        (0.001, 1.75, 15000),
        (0.00036, 1.85, 100000),
        (0.0001, 1.95, 100000)
    ]
}

RUNNING = True

if __name__ == "__main__":
    if RUNNING:
        for ds in GROUP_SPARSE_PARAMS.keys():
            i, o = load_dataset(ds)
            with Pool(40) as p:
                gs_params = GROUP_SPARSE_PARAMS[ds]
                s_params = SHRINK_PARAMS[ds]
                gs_logs = p.map(run_experiment, [(GroupSparseModel, i, o, l, 0, 10**-lr, it) for (l, lr, it) in gs_params])
                s_logs = p.map(run_experiment, [(ShrinkModel, i, o, l, 1e-8, 10**-lr, it) for (l, lr, it) in s_params])
                with open('./experiments_results/linear_reg_%s.pkl' % ds, 'wb') as f:
                    torch.save([gs_params, gs_logs, s_params, s_logs], f)
