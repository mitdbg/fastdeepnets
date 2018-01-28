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
from linear_regression_experiment import (
    tn, normalize, get_sparse_columns, get_sparsity,
    GroupSparseModel, ShrinkModel
)
import openml
import torch
from sklearn import preprocessing

def load_dataset(id, do_normalize=True, factor=1):
    dataset = openml.datasets.get_dataset(id)
    X, y, categorical = dataset.get_data(
        target=dataset.default_target_attribute,
        return_categorical_indicator=True)
    lines_without_nans = (np.isnan(X).sum(1) == 0)
    X = X[lines_without_nans]
    y = y[lines_without_nans]
    if any(categorical):
        enc = preprocessing.OneHotEncoder(categorical_features=categorical)
        X = enc.fit_transform(X).todense()
    inputs = torch.from_numpy(X).float().cuda()
    outputs = torch.from_numpy(y).long().cuda()
    if do_normalize:
        inputs = normalize(inputs)
    return Variable(inputs * factor), Variable(outputs)

def train(inputs, outputs, model, l1, l2, lr=1e-5, epochs=10000):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adagrad(model.parameters(), lr=lr)

    log = []
    for _ in range(epochs):
        prediction = model(inputs)
        acc = tn((prediction.max(1)[1] == outputs).float().mean())
        original_loss = criterion(prediction, outputs)
        penalty = model.penalty(l1, l2)
        error = original_loss + penalty
        optimizer.zero_grad()
        error.backward()
        log.append((tn(original_loss), tn(penalty), tn(get_sparsity(model)), acc))
        optimizer.step()

    return np.array(log)

def run_experiment(exp):
    Ctr, inputs, outputs, l1, l2, lr, epochs = exp
    classes_count = int(tn(outputs.max())) + 1
    model = Ctr(inputs.size(1), classes_count).cuda()
    return train(inputs, outputs,model, l1, l2, lr, epochs)

GROUP_SPARSE_PARAMS = {
    1041: [
        (0.1, 2, 2000),
        (0.036, 2, 2000),
        (0.01, 2, 20000),
        (0.0036, 2.05, 40000),
        (0.001, 2.1, 40000),
        (0.00036, 2.1, 100000),
        (0.0001, 2.1, 100000),
        (0, 2.1, 100000),
    ],
    1477: [
        (1, 2, 3000),
        (0.36, 2.05, 5000),
        (0.1, 2.1, 5000),
        (0.036, 1.9, 15000),
        (0.01, 1.7, 15000),
        (0.0036, 1.65, 200000),
        (0.001, 1.6, 200000),
        (0.00036, 1.6, 200000),
        (0.0001, 1.6, 100000)
    ]
}

SHRINK_PARAMS = {
    1041: [
        (1, 1.55, 6000),
        (0.36, 1.575, 100000),
        (0.1, 1.6, 100000),
        (0.036, 1.6, 100000),
        (0.01, 1.6, 30000),
        (0.0036, 1.55, 100000),
        (0.001, 1.5, 100000),
        (0.00036, 1.45, 200000),
        (0.0001, 1.4, 200000),
        (0, 1.4, 200000),
    ],
    1477: [
        (1, 1.6, 10000),
        (0.36, 1.5, 45000),
        (0.1, 1.4, 45000),
        (0.036, 1.4, 45000),
        (0.01, 1.4, 45000),
        (0.0036, 1.4, 50000),
        (0.001, 1.4, 50000),
        (0.00036, 1.4, 300000),
        (0.0001, 1.45, 300000),
        (0.000036, 1.475, 500000),
        (0.00001, 1.5, 500000),
        (0, 1.45, 300000),
    ]
}

DATASETS = [
    (1477, True, 1.0),
    (1041, False, 1/255.0),
]

RUNNING = True

if __name__ == "__main__":
    if RUNNING:
        for ds_args in DATASETS:
            i, o = load_dataset(*ds_args)
            with Pool(40) as p:
                gs_params = GROUP_SPARSE_PARAMS[ds_args[0]]
                s_params = SHRINK_PARAMS[ds_args[0]]
                gs_logs = p.map(run_experiment, [(GroupSparseModel, i, o, l, 0, 10**-lr, it) for (l, lr, it) in gs_params])
                s_logs = p.map(run_experiment, [(ShrinkModel, i, o, l, 1e-8, 10**-lr, it) for (l, lr, it) in s_params])
                with open('./experiments_results/logistic_reg_%s.pkl' % str(ds_args[0]), 'wb') as f:
                    torch.save([gs_params, gs_logs, s_params, s_logs], f)
