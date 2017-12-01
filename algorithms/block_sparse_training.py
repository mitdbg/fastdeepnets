import torch
from copy import deepcopy
import numpy as np
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torchvision.datasets import MNIST, FashionMNIST
from utils.misc import tn
from utils.wrapping import wrap

EPOCHS = 15

def forward(model, dl, lamb=0, optimizer=None):
    criterion = CrossEntropyLoss()
    acc_sum = 0
    tot = 0
    for i, (images, labels) in enumerate(dl):
        images = wrap(Variable(images, requires_grad=False))
        labels = wrap(Variable(labels, requires_grad=False))
        output = model(images)
        if optimizer is not None:
            original_loss = criterion(output, labels)
            penalized_loss = original_loss + float(lamb) * model.loss()
            optimizer.zero_grad()
            penalized_loss.backward()
            optimizer.step()

        acc = (output.max(1)[1] == labels).float().sum()
        tot += len(labels)
        acc_sum += acc

    acc = tn(acc_sum.data / tot)
    return acc

def train_until_convergence(model, train_dl, val_dl, lamb, patience=3, min_epochs=2):
    best_model = None
    best_score = (-np.inf, 0)
    early_stop = 0

    print('lambda', lamb)

    optimizer = Adam(model.parameters())

    while True:
        train_acc = forward(model, train_dl, lamb, optimizer)
        current_accuracy = forward(model, val_dl)
        capacities = model.training_component().get_capacities().data.cpu().numpy()
        current_score = (-capacities.sum(), current_accuracy)

        print(train_acc, current_accuracy, capacities)

        if model.has_collapsed():
            raise StopIteration('Model collapsed')
        elif min_epochs > 0:
            min_epochs -=1
        elif current_score > best_score:
            best_model = deepcopy(model)
            best_score = current_score
            early_stop = 0
        elif early_stop == patience - 1:
            return best_score[1], best_model
        else:
            early_stop += 1

def train(gen_block, base_model, train_dl, val_dl, test_dl, start_lamb=10, lamb_decay=10, max_patience=4, default_block_size=25, min_lambda=1e-7, max_block_size=100):
    block_size = default_block_size
    lamb = start_lamb
    val_accuracies = []
    test_accuracies = []
    Global_best_acc = 0
    while block_size <= max_block_size: # Add blocks
        current_model = base_model.next_block(gen_block(block_size))
        best_model = None
        while True: # Find Grip
            try:
                current_accuracy, current_model = train_until_convergence(current_model, train_dl, val_dl, lamb, patience=max_patience)
                break
            except StopIteration:
                lamb /= lamb_decay
                if lamb < min_lambda:
                    return val_accuracies, test_accuracies, base_model
                current_model = base_model.next_block(gen_block(block_size))
        best_acc = current_accuracy
        best_model = deepcopy(current_model)
        while False: # Optimize
            best_acc = current_accuracy
            if best_acc < global_best_acc:
                break
            best_model = deepcopy(current_model)
            lamb *= lamb_decay
            try:
                current_accuracy, current_model = train_until_convergence(current_model, train_dl, val_dl, lamb, patience=max_patience)
            except StopIteration:
                current_accuracy = 0
            if current_accuracy < best_acc:
                break
            else:
                best_acc = current_accuracy
        if best_acc > global_best_acc:  # We did improve the model
            print('did improve')
            base_model = best_model # We build on top of this block
            block_size = max(block_size, int(min(max_block_size, 3 * tn(base_model.training_component().get_capacities().max().data))))
            global_best_acc = best_acc
            val_accuracies.append(best_acc)
            test_acc = forward(base_model, test_dl)
            print('TEST ACC:', test_acc)
            test_accuracies.append(test_acc)
        else:
            print('did not improve')
            lamb /= lamb_decay
            if lamb <= min_lambda: # We can only increase the size
                break
    return val_accuracies, test_accuracies, base_model
