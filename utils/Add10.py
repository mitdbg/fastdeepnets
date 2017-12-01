from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import numpy as np

testing_data = np.loadtxt('./datasets/add10/Dataset.data')

def add10_function(x, noise=True):
    x = x.T
    values = 10 * np.sin(np.pi * x[0] * x[1]) + 20*(x[2] - 0.5)**2 + 10*x[3] + 5*x[4]
    if noise:
        values+= np.random.normal(size=values.shape[0])
    return np.row_stack((x, values)).T


def generate_synthetic(sample_size=testing_data.shape[0], bounds=(0, 1)):
    input_data = np.random.uniform(bounds[0], bounds[1], size=(sample_size, 10))
    return input_data

class Add10Dataset(Dataset):

    def __init__(self, train=True, size=testing_data.shape[0], bounds=(0, 1)):
        if train:
            self.data = add10_function(generate_synthetic(size, bounds))
        else:
            self.data = testing_data
        super(Add10Dataset, self).__init__()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ix):
        return torch.from_numpy(self.data[ix, :10]).float(), torch.from_numpy(np.array([self.data[ix, 10]])).float()

def get_dl(ds, train=True, bs=100):
    return DataLoader(
        ds,
        batch_size=bs if train else len(ds),
        pin_memory=torch.cuda.device_count() > 0,
        shuffle=True
    )
