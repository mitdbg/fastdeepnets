from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import numpy as np

all_data = np.loadtxt('./datasets/Airfoil/Dataset.data')

# Test/train split from:
# A Neural Networks Approach to Aerofoil Noise Prediction
# - K. Lau, R. López, E. Oñate
testing_mask = all_data[:, 2] == 0.1016

testing_data = all_data[testing_mask]
training_data = all_data[~testing_mask]

training_mean = training_data[:, :5].mean(axis=0)
training_std = (training_data[:, :5] - training_mean).std(axis=0)

class AirfoilDataset(Dataset):
    def __init__(self, train=True, normalized=True):
        self.normalized = normalized
        if train:
            self.data = training_data
        else:
            self.data = testing_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        d = self.data[ix]
        inputs =  d[0:5]
        if self.normalized:
            inputs -= training_mean
            inputs /= training_std
        return inputs.astype('float32'), d[5].astype('float32')

def get_dl(ds, train=True, bs=50):
    return DataLoader(
        ds,
        batch_size=bs if train else len(ds),
        pin_memory=torch.cuda.device_count() > 0,
        shuffle=True
    )
