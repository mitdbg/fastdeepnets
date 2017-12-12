import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from utils.wrapping import wrap

testing_data = None
training_data = None
weights = None

def load_data():
    global testing_data, training_data, weights
    if weights == None:
        testing_raw = np.asarray(pd.read_csv('./datasets/Poker/testing.csv').values)
        training_raw = np.asarray(pd.read_csv('./datasets/Poker/training.csv').values)

        encoder = OneHotEncoder(categorical_features=list(range(10))).fit(testing_raw)

        testing_data = np.asarray(encoder.transform(testing_raw).todense().astype('float32'))
        training_data = np.asarray(encoder.transform(training_raw).todense().astype('float32'))
        weights = wrap(torch.from_numpy(1 / np.bincount(training_data[:, -1].astype('int64')))).float()


class PokerDataset(Dataset):
    def __init__(self, train=True):
        load_data()
        if train:
            self.data = training_data
        else:
            self.data = testing_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        d = self.data[ix]
        inputs =  d[0:-1]
        return inputs, int(d[-1])

def get_dl(ds, train=True, bs=1000):
    return DataLoader(
        ds,
        batch_size=bs if train else len(ds),
        pin_memory=torch.cuda.device_count() > 0,
        shuffle=True
    )

if __name__ == '__main__':
    print(weights)
    load_data()
    print(weights)
