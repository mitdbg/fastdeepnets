import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def load_file(filename):
    with open(filename, 'r') as f:
        lines = [list(map(float, x.strip().split())) for x in f.readlines()]
    frame =  pd.DataFrame(np.array(lines), columns=['loss', 'size'])
    window_size = int(len(frame) / 100)
    return frame.rolling(window_size, min_periods=0).mean()

files = glob('./experiments_results/*')

file = load_file('./experiments_results/conv_MNIST_bs128_lamb1e-05.dat')

plt.figure()
plt.yscale('log')
for lamb in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
    file = load_file(
        './experiments_results/conv_MNIST_bs128_lamb%s.dat' % lamb)
    plt.plot(file['size'])

