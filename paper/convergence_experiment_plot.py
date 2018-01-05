from matplotlib import rc, use
use('agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
# for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

lambdas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
tex_labels = ["$\lambda=10^{-%s}$" % x for x in range(1, 6)]

def load_file(filename):
    with open(filename, 'r') as f:
        lines = [list(map(float, x.strip().split())) for x in f.readlines()]
    frame =  pd.DataFrame(np.array(lines), columns=['loss', 'size'])
    window_size = int(len(frame) / 100)
    return frame.rolling(window_size, min_periods=0).mean()

files = glob('./experiments_results/*')

comparisons = []
for l in lambdas:
    f_mnist = load_file('./experiments_results/conv_MNIST_bs256_lamb%s.dat' % l).iloc[-100:].min()
    f_fm = load_file('./experiments_results/conv_FashionMNIST_bs256_lamb%s.dat' % l).iloc[-100:].min()
    comparisons.append([
        f_mnist['size'], f_mnist['loss'], f_fm['size'], f_fm['loss']
    ])
comparisons = np.array(comparisons)

r = np.array(list(range(0, comparisons.shape[0])))
bw = 0.25

f, (ax_size, ax_loss) = plt.subplots(1, 2, sharex=True)
l1, = ax_size.plot(r, comparisons[:, 0], '-x')
l2, = ax_size.plot(r, comparisons[:, 2], '-o')
ax_loss.plot(r, comparisons[:, 1], '-x')
ax_loss.plot(r, comparisons[:, 3], '-o')
ax_loss.set_xlim(0, 4)
ax_size.set_xlim(0, 4)
ax_loss.set_yscale('log')
ax_size.set_xlabel('$\lambda$')
ax_loss.set_xlabel('$\lambda$')
ax_size.set_ylabel('Hidden Units')
ax_loss.set_ylabel('Cross-Entropy Loss')
ax_loss.yaxis.grid(b=True, which='major', alpha=0.4, linestyle='--')
ax_size.yaxis.grid(b=True, which='major', alpha=0.4, linestyle='--')
ax_loss.xaxis.grid(b=True, which='major', alpha=0.4, linestyle='--')
ax_size.xaxis.grid(b=True, which='major', alpha=0.4, linestyle='--')
ax_size.set_xticks(r)
ax_loss.set_xticks(r)
ax_size.set_xticklabels(["$10^{-%s}$" % x for x in range(1, 6)])
ax_loss.set_xticklabels(["$10^{-%s}$" % x for x in range(1, 6)])
plt.tight_layout()
ax_size.legend(handles=[l1, l2] , labels=['\\texttt{MNIST}', '\\texttt{FashionMNIST}'], bbox_to_anchor=(1.175, 1), ncol=3, columnspacing=0.5)
f.set_size_inches(5, 2.05)
plt.savefig('conv_MNIST_FM_comp.pdf', bbox_inches='tight')




f, (ax_size, ax_loss) = plt.subplots(2, 1, sharex=True)
plt.subplots_adjust(wspace=0, hspace=0)
ax_size.set_yscale('log')
ax_loss.set_yscale('log')
ax_size.set_ylabel('Hidden Units')
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Cross-Entropy loss')
plt.title("")
# plt.gca().yaxis.grid(b=True, which='major', linestyle='-')
ax_loss.yaxis.grid(b=True, which='major', alpha=0.4, linestyle='--')
ax_size.yaxis.grid(b=True, which='major', alpha=0.4, linestyle='--')
markers = ['>', 'o', 'x', 'D', 'p']
lines = []
for m, lamb in zip(reversed(markers), reversed(lambdas)):
    file = load_file(
        './experiments_results/conv_MNIST_bs256_lamb%s.dat' % lamb)
    l, = ax_size.plot(np.arange(0, 100, 1 / len(file) * 100), file['size'], m + '-', markevery=int(len(file) / 5))
    lines.append(l)
for m, lamb in zip(reversed(markers), reversed(lambdas)):
    file = load_file(
        './experiments_results/conv_MNIST_bs256_lamb%s.dat' % lamb)
    ax_loss.plot(np.arange(0, 100, 1 / len(file) * 100), file['loss'], m + '-', markevery=int(len(file) / 5))
ax_loss.legend(handles=lines , labels=reversed(tex_labels) ,loc='lower center', bbox_to_anchor=(0.595, 0.68), ncol=3, columnspacing=0.5)
f.set_size_inches(5, 4.5)
plt.show()
plt.savefig('convergence.pdf', bbox_inches='tight')
