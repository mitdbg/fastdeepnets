import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch.nn.functional import sigmoid
from algorithms.sigmoid_integral import integral
from models.MNIST_1h_flexible import MNIST_1h_flexible

EPSILON = 0.01
FACTOR = np.log(EPSILON / (1 - EPSILON))

class MNIST_1h_flexible_random(MNIST_1h_flexible):

    def get_scaler(self):
        gray_zone_size = np.random.uniform(1, 30)
        k = float(-2 * FACTOR / gray_zone_size)
        return sigmoid(-k * (self.range - self.x_0))

