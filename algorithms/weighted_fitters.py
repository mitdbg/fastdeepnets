import numpy as np
from scipy.optimize import newton, bisect
from scipy.special import digamma

EPS = 1e-6

def fit_exp(values, weights):
    return (values * weights).sum() / weights.sum()


def fit_normal(values, weights):
    ws = weights.sum()
    mu = (values * weights).sum() / ws
    diffs = values - mu
    sigma = np.sqrt((diffs * diffs * weights).sum() / ws)
    return mu, sigma


def fit_gamma(values, weights):
    N = weights.sum()
    weighted_value_sum = (values * weights).sum()
    constant = np.log(weighted_value_sum / N) - (weights * np.log(values)).sum() / N
    f = lambda k: np.log(k) - digamma(k) - constant
    k = bisect(f, EPS, 1000)
    theta = weighted_value_sum / (N * k)
    return k, theta
