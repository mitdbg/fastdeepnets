import numpy as np
from scipy.optimize import newton, bisect
from scipy.special import digamma


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
    def f(k):
        try:
            return np.log(k) - digamma(k) - constant
        except:
            print(k)
    a = 0.5
    b = 1.0
    while np.isfinite(b) and f(a) * f(b) > 0: # While they have the same sign
        if f(a) < 0:
            a /= 2
        if f(b) > 0:
            b *= 2
    if np.isfinite(b):
        k = bisect(f, a, b)
    else:
        k = 1
    theta = weighted_value_sum / (N * k)
    return k, theta
