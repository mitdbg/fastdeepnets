import numpy as np
from scipy.stats import norm, expon

EPS = 1e-6

def fit(x, maxiter=100, l=1, pi=0.5):
    mu = x.mean()
    sigma = x.std()
    for _ in range(maxiter):
        if pi < EPS or l < EPS:
            l = 1
            pi = 0
            mu = x.mean()
            sigma = x.std()
            tau = np.zeros(shape=x.shape)
            break
        if 1 - pi < EPS or sigma < EPS:
            pi = 1
            mu = 0
            sigma = 1
            l = x.mean()
            tau = np.ones(shape=x.shape)
            break
        p_expon = pi * expon.pdf(x, scale=l)
        p_norm = (1 - pi) * norm.pdf(x, loc=mu, scale=sigma)
        tau = p_expon / (p_expon + p_norm)
        pi, oldpi = tau.mean(), pi
        l = (x * tau).sum() / tau.sum()
        mu = (x * (1 - tau)).sum() / (1 - tau).sum()
        diffs = x - mu
        sigma = np.sqrt((diffs * diffs * (1 - tau)).sum() / (1 - tau).sum())
        if abs(pi - oldpi) < EPS:
            break
    return tau, pi, l, mu, sigma

def exp_norm_sample(size, pi, l, mu, sigma):
    e = expon.rvs(size=size, scale=l)
    n = norm.rvs(size=size, loc=mu, scale=sigma)
    tau = np.random.uniform(0, 1, size=size) < pi
    return e * tau + (1 - tau) * n
