import numpy as np
from scipy.stats import gamma
from algorithms.weighted_fitters import fit_gamma

EPS = 1e-5

def fit(x, maxiter=100, pi=0.5):
    x = x + EPS
    k1 = 0.5
    theta1 = 1
    k2 = 10
    theta2 = x.mean() / k2
    for _ in range(maxiter):
        p1 = pi * gamma.pdf(x, k1, scale=theta1)
        p2 = (1 - pi) * gamma.pdf(x, k2, scale=theta2)
        tau = p1 / (p1 + p2)
        pi, oldpi = tau.mean(), pi
        print(pi, k1, theta1, k2, theta2)
        k1, theta1 = fit_gamma(x, tau)
        k2, theta2 = fit_gamma(x, 1 - tau)
        if abs(pi - oldpi) < EPS:
            break
    return tau, pi, k1, theta1, k2, theta2

def digamma_sample(size, pi, k1, theta1, k2, theta2):
    p1 = gamma.rvs(k1, size=size, scale=theta1)
    p2 = gamma.rvs(k2, size=size, scale=theta2)
    tau = np.random.uniform(0, 1, size=size) < pi
    return p1 * tau + (1 - tau) * p2
