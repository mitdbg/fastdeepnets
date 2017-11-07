def stable_logexp(factor, k):
    f = factor * k
    if f.data.cpu().numpy()[0] < 0:
        return (f.exp() + 1).log() / k
    else:
        return (1 + (-f).exp()).log() / k + factor

def integral(k, x_0, size):
        return stable_logexp(x_0, k) - stable_logexp(x_0 - size, k)
