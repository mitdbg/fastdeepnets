import numpy as np

seq = [2, 3, 9, 4, 6, 8, 7, 5]

def max_expectation(values, measure=None):
    if measure is None:
        measure = values
    values = np.array(values)
    measure = np.array(measure)
    order = np.argsort(values)
    values = values[order]
    measure = measure[order]
    result = []
    n = len(values)
    s_prev = np.ones(n) / n
    result.append((s_prev * measure).sum())
    for t in range(1, n):
        factor = 1 / (n - t)
        c_sum = 0
        for k in range(0, n):
            c_value = s_prev[k]
            s_prev[k] = factor * (c_sum + max(0, k - t + 1) * c_value)
            c_sum += c_value
        result.append((s_prev * measure).sum())
    return result


