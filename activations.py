import numpy as np


def relu(x, is_derivative=False):
    y = x * (x > 0).astype(np.float64)
    if not is_derivative:
        return y
    return (x > 0).astype(np.float64)


def sigmoid(x, is_derivative=False):
    y = 1. / (1. + np.exp(-x))
    if not is_derivative:
        return y
    return (1 - y) * y


def softmax(x, is_derivative=False):
    shifted = x - np.max(x)
    exps = np.exp(shifted)
    y = exps / exps.sum()
    if not is_derivative:
        return y
    return (np.identity(len(x)) - y).transpose() * y


def identity(x, is_derivative=False):
    y = x
    if not is_derivative:
        return y
    return np.ones(x.shape)
