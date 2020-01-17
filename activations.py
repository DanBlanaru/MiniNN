import numpy as np
from sys import exit


def relu(x, is_derivative=False):
    try:
        y = x * (x > 0).astype(np.float64)
        if not is_derivative:
            return y
        return (x > 0).astype(np.float64)
    except:
        print(x)
        exit()


def sigmoid(x, is_derivative=False):
    try:
        x = np.minimum(x, 20)
        x = np.maximum(x, -20)
        y = 1. / (1. + np.exp(-x))
        if not is_derivative:
            return y
        return (1 - y) * y
    except:
        print("sigmoid", x)
        exit()


def softmax(x, is_derivative=False):
    _x = x - np.max(x, axis=1)[:, np.newaxis]
    y = np.exp(_x) / np.sum(np.exp(_x), axis=1)[:, np.newaxis]
    if not is_derivative:
        return y
    return (np.identity(len(x)) - y).transpose() * y


def identity(x, is_derivative=False):
    y = x
    if not is_derivative:
        return y
    return np.ones(x.shape)
