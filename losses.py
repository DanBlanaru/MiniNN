import numpy as np


def crossentropy(yhat, ytrue, is_derivative=False):
    if not is_derivative:
        return - np.sum(ytrue * np.log(yhat), axis=1)
    return -(ytrue / yhat)  # am rezolvat-o


def mse(yhat, ytrue, is_derivative=False):
    if not is_derivative:
        return ((yhat - ytrue) ** 2).sum(axis=1) / (2 * len(yhat))
    return (ytrue - yhat) / len(yhat)
