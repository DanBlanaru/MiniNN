import numpy as np


def crossentropy(yhat, ytrue, is_derivative=False):
    if not is_derivative:
        return - np.dot(ytrue, np.log(yhat))
    return -(ytrue / yhat)#ii busita derivata asta varucu


def mse(yhat, ytrue, is_derivative=False):
    if not is_derivative:
        return ((yhat - ytrue)**2).sum(axis=1) / (2 * len(yhat))
    return (ytrue - yhat) / len(yhat)
