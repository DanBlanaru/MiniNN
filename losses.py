import numpy as np


def crossentropy(yhat, ytrue, is_derivative=False):
    if not is_derivative:
        return - np.dot(ytrue, np.log(yhat))
    return -(ytrue / yhat)


def mse(yhat, ytrue, is_derivative=False):
    if not is_derivative:
        return np.dot(yhat - ytrue, yhat - ytrue) / (2 * len(yhat))
    return (ytrue - yhat) / len(yhat)
