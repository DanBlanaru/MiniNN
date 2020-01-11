import numpy as np
import gzip
import pickle
import math
import activations as af


class Optimizer():
    def __init__(self):
        self.iterations = 0

    def increment_it(self):
        self.iterations += 1


class SGD(Optimizer):
    def __init__(self, lr=0.01, decay=0, momentum=0, nesterov=False):
        super().__init__()
        self.lr = lr
        self.decay = decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.moments = []

    def compile(self, total_weights):
        self.moments = [[np.zeros(weights.shape) for weights in weights_per_layer] for weights_per_layer in
                        total_weights]

    def recalculate_weights(self, total_weights, total_gradients):
        new_weights = []
        lr = self.lr
        if self.decay > 0:
            lr *= (1 / (1 + self.decay * self.iterations))

        for i, (weights_per_layer, gradients_per_layer) in enumerate(
                zip(total_weights, total_gradients)):  # iterate throuh layers
            new_weights_per_layer = []
            for j, (weights, gradients) in enumerate(
                    zip(weights_per_layer, gradients_per_layer)):  # weights then biases
                self.moments[i][j] = self.momentum * self.moments[i][j] - lr * gradients[j]
                if self.nesterov:
                    curr_new_weights = weights + self.momentum * self.moments[i][j] - lr * gradients[j]
                else:
                    curr_new_weights = weights + self.moments[i][j]
                new_weights_per_layer.append(curr_new_weights)
            new_weights.append(new_weights_per_layer)
        return new_weights


class Layer():
    def __init__(self):
        self.weights = []
        self.gradients = []

    def compile(self, input_size):
        pass

    def feed(self, input):
        pass

    def backprop(self, z_errors):
        pass

    def get_weights(self):
        return self.weights

    def get_gradients(self):
        return self.gradients


class Dense(Layer):
    def __init__(self, output_size, activation=af.identity):
        super().__init__()
        self.output_size = output_size
        self.activation = activation
        self.input_size = 0

        self.input = []
        self.z = []
        self.y = []

    def feed(self, input_):
        self.input = input_
        self.z = np.dot(self.input, self.weights[0]) + self.weights[1]
        self.y = self.activation(self.z)

    def compile(self, input_size):
        self.input_size = input_size
        self.weights = [
            np.random.normal(0, math.sqrt(2 / self.input_size), size=(self.input_size, self.output_size)),
            np.zeros(1, self.output_size)
        ]
        self.gradients = [
            np.zeros((self.input_size, self.output_size)),
            np.zeros((1, self.output_size))
        ]
        return self.output_size
