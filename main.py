import numpy as np
import gzip
import pickle
import math
import activations as af


class Optimizer():
    def __init__(self):
        self.iterations = 0

    def increment_iterations(self):
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

    def backpropagate(self, z_errors):
        pass

    def get_weights(self):
        return self.weights

    def get_gradients(self):
        return self.gradients


class Dense(Layer):
    def __init__(self, output_size, activation=af.identity):
        super().__init__()
        self.out_size = output_size
        self.activation = activation
        self.in_size = 0

        self.input = []
        self.z = []
        self.y = []

    def feed(self, input_):
        self.input = input_
        self.z = np.dot(self.input, self.weights[0]) + self.weights[1]
        self.y = self.activation(self.z)

    def compile(self, input_size):
        self.in_size = input_size
        self.weights = [np.random.normal(0, math.sqrt(2 / self.in_size), size=(self.in_size, self.out_size)),
                        np.zeros(1, self.out_size)]
        self.gradients = [np.zeros((self.in_size, self.out_size)),
                          np.zeros((1, self.out_size))]
        return self.out_size

    def backpropagate(self, z_errors):
        pass


class Input(Layer):
    def __init__(self, input_size, activation=af.identity):
        super().__init__()

        self.input_size = input_size
        self.output_size = input_size
        self.activation = activation
        self.weights = []
        self.gradients = []

    def compile(self, _):
        return self.output_size

    def feed(self, input_):
        return self.activation(input_)


class Model:
    def __init__(self):
        self.layers = []
        self.loss = af.mse
        self.optimizer = SGD

    def add_layer(self, new_layer):
        self.layers.append(new_layer)

    def compile(self, loss, optimizer):
        pass

    def feed(self, input):
        for layer in self.layers:
            input = layer.feed(input)
        return input

    def backpropagate(self, ytrue):
        pass

    def make_minibatches(self,dataset,target,minibatch_size):


    def fit(self):
        pass

# To do:
# model.fit
# all backprop
# dropout layer
# model.fit
# model.predict
