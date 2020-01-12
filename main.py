import numpy as np
import gzip
import pickle
import math
import activations as af
import losses as ls


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
        return self.y

    def compile(self, input_size):
        self.in_size = input_size
        self.weights = [np.random.normal(0, math.sqrt(2 / self.in_size), size=(self.in_size, self.out_size)),
                        np.zeros(1, self.out_size)]
        self.gradients = [np.zeros((self.in_size, self.out_size)),
                          np.zeros((1, self.out_size))]
        return self.out_size

    def backpropagate(self, y_errors):
        z_errors = self.activation(y_errors, is_derivative=True)
        y_last_errors = np.dot(z_errors, np.transpose(z_errors))
        weight_grads = np.zeros(self.weights[0].shape)
        bias_grads = np.zeros(self.weights[1].shape)
        for z_iterator, y_iterator in zip(z_errors, y_last_errors):
            bias_grads += z_iterator
            z_iterator = z_iterator.reshape(len(z_iterator), 1)
            y_iterator = y_iterator.reshape(1, len(y_iterator))
            w_grads_curr = np.dot(z_iterator, y_iterator)
            weight_grads += w_grads_curr

        weight_grads /= len(y_errors)
        bias_grads /= len(y_errors)
        return y_last_errors


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

    def compile(self, loss, optimizer=SGD):
        self.optimizer = optimizer
        self.loss = loss

        curr_out_size = 0
        for layer in self.layers:
            curr_out_size = layer.compile(curr_out_size)

    def feed(self, input_):
        for layer in self.layers:
            input_ = layer.feed(input_)
        return input

    def backpropagate(self, ytrue):
        last_errors = self.loss(self.layers[-1].y, ytrue, is_derivative=True)
        for layer in reversed(self.layers[1:]):
            last_errors = layer.backpropagate(last_errors)

    def make_minibatches(self, dataset, target, minibatch_size):
        perm = np.arange(len(target))
        np.random.shuffle(perm)

        # returneaza o lista, fiecare element are minibatch_size elemente din dataset
        pass

    def fit(self):
        # apelezi generate_batches
        # feed
        # aplici loss
        # apelezi backprop
        # apelezi recalc pt fiecare layer
        #
        pass

# To do:
# all backprop Dan gata!!!!!!
# model.compile Dan gata!!!!!!
# RMSprop, Vivi
# mode.generate_batches
# model.fit Dan
# model.predict


# dropout layer
