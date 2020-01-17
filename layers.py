import numpy as np
import activations as af

class Layer:
    def __init__(self):
        self.weights = []
        self.gradients = []

    def compile(self, input_size):
        pass

    def feed(self, input_):
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
        self.weights = [np.random.normal(0, np.sqrt(2 / self.in_size), size=(self.in_size, self.out_size)),
                        np.zeros((1, self.out_size))]
        self.gradients = [np.zeros((self.in_size, self.out_size)),
                          np.zeros((1, self.out_size))]
        return self.out_size

    def backpropagate(self, y_errors, is_already_z=False):
        if is_already_z:
            z_errors = y_errors
        else:
            z_errors = y_errors * self.activation(self.y, is_derivative=True)
        y_last_errors = np.dot(z_errors, np.transpose(self.weights[0]))

        self.y_last_errors = np.copy(y_last_errors)
        self.z_errors = np.copy(z_errors)

        weight_grads = np.zeros(self.weights[0].shape)
        bias_grads = np.zeros(self.weights[1].shape)
        for z_iterator, y_iterator in zip(z_errors, self.input):
            bias_grads += z_iterator
            z_iterator = z_iterator.reshape(1, len(z_iterator))
            y_iterator = y_iterator.reshape(len(y_iterator), 1)
            w_grads_curr = np.dot(y_iterator, z_iterator)
            weight_grads += w_grads_curr

        weight_grads /= len(y_errors)
        bias_grads /= len(y_errors)

        self.gradients[0] = weight_grads
        self.gradients[1] = bias_grads
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