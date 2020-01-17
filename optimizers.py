import numpy as np


class Optimizer:
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
        if self.decay > 0:
            self.lr *= (1 / (1 + self.decay * self.iterations))

        for i, (weights_per_layer, gradients_per_layer) in enumerate(
                zip(total_weights, total_gradients)):  # iterate throuh layers
            new_weights_per_layer = []
            for j, (weights, gradients) in enumerate(
                    zip(weights_per_layer, gradients_per_layer)):  # weights then biases
                self.moments[i][j] = self.momentum * self.moments[i][j]
                self.moments[i][j] -= self.lr * gradients
                if self.nesterov:
                    curr_new_weights = weights - self.momentum * self.moments[i][j] - self.lr * gradients[j]
                else:
                    curr_new_weights = weights - self.moments[i][j]
                new_weights_per_layer.append(curr_new_weights)
            new_weights.append(new_weights_per_layer)
        return new_weights


class RMSProp(Optimizer):

    def __init__(self, learning_rate=0.001, decay=0., rho=0.9, epsilon=1e-07):
        super().__init__()
        self.lr = learning_rate
        self.decay = decay
        self.rho = rho
        self.epsilon = epsilon
        self.moments = []

    def compile(self, total_weights):
        self.moments = [[np.zeros(weight.shape)
                         for weight in weights] for weights in total_weights]

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
                self.moments[i][j] = self.rho * self.moments[i][j] + \
                                     (1. - self.rho) * np.square(gradients)
                curr_new_weights = weights - lr * gradients / \
                                   (np.sqrt(self.moments[i][j]) + self.epsilon)
                new_weights_per_layer.append(curr_new_weights)
            new_weights.append(new_weights_per_layer)
        return new_weights


class Rprop(Optimizer):

    def __init__(self, incFactor=1.2, decFactor=0.5, stepSizeMax=50, stepSizeMin=1e-6):
        super().__init__()
        self.incFactor = incFactor
        self.decFactor = decFactor
        self.stepSizeMax = stepSizeMax
        self.stepSizeMin = stepSizeMin
        self.steps = []
        self.signs = []

    def compile(self, total_weights):
        self.steps = [[np.zeros(weight.shape) + 0.001
                       for weight in weights] for weights in total_weights]
        self.signs = [[np.ones(weight.shape)
                       for weight in weights] for weights in total_weights]

    def recalculate_weights(self, total_weights, total_gradients):
        new_weights = []

        for i, (weights_per_layer, gradients_per_layer) in enumerate(
                zip(total_weights, total_gradients)):  # iterate throuh layers
            new_weights_per_layer = []
            for j, (weights, gradients) in enumerate(
                    zip(weights_per_layer, gradients_per_layer)):  # weights then biases
                if np.all(gradients[i][j] * self.signs[i][j]) == 1:
                    self.steps[i][j] = min(self.stepSizeMax, self.steps[i][j] * self.incFactor)
                else:
                    self.steps[i][j] = max(self.stepSizeMin, self.steps[i][j] * self.decFactor)

                self.signs[i][j] = np.sign(gradients)

                curr_new_weights = weights - np.sign(total_gradients) * self.steps[i][j]

                new_weights_per_layer.append(curr_new_weights)
            new_weights.append(new_weights_per_layer)
        return new_weights


class Adagrad(Optimizer):
    def __init__(self, learning_rate=0.001, decay=0.):
        super().__init__()
        self.lr = learning_rate
        self.decay = decay
        self.squared_gradients = []

    def compile(self, total_weights):
        self.squared_gradients = [[np.zeros(weights.shape) for weights in weights_per_layer]
                                  for weights_per_layer in total_weights]

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
                self.squared_gradients[i][j] += np.square(gradients)
                current_lr = self.lr / np.sqrt(self.squared_gradients[i][j])

                curr_new_weights = weights - current_lr * gradients

                new_weights_per_layer.append(curr_new_weights)
            new_weights.append(new_weights_per_layer)
        return new_weights



class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        super().__init__()
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

    def compile(self, total_weights):
        self.moments = [[np.zeros(weights.shape) for weights in weights_per_layer]
                                  for weights_per_layer in total_weights]
        self.squared_gradients = [[np.zeros(weights.shape) for weights in weights_per_layer]
                                  for weights_per_layer in total_weights]
        self.t = 0
    
    def recalculate_weights(self, total_weights, total_gradients):
        new_weights = []
        lr = self.lr

        for i, (weights_per_layer, gradients_per_layer) in enumerate(
                zip(total_weights, total_gradients)):  # iterate throuh layers
            new_weights_per_layer = []
            for j, (weights, gradients) in enumerate(
                    zip(weights_per_layer, gradients_per_layer)):  # weights then biases
                self.moments[i][j] = self.beta1 * self.moments[i][j] + \
                                (1. - self.beta1) * gradients

                self.squared_gradients[i][j] = self.beta2 * self.squared_gradients[i][j] + \
                                        (1. - self.beta2) * np.square(gradients)
                self.t += 1

                curr_momentum =self.moments[i][j] / (1 - self.beta1 ** self.t)
                curr_squared_gradient = self.squared_gradients[i][j] / (1 - self.beta2 ** self.t)
                
                curr_new_weights = weights - lr * (curr_momentum / (np.sqrt(curr_squared_gradient) + 1e-6))

                new_weights_per_layer.append(curr_new_weights)
            new_weights.append(new_weights_per_layer)
        return new_weights

