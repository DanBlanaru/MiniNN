import numpy as np
import gzip
import pickle
import math
import activations as af
import losses as ls


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
                self.moments[i][j] -= self.lr * gradients[j]
                if self.nesterov:
                    curr_new_weights = weights + self.momentum * self.moments[i][j] - self.lr * gradients[j]
                else:
                    curr_new_weights = weights + self.moments[i][j]
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


class Rrop(Optimizer):

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

                if gradients * self.signs[i][j] == 1:
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
                self.squared_gradients[i][j] += np.square(gradients[i][j])
                current_lr = self.lr / np.sqrt(self.squared_gradients[i][j])

                curr_new_weights = weights - current_lr * gradients

                new_weights_per_layer.append(curr_new_weights)
            new_weights.append(new_weights_per_layer)
        return new_weights


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
        self.weights = [np.random.normal(0, math.sqrt(2 / self.in_size), size=(self.in_size, self.out_size)),
                        np.zeros((1, self.out_size))]
        self.gradients = [np.zeros((self.in_size, self.out_size)),
                          np.zeros((1, self.out_size))]
        return self.out_size

    def backpropagate(self, y_errors):
        z_errors = self.activation(y_errors, is_derivative=True)
        y_last_errors = np.dot(z_errors, np.transpose(self.weights[0]))
        weight_grads = np.zeros(self.weights[0].shape)
        bias_grads = np.zeros(self.weights[1].shape)
        for z_iterator, y_iterator in zip(z_errors, y_last_errors):
            bias_grads += z_iterator
            z_iterator = z_iterator.reshape(1, len(z_iterator))
            y_iterator = y_iterator.reshape(len(y_iterator), 1)
            w_grads_curr = np.dot(y_iterator, z_iterator)
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
        self.loss = ls.mse
        self.optimizer = SGD()

    def add_layer(self, new_layer):
        self.layers.append(new_layer)

    def compile(self, loss, optimizer=SGD()):
        self.optimizer = optimizer
        self.loss = loss

        curr_out_size = 0
        for layer in self.layers:
            curr_out_size = layer.compile(curr_out_size)

        self.optimizer.compile([l.weights for l in self.layers[1:]])

    def feed(self, input_):
        for layer in self.layers:
            input_ = layer.feed(input_)
        return input_

    def backpropagate(self, ytrue):
        last_errors = self.loss(self.layers[-1].y, ytrue, is_derivative=True)
        for layer in reversed(self.layers[1:]):
            last_errors = layer.backpropagate(last_errors)

    def make_onehot_1d(self, true_class, total_classes):
        y = np.zeros(total_classes, dtype=np.float)
        y[true_class] = 1
        return y

    def make_onehot_2d(self, true_classes, total_classes):
        y = np.zeros((true_classes.shape[0], total_classes))
        for i, true_class in enumerate(true_classes):
            y[i][true_class] = 1
        return y

    def make_minibatches(self, dataset, target, minibatch_size):
        x = np.array(dataset)
        y = np.array(target)
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)

        x = x[indices]
        y = y[indices]

        full_minibatches = x.shape[0] // minibatch_size

        batches = [(x[i * minibatch_size: (i + 1) * minibatch_size],
                    y[i * minibatch_size: (i + 1) * minibatch_size])
                   for i in range(full_minibatches)]

        if x.shape[0] % minibatch_size:
            batches.append((x[minibatch_size * full_minibatches:], y[minibatch_size * full_minibatches:]))

        return batches

    def fit(self, x, y, minibatch_size, epochs):
        losses = []
        train_wrongs = []

        for epoch_number in range(1, epochs + 1):
            minibatches = self.make_minibatches(x, y, minibatch_size)
            for mini_x, mini_y in minibatches:
                mini_y_onehots = self.make_onehot_2d(mini_y, 10)
                # am hardcodat 10 pt mnist, nu e tocmai ok da mnaaaaaaa, e ceva cu shape de ultimu layer

                train_predicts = self.feed(mini_x)

                # de adaugat ceva pt loss ca metrica
                minibatch_losses = self.loss(train_predicts, mini_y_onehots)
                predicted_classes = np.argmax(train_predicts, axis=1)
                misclassified = np.count_nonzero(predicted_classes != mini_y)
                # aici sunt doar metrici

                self.backpropagate(mini_y_onehots)
                updated_weights = self.optimizer.recalculate_weights([l.weights for l in self.layers[1:]],
                                                                     [l.gradients for l in self.layers[1:]])
                for l, weights in zip(self.layers[1:], updated_weights):
                    l.weights = weights
                self.optimizer.increment_iterations()

            trainset_predicted_y = self.predict(x)
            wrongs = np.count_nonzero(trainset_predicted_y != y)
            print(wrongs)
            train_wrongs.append(wrongs)
        return train_wrongs

        # apelezi generate_batches
        # feed
        # aplici loss
        # apelezi backprop
        # apelezi recalc pt fiecare layer
        #

        pass

    def predict(self, dataset):
        last_layers = self.feed(dataset)
        predicted_y = np.argmax(last_layers, axis=1)
        return predicted_y


model = Model()
model.add_layer(Input(784))
model.add_layer(Dense(100, activation=af.relu))
model.add_layer(Dense(10, activation=af.sigmoid))
model.compile(ls.mse, SGD())

with gzip.open('D:\\F\\AI\\Proiect\\data\\mnist.pkl.gz', 'rb') as f:
    train_set, _, test_set = pickle.load(f, encoding='latin1')
n_train = train_set[0].shape[0]
n_test = test_set[0].shape[0]

model.fit(train_set[0], train_set[1], 50, 10)

# To do:
# all backprop Dan gata!!!!!!
# model.compile Dan gata!!!!!!
# RMSprop, Vivi gata!!!!!
# model.generate_batches gata!!!!!!
# model.fit Dan
# model.predict

# mutat make_minibatches si make_onehot in file separat
# fit_transform pe n dimensiuni
# de schimbat pt regresie


# dropout layer NOPEEE
