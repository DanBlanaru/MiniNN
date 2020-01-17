import numpy as np
import gzip
import pickle
import activations as af
import losses
import helpers
import optimizers
import layers
from os import path, getcwd


class Model:
    def __init__(self):
        self.layers = []
        self.loss = losses.mse
        self.optimizer = optimizers.SGD()

    def add_layer(self, new_layer):
        self.layers.append(new_layer)

    def compile(self, loss, optimizer=optimizers.SGD()):
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

    def get_weights(self):
        return [l.weights for l in self.layers]

    def get_gradients(self):
        return [l.gradients for l in self.layers]

    def backpropagate(self, ytrue):

        if self.layers[-1].activation == af.softmax:
            z_last_errors = losses.softmax_loss_derivative(self.layers[-1].y, ytrue)
            last_errors = self.layers[-1].backpropagate(z_last_errors, True)
            for layer in reversed(self.layers[1:-1]):
                last_errors = layer.backpropagate(last_errors)
        else:
            last_errors = self.loss(self.layers[-1].y, ytrue, is_derivative=True)
            for layer in reversed(self.layers[1:]):
                last_errors = layer.backpropagate(last_errors)

    def fit(self, x, y, minibatch_size, epochs, metric_dataset_x=None, metric_dataset_y=None):
        losses = []
        train_wrongs = []
        discrete_y = y.argmax(axis=1)

        for epoch_number in range(1, epochs + 1):
            print("epoch %d" % epoch_number)
            minibatches = helpers.make_minibatches(x, y, minibatch_size)
            for mini_x, mini_y in minibatches:
                train_predicts = self.feed(mini_x)

                # de adaugat ceva pt loss ca metrica
                minibatch_losses = self.loss(train_predicts, mini_y)
                losses.append(minibatch_losses)
                # aici sunt doar metrici

                self.backpropagate(mini_y)
                updated_weights = self.optimizer.recalculate_weights([l.weights for l in self.layers[1:]],
                                                                     [l.gradients for l in self.layers[1:]])

                for l, weights in zip(self.layers[1:], updated_weights):
                    l.weights = weights

                self.optimizer.increment_iterations()

            trainset_predicted_y = self.predict(x)
            wrongs = np.count_nonzero(trainset_predicted_y != discrete_y)
            print(wrongs)
            train_wrongs.append(wrongs)
            losses = []
        return train_wrongs

    def predict(self, dataset):
        last_layers = self.feed(dataset)
        predicted_y = np.argmax(last_layers, axis=1)
        return predicted_y


if __name__ == "__main__":
    model = Model()
    model.add_layer(layers.Input(784))
    model.add_layer(layers.Dense(100, activation=af.relu))
    model.add_layer(layers.Dense(10, activation=af.softmax))
    model.compile(losses.crossentropy, optimizers.Adam())

    # with gzip.open(path.join(getcwd(), 'data', 'mnist.pkl.gz'), 'rb') as f:
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        train_set, validation_set, test_set = pickle.load(f, encoding='latin1')
    n_train = train_set[0].shape[0]
    n_test = test_set[0].shape[0]

    train_set_onehots = helpers.make_onehot_2d(train_set[1], 10)
    model.fit(train_set[0], train_set_onehots, 50, 50)

# de schimbat pt regresie
