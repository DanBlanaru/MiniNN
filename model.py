import numpy as np
import activations as af
import losses
import helpers
import optimizers
import layers
import dill


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

        # if self.layers[-1].activation == af.softmax:
        if True:
            z_last_errors = losses.softmax_loss_derivative(self.layers[-1].y, ytrue)
            last_errors = self.layers[-1].backpropagate(z_last_errors, True)
            for layer in reversed(self.layers[1:-1]):
                last_errors = layer.backpropagate(last_errors)
        else:
            last_errors = self.loss(self.layers[-1].y, ytrue, is_derivative=True)
            for layer in reversed(self.layers[1:]):
                last_errors = layer.backpropagate(last_errors)

    def fit(self, x, y, minibatch_size, epochs, metric_dataset_x=None, metric_dataset_y=None, metric_callback=None):
        losses = []
        train_wrongs = []

        for epoch_number in range(1, epochs + 1):
            print("epoch %d" % epoch_number)
            minibatches = helpers.make_minibatches(x, y, minibatch_size)
            for mini_x, mini_y in minibatches:
                train_predicts = self.feed(mini_x)

                self.backpropagate(mini_y)
                updated_weights = self.optimizer.recalculate_weights([l.weights for l in self.layers[1:]],
                                                                     [l.gradients for l in self.layers[1:]])

                for l, weights in zip(self.layers[1:], updated_weights):
                    l.weights = weights

                self.optimizer.increment_iterations()

            if metric_callback:
                trainset_predicted_y = self.predict(x)[..., np.newaxis]
                metric = metric_callback(trainset_predicted_y, y)
                # metric = 0
                print(metric)
                train_wrongs.append(metric)
            # wrongs = np.count_nonzero(trainset_predicted_y != discrete_y)
            # print(wrongs)
        return train_wrongs

    def predict(self, dataset):
        last_layers = self.feed(dataset)
        predicted_y = np.argmax(last_layers, axis=1)
        return predicted_y


def save_model(model, filename):
    dill.dump(model, file=open(filename, 'wb'))


def load_model(filename):
    model = dill.load(open(filename, 'rb'))
    return model
