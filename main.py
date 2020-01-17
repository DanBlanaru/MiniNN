import model as Model
import layers
import activations as af
import losses
import optimizers
import helpers
import gzip
import pickle
import pandas as pd
import numpy as np


def classification_metric_accuracy(pred, true):
    true = np.argmax(true, axis=1)
    pred = np.argmax(pred,axis=1)
    true = np.squeeze(true)
    pred = np.squeeze(pred)
    wrongs = np.count_nonzero(true - pred)
    return wrongs


def regression_metric_mse(pred, true):
    return losses.mse(pred, true)


def run_classification():
    model = Model.Model()
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
    model.fit(train_set[0], train_set_onehots, 50, 50, metric_callback=classification_metric_accuracy)


def run_regression():
    # if __name__ == "__main__":
    df = np.array(pd.read_csv('data/Dataset/Training/Features_Variant_1.csv'))
    model = Model.Model()
    model.add_layer(layers.Input(53))
    model.add_layer(layers.Dense(20, activation=af.relu))
    model.add_layer(layers.Dense(1, activation=af.sigmoid))
    model.compile(losses.mse, optimizers.Adam())

    input_set = np.array([x[:-1] for x in df])
    output_set = np.array([x[-1] for x in df]).reshape(len(input_set), 1)
    # Model.save_model(model, "test")
    # tmp = Model.load_model("test")
    # tmp.fit(input_set, output_set, 50, 50, metric_callback=regression_metric_mse)
    input_set = helpers.standard_scaler(input_set)
    output_set = helpers.standard_scaler(output_set)

    np.seterr(all="raise")
    model.fit(input_set, output_set, 50, 50, metric_callback=regression_metric_mse)
    # return model


if __name__ == "__main__":
    m = run_regression()

# de schimbat pt regresie
