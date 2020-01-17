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
    model.fit(train_set[0], train_set_onehots, 50, 50)


def run_regression():
    df = np.array(pd.read_csv('data/Dataset/Training/Features_Variant_1.csv'))
    model = Model.Model()
    model.add_layer(layers.Input(53))
    model.add_layer(layers.Dense(20, activation=af.relu))
    model.add_layer(layers.Dense(1, activation=af.identity))
    model.compile(losses.crossentropy, optimizers.Adam())


    input_set = np.array([x[:-1] for x in df])
    output_set = np.array([x[-1] for x in df]).reshape(len(input_set),1)
    Model.save_model(model,"test")
    tmp = Model.load_model("test")
    tmp.fit(input_set,output_set,50,50)
    #model.fit(input_set,output_set,50,50)



    



if __name__ == "__main__":
    run_regression()

# de schimbat pt regresie