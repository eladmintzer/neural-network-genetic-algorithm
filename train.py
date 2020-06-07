"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, Normalizer
import numpy as np
from sklearn.metrics import fbeta_score, make_scorer

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

def f_beta(y_true, y_pred):
    return fbeta_score(y_true,y_pred,beta=0.25)

def get_train():
    """Retrieve the CIFAR dataset and process the data."""
    # Set defaults.
    nb_classes = 1
    batch_size = 64
    input_shape = (120,)

    # Get the data.
    train_file_path = "/home/eladm/project/evaluationAlgorithm/ex2/" \
                      "Dataset/data/my_min_train.csv"
    df_train = pd.read_csv(train_file_path)
    y = df_train.pop('s').values
    X = df_train.values
    print("X!!!!~!")
    print(X.shape)
    print("Y!!!!~!")
    print(y)

    X = X.astype('float64')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
    normalizer = Normalizer().fit(X)
    x_train= normalizer.transform(X_train)
    x_test = normalizer.transform(X_test)


    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)


def compile_model(network, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])#metrics=[f_beta])

    return model

def train_and_score(network, dataset):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    if dataset == 'train':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_train()


    model = compile_model(network, nb_classes, input_shape)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=100,  # using early stopping, so no real limit
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1]  # 1 is accuracy. 0 is loss.
