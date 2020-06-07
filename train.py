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
from keras import backend as K

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

def f_beta(y_true, y_pred):
    beta = 0.25

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())

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

    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=[f_beta])# metrics=['accuracy'])#

    return model



def train_and_score(network, ds_class):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """

    model = compile_model(network, ds_class.nb_classes, ds_class.input_shape)

    model.fit(ds_class.ga_x_train, ds_class.ga_y_train,
              batch_size=ds_class.batch_size,
              epochs=1000,  # using early stopping, so no real limit
              verbose=0,
              validation_data=(ds_class.ga_x_test, ds_class.ga_y_test),
              callbacks=[early_stopper])

    score = model.evaluate(ds_class.ga_x_test, ds_class.ga_y_test, verbose=0)

    return score[1],model  # 1 is accuracy. 0 is loss.
