"""Class that represents the network to be evolved."""
import random
import logging
from train import train_and_score
from keras.callbacks import EarlyStopping

early_stopper = EarlyStopping(patience=5)

class Network():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices=None,):
        """Initialize our network.

        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                nb_neurons (list): [64, 128, 256]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['relu', 'elu']
                optimizer (list): ['rmsprop', 'adam']
        """
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters
        self.model = None

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        """Set network properties.

        Args:
            network (dict): The network parameters

        """
        self.network = network

    def train(self, ds_class):
        """Train the network and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        """
        if self.accuracy == 0.:
            self.accuracy,self.model = train_and_score(self.network, ds_class)

    def print_network(self):
        """Print out a network."""
        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))

    def WriteModelToFile(self):
        print("save net to model")
        print("Network accuracy: %.2f%%" % (self.accuracy * 100))
        print(self.network)
        self.print_network()
        self.model.save("model.h5")

    def train_final_net(self, ds_class):
        """Train the model, return test loss.

        Args:
            network (dict): the parameters of the network
            dataset (str): Dataset to use for training/evaluating

        """
        print("train the best Network..")
        self.model.fit(ds_class.f_x_train, ds_class.f_y_train,
                       batch_size=ds_class.batch_size,
                       epochs=10000,  # using early stopping, so no real limit
                       verbose=0,
                       validation_data=(ds_class.f_x_test, ds_class.f_y_test),
                       callbacks=[early_stopper])


        score = self.model.evaluate(ds_class.f_x_test, ds_class.f_y_test, verbose=0)

        return score[1]