"""Entry point to evolving the neural network. Start here."""
import logging
from optimizer import Optimizer
from tqdm import tqdm
import sys
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

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    filename='log.txt'
)

class DataSetClass():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """

    def __init__(self, file_path):
        # Set defaults.
        self.nb_classes = 1
        self.batch_size = 1024
        self.input_shape = (120,)

        l = ["s"]
        l.extend([f",f{i + 1}" for i in range(120)])
        # Get the data.
        df_train = pd.read_csv(file_path, names=l)
        y = df_train.pop('s').values
        X = df_train.values
        self.normalizer = Normalizer().fit_transform(X)

        print("get_train Again!!!!~!")

        X = X.astype('float64')

        f_X, ga_X, f_y, ga_y = train_test_split(X, y, test_size = 0.2, random_state = 42)
        self.f_x_train, self.f_x_test, self.f_y_train, self.f_y_test = train_test_split(f_X, f_y, test_size=0.1, random_state=44)
        self.ga_x_train, self.ga_x_test, self.ga_y_train, self.ga_y_test = train_test_split(ga_X, ga_y, test_size=0.1, random_state=46)


def train_networks(networks, dataset):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset)
        pbar.update(1)
    pbar.close()

def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.

    Args:
        networks (list): List of networks

    Returns:
        float: The average accuracy of a population of networks.

    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

def generate(generations, population, nn_param_choices, i_file_name,o_file_name):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)
    ds_class = DataSetClass(i_file_name)
    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))
        print("***Doing generation %d of %d***" %
                     (i + 1, generations))

        # Train and get accuracy for networks.
        train_networks(networks, ds_class)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-'*80)
        print("Generation average: %.2f%%" % (average_accuracy * 100))
        print('-' * 80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_networks(networks[:-5])

    # Write the network to file
    if i == generations - 1:
        networks[-1].train_final_net(ds_class)
        networks[-1].WriteModelToFile()

def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()

def main(i_file_name,o_file_name):
    """Evolve a network."""
    generations = 20  # Number of times to evole the population.
    population = 10  # Number of networks in each generation.


    nn_param_choices = {
        'nb_neurons': [64, 128, 512, 1024,2048,4096],
        'nb_layers': [2, 1, 4,5],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                      'adadelta', 'adamax', 'nadam'],
    }
    # nn_param_choices = {
    #     'nb_neurons': [1024],
    #     'nb_layers': [4],
    #     'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
    #     'optimizer': ['rmsprop', 'adam'],
    # }
    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    generate(generations, population, nn_param_choices, i_file_name,o_file_name)

if __name__ == '__main__':
    i_file_name = sys.argv[1]
    o_file_name = sys.argv[2]
    main(i_file_name,o_file_name)
