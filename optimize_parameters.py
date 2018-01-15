############################################################################################################
####################  This class to make choice for the right parameters  ##################################
############################################################################################################

# imports libraries
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

#******* Keras ********#
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model

#******* sickit optimizer ********#
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args


# This is the search-dimension for the learning-rate.
dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',
                         name='learning_rate')

# This is the search-dimension for the number of dense layers in the
# neural network
dim_num_dense_layers = Integer(low=1, high=5, name='num_dense_layers')

# This is the search-dimension for the number of nodes for each dense layer.
dim_num_dense_nodes = Integer(low=5, high=512, name='num_dense_nodes')

# This is the search-dimension for the activation-function.
dim_activation = Categorical(categories=['relu', 'sigmoid'],
                             name='activation')

# We then combine all these search-dimensions into a list.
dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_dense_nodes,
              dim_activation]

# The default list of hyper-parameters 
default_parameters = [1e-5, 1, 16, 'relu']

# The path of the best model
path_best_model = 'best_model.h5'


# For storing logs for each model evaluated 
def log_dir_name(learning_rate, num_dense_layers,
                 num_dense_nodes, activation):

    # The dir-name for the TensorBoard log-dir.
    s = "./19_logs/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}/"

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate,
                       num_dense_layers,
                       num_dense_nodes,
                       activation)

    return log_dir







