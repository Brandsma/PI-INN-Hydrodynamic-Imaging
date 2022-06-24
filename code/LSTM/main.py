# Use modules to keep code organized
import os
import sys

import numpy as np
import scipy.io as sio

from src import LSTM, params


def read_inputs():
    n_nodes = 128
    n_epochs = 30
    window_size = 16
    stride = 2
    alpha = 0.05
    decay = 1e-9
    shuffle_data = True
    data_split = 0.8
    dropout = 0
    train_loc = "../../data/a10_theta0.npy"
    ac_fun = "relu"
    return n_nodes, n_epochs, window_size, stride, \
        alpha, decay, shuffle_data, data_split, dropout, train_loc, ac_fun


if __name__ == "__main__":

    (n_nodes, n_epochs, window_size, stride, alpha, decay, \
     shuffle_data, data_split, dropout_ratio, train_location, ac_fun) =  \
        read_inputs()

    # Load settings
    settings = params.Settings(window_size, stride, n_nodes, \
                               alpha, decay, n_epochs, shuffle_data, data_split, dropout_ratio, \
               train_location, ac_fun)
    # Load data
    data = params.Data(settings, train_location)

    # Initiate the LSTM network using data and settings
    network = LSTM.LSTM_network(data, settings)
    network.model.summary()
    # Train the network
    network.train()
    # Test the network
    network.test(data.test_data, data.test_labels)

    network.save_model(
        f"./trained_models/win{window_size}_stride{stride}_epochs{n_epochs}_dropout{dropout_ratio}_latest"
    )

    # Save results
    network.save_results()
