# Use modules to keep code organized
from src import params
from src import LSTM
import scipy.io as sio
import numpy as np
import sys
import os


def read_inputs():
    if len(sys.argv) == 11:
        n_nodes, n_epochs, window_size, stride = [int(i) for i in sys.argv[1:5]]
        alpha, decay, data_split, dropout      = [float(i) for i in sys.argv[5:9]]
        train_loc, ac_fun                      = sys.argv[9:]
        return n_nodes, n_epochs, window_size, stride, \
               alpha, decay, data_split, dropout, train_loc, ac_fun
    else:
        print ("[[WARNING]] Incorrect number of arguments specified.")
        print ("[[WARNING]] Using default arguments.")
        return [100, 30, 15, 2, 0.05, 1e-9, 0.8, 0, "../../data/preprocessed_data/Experiment_I/dataset_used_in_experiments/all_ds64_zscore.mat", "relu"]        

if __name__ == "__main__":

    (n_nodes, n_epochs, window_size, stride, alpha, decay, \
     data_split, dropout_ratio, train_location, ac_fun) =  \
        read_inputs()


    # Load settings
    settings = params.Settings(window_size, stride, n_nodes, \
               alpha, decay, n_epochs, data_split, dropout_ratio, \
               train_location, ac_fun)
    # Load data
    data = params.Data(settings, train_location)

    # Initiate the LSTM network using data and settings
    network = LSTM.LSTM_network(data, settings)
    # Train the network
    network.train()
    # Test the network
    network.test(data.test_data, data.test_labels)

    # Save results
    network.save_results()
