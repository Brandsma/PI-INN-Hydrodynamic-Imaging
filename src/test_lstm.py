# Use modules to keep code organized
import os
from pathlib import Path

import tensorflow as tf

from lib import LSTM, params


def get_scratch_dir():
    # Find the folder where the data should be found and should be saved
    data_folder_key = "SCRATCHDIR"
    SCRATCHDIR = os.getenv(data_folder_key)
    if SCRATCHDIR is None:
        print(f"{data_folder_key} environment variable does not exist")
        exit(1)
    if SCRATCHDIR[-1] == "/":
        SCRATCHDIR = SCRATCHDIR[:-1]

    return SCRATCHDIR


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
    # train_loc = get_scratch_dir() + "/data/combined.npy"
    train_loc = "../data/simulation_data/combined.npy"
    ac_fun = "relu"
    return n_nodes, n_epochs, window_size, stride, \
        alpha, decay, shuffle_data, data_split, dropout, train_loc, ac_fun


def main(a, w, new_model):
    train_loc = f"../data/simulation_data/a{a}_normw{w}_theta0.npy"
    (n_nodes, n_epochs, window_size, stride, alpha, decay, \
     shuffle_data, data_split, dropout_ratio, train_location, ac_fun) =  \
        read_inputs()
    train_location = train_loc

    # Load settings
    settings = params.Settings(window_size, stride, n_nodes, \
                               alpha, decay, n_epochs, shuffle_data, data_split, dropout_ratio, \
               train_location, ac_fun)
    # Load data
    data = params.Data(settings, train_location)

    data.normalize()

    network = LSTM.LSTM_network(data, settings)
    network.model = new_model

    # Test the network
    network.test(data.test_data,
                 data.test_labels,
                 dirname=f"../results/a{a}_w{w}")
    # Save results
    network.save_results(dirname=f"../results/a{a}_w{w}")


if __name__ == "__main__":
    a_set = [10, 20, 30, 40, 50]
    w_set = [10, 20, 30, 40, 50]

    model = tf.keras.models.load_model(
        '../data/trained_models/win16_stride2_epochs5_dropout0_latest')

    count = 1
    for a in a_set:
        for w in w_set:
            print(f"Running {count}/{len(a_set) * len(w_set)}...")
            main(a, w, model)
            count += 1
