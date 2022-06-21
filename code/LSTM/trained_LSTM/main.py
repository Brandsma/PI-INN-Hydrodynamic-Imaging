import os
import sys

import numpy as np
import scipy.io as sio
import tensorflow as tf


class Data:
    """
    Data class constructor
    - expects one or two arguments; location argument is optional.
    - if no location is given to the constructor, the default location
      is used;
    - extracts number of sensors and number of samples from data.
    - splits data in train and test split. ratio depends on the settings.
    """
    def __init__(self, settings, location):
        print("Loading dataset from " + location + "...")

        self.settings = settings
        # load data
        if os.path.splitext(location)[-1] == ".npy":
            d_1, l_1, d_2, l_2, d_3, l_3, t_1, t_2, t_3 = self.__load_data_numpy(
                location, settings.data_split)
        else:
            print(
                f"Error, file extension {os.path.splitext(location[-1])} is unknown and cannot be loaded for data"
            )
            exit(1)

        # assign to data members.
        self.train_data = d_1
        self.train_labels = l_1
        self.test_data = d_2
        self.test_labels = l_2
        self.val_data = d_3
        self.val_labels = l_3

        self.test_timestamp = t_1
        self.val_timestamp = t_2
        self.test_timestamp = t_3

        # extract dimensionality and length of data.
        self.n_inputs = np.shape(self.train_data[0])[1]
        self.n_outputs = np.shape(self.train_labels[0])[1]
        self.n_datapoints = len(self.train_data)

        print("Dataset loaded.")

    """
    Data::__load_data_numpy(location):
    - private function
    - loads data provided by location, and returns split
      train, test and validation data and label sets.
    """

    def __load_data_numpy(self, file_location, split_ratio):
        print("Loading numpy data")
        # Extract name of numpy struct from file and load it
        base_name = os.path.splitext(file_location)
        labels = np.load(f"{base_name[0]}_labels{base_name[-1]}")
        data = np.load(file_location)
        timestamp = np.load(f"{base_name[0]}_timestamp{base_name[-1]}")

        # Split data into train, test and validation sets.
        train_d, train_l, test_d, test_l, val_d, val_l, train_t, val_t, test_t = \
            self.__split_data(data, labels, timestamp, split_ratio)

        return train_d, train_l, test_d, test_l, val_d, val_l, train_t, val_t, test_t

    """
    Data::__split_data(data, labels, train_test_ratio):
    - private function.
    - function which splits data into training and test set, depending on given
      train_test_ratio.
    """

    def __split_data(self, data, labels, timestamp, train_test_ratio):
        # Take the number of data points
        n_entries = len(data)

        # Generate a random permutation of indices
        perm = np.random.permutation(n_entries)

        # Determine indices for train, validation and test sets
        train_idx = int(train_test_ratio * n_entries)
        val_idx = n_entries - int((n_entries - train_idx) / 2)
        test_idx = n_entries

        # Take subsets by using the permutation array as index
        train_data = data[perm[0:train_idx]]
        val_data = data[perm[train_idx:val_idx]]
        test_data = data[perm[val_idx:test_idx]]

        train_labels = labels[perm[0:train_idx]]
        val_labels = labels[perm[train_idx:val_idx]]
        test_labels = labels[perm[val_idx:test_idx]]

        train_timestamp = timestamp[perm[0:train_idx]]
        val_timestamp = timestamp[perm[train_idx:val_idx]]
        test_timestamp = timestamp[perm[val_idx:test_idx]]

        return (train_data, train_labels, test_data, test_labels, val_data,
                val_labels, train_timestamp, val_timestamp, test_timestamp)


class Settings:
    def __init__(self, window_size, stride, n_nodes, alpha, decay, n_epochs,
                 data_split, dropout_ratio, train_location, ac_fun):
        # Window size
        self.window_size = window_size
        # Stride
        self.stride = min(stride, window_size)
        # # Number of hidden units in the LSTM layer
        self.n_nodes = n_nodes  #50
        # Learning rate
        self.alpha = alpha  #1e-4
        # Learning rate decay per update
        self.decay = decay  #0#1e-6
        # Number of Epochs (total iterations over all data)
        self.n_epochs = n_epochs  #30
        # Train-test split
        self.data_split = data_split  #0.80
        # Dropout layer ratio
        self.dropout_ratio = dropout_ratio  #0.20
        # Training file train_location
        self.train_location = train_location
        # Activation function used by LSTM.
        self.ac_fun = ac_fun

        if os.path.splitext(train_location)[-1] == ".mat":
            # Calculate length of data:
            matfile_name = sio.whosmat(train_location)[0][0]
            loaded_file = sio.loadmat(train_location)[matfile_name]
            data_name, label_name = loaded_file.dtype.names
            self.len_data = len(loaded_file[data_name][0])
        elif os.path.splitext(train_location)[-1] == ".npy":
            data = np.load(train_location)
            self.len_data = data.shape[0]
        else:
            print(
                f"Error, file extension {os.path.splitext(train_location[-1])} is unknown and cannot be loaded for data"
            )
            exit(1)

        self.__printSettings()

    def __printSettings(self):
        print("Starting a run with the following settings: ")
        print("Number of LSTM cells: \t", self.n_nodes)
        print("Number of Epochs: \t", self.n_epochs)
        print("Window size: \t\t", self.window_size)
        print("Stride:\t\t\t", self.stride)
        print("Learning rate (alpha): \t", self.alpha)
        print("Decay rate: \t\t", self.decay)
        print("Dropout ratio: \t\t", self.dropout_ratio)
        print("Activation Function:\t", self.ac_fun)
        print("Number of runs:\t\t", self.len_data)


def read_inputs():
    if len(sys.argv) == 11:
        n_nodes, n_epochs, window_size, stride = [
            int(i) for i in sys.argv[1:5]
        ]
        alpha, decay, data_split, dropout = [float(i) for i in sys.argv[5:9]]
        train_loc, ac_fun = sys.argv[9:]
        return n_nodes, n_epochs, window_size, stride, \
               alpha, decay, data_split, dropout, train_loc, ac_fun
    else:
        print("[[WARNING]] Incorrect number of arguments specified.")
        print("[[WARNING]] Using default arguments.")
        # return [
        #     100, 30, 15, 2, 0.05, 1e-9, 0.8, 0,
        #     "../old/data/deflection-reconstructed_data/Experiment_II/clearsignal_close.mat",
        #     "relu"
        # ]
        return [
            100, 30, 15, 2, 0.05, 1e-9, 0.8, 0,
            "../../../data/a1_normw1_theta0.npy", "relu"
        ]


def main():
    (n_nodes, n_epochs, window_size, stride, alpha, decay, \
     data_split, dropout_ratio, train_location, ac_fun) =  \
        read_inputs()

    # Load settings
    settings = Settings(window_size, stride, n_nodes, \
               alpha, decay, n_epochs, data_split, dropout_ratio, \
               train_location, ac_fun)
    # Load data
    data = Data(settings, train_location)

    new_model = tf.keras.models.load_model('../models/latest')
    new_model.summary()
    prev_x = 0

    for idx in range(0, 1024, 15):
        input_data = data.test_data[0][idx:idx + 15]
        input_data = np.reshape(input_data, (1, 15, 128))
        y_pred = new_model.predict(input_data)
        time0 = data.test_timestamp[0][idx + 15 - 1:idx + 15]
        time1 = data.test_timestamp[0][idx + 15:idx + 16]

        if idx != 0:

            # prev_x = data.test_labels[0][idx + 15 - 1:idx + 15][0][0]
            # y_pred[0][0] = data.test_labels[0][idx + 15:idx + 16][0][0]
            speed = (y_pred[0][0] - prev_x) / (time1[0][0] - time0[0][0])

            print(y_pred[0][0])
            print(prev_x)
            print("m", (y_pred[0][0] - prev_x), "s",
                  (time1[0][0] - time0[0][0]))
            print(speed)

        prev_x = y_pred[0][0]


if __name__ == '__main__':
    main()
