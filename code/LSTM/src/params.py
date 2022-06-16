import os

import numpy as np
import scipy.io as sio

"""
Settings class.
- contains all meta-parameters such as number of
  hidden units, train-test split etc.
"""


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


"""
Data class.
- reads and splits data and labels into training and test sets
"""


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
        if os.path.splitext(location)[-1] == ".mat":
            d_1, l_1, d_2, l_2, d_3, l_3, = self.__load_data(
                location, settings.data_split)
        elif os.path.splitext(location)[-1] == ".npy":
            d_1, l_1, d_2, l_2, d_3, l_3, = self.__load_data_numpy(
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
        label_name = os.path.splitext(file_location)
        labels = np.load(f"{label_name[0]}_labels{label_name[-1]}")
        data = np.load(file_location)

        # Split data into train, test and validation sets.
        train_d, train_l, test_d, test_l, val_d, val_l = \
            self.__split_data(data, labels, split_ratio)

        return train_d, train_l, test_d, test_l, val_d, val_l

    """
    Data::__load_data(location):
    - private function
    - loads data provided by location, and returns split
      train, test and validation data and label sets.
    """

    def __load_data(self, file_location, split_ratio):
        # Extract name of matlab struct from file and load it
        matfile_name = sio.whosmat(file_location)[0][0]
        loaded_file = sio.loadmat(file_location)[matfile_name]

        # Split into labels and data
        # if first name is not label, switch them
        if loaded_file.dtype.names[0].find('lab') == -1:
            data_name, label_name = loaded_file.dtype.names
        else:
            label_name, data_name = loaded_file.dtype.names
        all_data = loaded_file[data_name][0]
        all_labels = loaded_file[label_name][0]

        print(all_data.shape)
        print(all_data[0].shape)
        print(all_data[0][0].shape)

        # Split data into train, test and validation sets.
        train_d, train_l, test_d, test_l, val_d, val_l = \
            self.__split_data(all_data, all_labels, split_ratio)

        return train_d, train_l, test_d, test_l, val_d, val_l

    """
    Data::__split_data(data, labels, train_test_ratio):
    - private function.
    - function which splits data into training and test set, depending on given
      train_test_ratio.
    """

    def __split_data(self, data, labels, train_test_ratio):
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

        return (train_data, train_labels, test_data, test_labels, val_data,
                val_labels)
