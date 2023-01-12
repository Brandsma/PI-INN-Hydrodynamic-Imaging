import os
from typing import Tuple

import numpy as np
import scipy.io as sio
from sklearn.preprocessing import normalize

from lib.util import is_boolean, is_float, is_int
"""
Settings class.
- contains all meta-parameters such as number of
  hidden units, train-test split etc.
"""


class Settings:

    def __init__(self,
                 window_size=16,
                 stride=2,
                 n_nodes=128,
                 alpha=0.05,
                 decay=1e-9,
                 n_epochs=4,
                 shuffle_data=True,
                 data_split=0.8,
                 dropout_ratio=0,
                 train_location="../data/simulation_data/combined.npy",
                 ac_fun="relu",
                 num_sensors=64):
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
        # Whether or not to shuffle the data
        self.shuffle_data = shuffle_data
        # Train-test split
        self.data_split = data_split  #0.80
        # Dropout layer ratio
        self.dropout_ratio = dropout_ratio  #0.20
        # Training file train_location
        self.train_location = train_location
        # Activation function used by LSTM.
        self.ac_fun = ac_fun
        # Number of sensors to take from the dataset
        self.num_sensors = num_sensors

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

        # self.__printSettings()

    @classmethod
    def from_model_location(cls, folder_path, data_location=None):
        settings_folder = os.path.basename(os.path.normpath(folder_path))
        setting_elements = [x.split(':') for x in settings_folder.split('&')]
        for elem in setting_elements:
            if is_int(elem[1]):
                elem[1] = int(float(elem[1]))
            elif is_float(elem[1]):
                elem[1] = float(elem[1])
            elif is_boolean(elem[1]):
                elem[1] = bool(elem[1])

        return cls(setting_elements[0][1], setting_elements[1][1],
                   setting_elements[2][1], setting_elements[3][1],
                   setting_elements[4][1], setting_elements[5][1],
                   setting_elements[6][1], setting_elements[7][1],
                   setting_elements[8][1], data_location,
                   setting_elements[9][1])

    def __hash__(self):
        return hash(repr(self))

    @property
    def name(self):
        setting_values = self.__dict__
        if 'train_location' in setting_values:
            del setting_values['train_location']
        if 'len_data' in setting_values:
            del setting_values['len_data']
        settings_name = "&".join(
            [f"{k}:{setting_values[k]}" for k in setting_values])
        return settings_name

    def __printSettings(self):
        print("Starting a run with the following settings: ")
        print("Number of LSTM cells: \t", self.n_nodes)
        print("Number of Epochs: \t", self.n_epochs)
        print("Window size: \t\t", self.window_size)
        print("Stride:\t\t\t", self.stride)
        print("Shuffling Data:\t\t\t", self.shuffle_data)
        print("Data Split:\t\t\t", self.data_split)
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

    def __init__(self, settings, location, supplied_data: Tuple = None):
        # print("Loading dataset from " + location + "...")

        self.settings = settings
        # load data

        if supplied_data == None:
            if os.path.splitext(location)[-1] == ".npy":
                d_1, l_1, d_2, l_2, d_3, l_3, t_1, t_2, t_3, v_1, v_2, v_3 = self._load_data_numpy(
                    location, settings.data_split)
            else:
                print(
                    f"Error, file extension {os.path.splitext(location[-1])} is unknown and cannot be loaded for data"
                )
                exit(1)
        else:
            d_1, l_1, d_2, l_2, d_3, l_3, t_1, t_2, t_3, v_1, v_2, v_3 = self._split_data(
                supplied_data[0], supplied_data[1], supplied_data[2],
                supplied_data[3], settings.data_split)

        # assign to data members.
        self.train_data = d_1
        self.train_labels = l_1
        self.test_data = d_2
        self.test_labels = l_2
        self.val_data = d_3
        self.val_labels = l_3

        self.select_subset_of_data_based_on_sensors()

        self.train_timestamp = t_1
        self.val_timestamp = t_2
        self.test_timestamp = t_3

        self.train_volumes = v_1
        self.val_volumes = v_2
        self.test_volumes = v_3

        # extract dimensionality and length of data.
        self.n_inputs = np.shape(self.train_data[0])[1]
        self.n_outputs = np.shape(self.train_labels[0])[1]
        self.n_datapoints = len(self.train_data)

        # print("Dataset loaded.")

    def select_subset_of_data_based_on_sensors(self):
        if self.settings.num_sensors == 0:
            return

        lower_bound_sensor = (self.train_data.shape[2] // 2 -
                              self.settings.num_sensors)
        upper_bound_sensor = (self.train_data.shape[2] // 2 +
                              self.settings.num_sensors)

        self.train_data = self.train_data[:, :, lower_bound_sensor:
                                          upper_bound_sensor]
        self.test_data = self.test_data[:, :,
                                        lower_bound_sensor:upper_bound_sensor]
        self.val_data = self.val_data[:, :,
                                      lower_bound_sensor:upper_bound_sensor]

    @classmethod
    def from_data(cls, settings, data, labels, timestamp, volumes):
        return cls(settings,
                   "custom",
                   supplied_data=(data, labels, timestamp, volumes))

    """
    Data::__load_data_numpy(location):
    - private function
    - loads data provided by location, and returns split
      train, test and validation data and label sets.
    """

    def _load_data_numpy(self, file_location, split_ratio):
        # print("Loading numpy data")
        # Extract name of numpy struct from file and load it
        base_name = os.path.splitext(file_location)
        labels = np.load(f"{base_name[0]}_labels{base_name[-1]}")
        data = np.load(file_location)
        timestamp = np.load(f"{base_name[0]}_timestamp{base_name[-1]}")
        volumes = np.load(f"{base_name[0]}_volumes{base_name[-1]}")

        # Split data into train, test and validation sets.
        train_d, train_l, test_d, test_l, val_d, val_l, train_t, val_t, test_t, train_v, val_v, test_v = \
            self._split_data(data, labels, timestamp, volumes, split_ratio)

        return train_d, train_l, test_d, test_l, val_d, val_l, train_t, val_t, test_t, train_v, val_v, test_v

    """
    Data::__split_data(data, labels, train_test_ratio):
    - private function.
    - function which splits data into training and test set, depending on given
      train_test_ratio.
    """

    def _split_data(self, data, labels, timestamp, volumes, train_test_ratio):
        # Take the number of data points
        n_entries = len(data)

        # Generate a random permutation of indices
        if self.settings.shuffle_data:
            perm = np.random.permutation(n_entries)
        else:
            perm = np.arange(n_entries)

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

        train_volumes = volumes[perm[0:train_idx]]
        val_volumes = volumes[perm[train_idx:val_idx]]
        test_volumes = volumes[perm[val_idx:test_idx]]

        return (train_data, train_labels, test_data, test_labels, val_data,
                val_labels, train_timestamp, val_timestamp, test_timestamp,
                train_volumes, val_volumes, test_volumes)

    def normalize(self):
        for run_idx in range(self.train_data.shape[0]):
            self.train_data[run_idx, :, :] = normalize(
                self.train_data[run_idx, :, :])
        for run_idx in range(self.test_data.shape[0]):
            self.test_data[run_idx, :, :] = normalize(
                self.test_data[run_idx, :, :])
        for run_idx in range(self.val_data.shape[0]):
            self.val_data[run_idx, :, :] = normalize(
                self.val_data[run_idx, :, :])
