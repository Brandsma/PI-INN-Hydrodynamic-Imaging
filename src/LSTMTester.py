import os
from sklearn.metrics import mean_squared_error
from util import cartesian_coord
import numpy as np
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm

from LSTM.get_speed import get_speed_from_model_predicts
from LSTM.get_volume import extract_volume

from lib import LSTM, params

import tensorflow as tf

class LSTMTester():
    def __init__(self, model_folder="../data/trained_models/LSTM/", model_name=None, result_path="../data/results/LSTM/", save_to_file=True, debug=False):
        # TODO: path maybe?
        self.result_path = result_path
        self.save_to_file = save_to_file

        if model_name is None:
            print("Error: No model name was provided to tester")
            return

        self.model_location = model_name

        self.model = tf.keras.models.load_model(
        f"{model_folder}{self.model_location}"
    )
        self.debug = debug

    def set_input_data(self, a, w, simulation_subset):
        self.train_location = f"../data/simulation_data/{simulation_subset}/a{a}_normw{w}_data.npy"

        self.a = a
        self.w = w
        self.simulation_subset = simulation_subset

        # Load settings
        self.settings = params.Settings.from_model_location(self.model_location,
                                                    self.train_location)
        # Load data
        self.data = params.Data(self.settings, self.train_location)
        self.unnormalized_data = deepcopy(self.data)
        self.data.normalize()

        self.network = LSTM.LSTM_network(self.data, self.settings)
        self.network.model = self.model

    def get_data(self):
        if not hasattr(self, 'settings') or not hasattr(self, 'data') or not hasattr(self, 'network'):
            print("Make sure to run 'set_input_data' before trying to retrieve data!")
            return

        result_data = []

        for run_idx in range(0, len(self.data.test_labels)):
            # Get location and angle
            results, associated_labels = self.__predict_data(run_idx)

            localization_error = mean_squared_error(results, associated_labels)

            # Calculate speed
            speed, _= get_speed_from_model_predicts(
                results,
                self.data.test_labels[run_idx],
                self.data.test_timestamp[run_idx], # NOTE: Taking a random timestamp, since they should all be the same
                window_size=self.settings.window_size)


            # Calculate volume
            vx_data = self.unnormalized_data.test_data[run_idx][:, ::2]
            vy_data = self.unnormalized_data.test_data[run_idx][:, 1::2]

            volume = extract_volume(results,
                                    speed,
                                    vx_data,
                                    vy_data,
                                    self.data.test_labels[run_idx],
                                    self.settings.window_size,
                                    real_volume=self.a)


            if self.debug:
                print(f" --- Data for run {run_idx} ---\n")
                print(f"{volume=}\n {speed=}\n {localization_error=}\n")

            result_data.append((results, localization_error, volume, speed))
        self.result_data = np.array(result_data, dtype=object)


    def save_result_data(self):
        if not hasattr(self, "result_data"):
            print("First run get_data and set 'result_data' before trying to save it")

        np.save(Path(f"{self.result_path}{self.simulation_subset}_a{self.a}_w{self.w}.npy"), self.result_data)
        del self.result_data


    def __predict_data(self, run_idx):
        self.settings.window_size = self.settings.window_size

        # Test all windows in the test set
        y_pred = np.zeros((0, self.data.n_outputs))
        y_true = np.zeros((0, self.data.n_outputs))
        for idx in range(0, len(self.data.test_data[run_idx]) - self.settings.window_size):
            dat = self.data.test_data[run_idx][idx:idx + self.settings.window_size]
            if np.shape(dat) != (self.settings.window_size, self.data.n_inputs):
                print("ERROR: invalid size: ", np.shape(dat))
                continue

            dat = np.reshape(dat, (1, self.settings.window_size, self.data.n_inputs))

            test_result = self.network.model.predict(dat, verbose=0)
            true_label = self.data.test_labels[run_idx][idx + self.settings.window_size - 1:idx +
                                            self.settings.window_size][0]

            y_pred = np.vstack((y_pred, test_result))
            y_true = np.vstack((y_true, true_label))

        return (y_pred, y_true)

def generate_lstm_options():
    n_nodes_options = [64, 128, 256]
    ac_fun_options = ["relu", "tanh", "sigmoid"]
    dropout_options = [0, 0.2]

    return cartesian_coord(n_nodes_options, ac_fun_options, dropout_options)
