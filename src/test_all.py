# Use modules to keep code organized
import os
from get_speed import get_speed_from_model_predicts
from get_volume import extract_volume
from sklearn.metrics import mean_squared_error
import numpy as np
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm

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

class INNTester():
    def __init__(self, new_model, result_path="../results/", save_to_file=True):
        self.result_path = result_path
        self.save_to_file = save_to_file

        self.model_location = "window_size:16&stride:2&n_nodes:128&alpha:0.05&decay:1e-09&n_epochs:8&shuffle_data:True&data_split:0.8&dropout_ratio:0.5&ac_fun:tanh"

        self.model = new_model

    def set_input_data(self, a, w, simulation_subset):
        self.train_location = f"../data/simulation_data/{simulation_subset}/a{a}_normw{w}_data.npy"

        self.config: INNConfig = INNConfig.from_file()

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
        if not hasattr(self, 'settings') or not hasattr(self, 'data') or not hasattr(self, 'network') or not hasattr(self, 'config'):
            print("Make sure to run 'set_input_data' before trying to retrieve data!")
            return

class LSTMTester():
    def __init__(self, new_model, result_path="../results/", save_to_file=True):
        # TODO: path maybe?
        self.result_path = result_path
        self.save_to_file = save_to_file

        self.model_location = "window_size:16&stride:2&n_nodes:128&alpha:0.05&decay:1e-09&n_epochs:8&shuffle_data:True&data_split:0.8&dropout_ratio:0.5&ac_fun:tanh"

        self.model = new_model

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


        for run_idx in tqdm(range(0, len(self.data.test_labels))):
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


            print(f" --- Data for run {run_idx} ---\n")
            print(f"{volume=}\n {speed=}\n {localization_error=}\n")


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

def main():
    model = tf.keras.models.load_model(
        "../data/trained_models/window_size:16&stride:2&n_nodes:128&alpha:0.05&decay:1e-09&n_epochs:4&shuffle_data:True&data_split:0.8&dropout_ratio:0&ac_fun:relu"
    )

    lstm_tester = LSTMTester(model)


    lstm_tester.set_input_data(30, 30, "offset")

    lstm_tester.get_data()

if __name__ == "__main__":
    main()
