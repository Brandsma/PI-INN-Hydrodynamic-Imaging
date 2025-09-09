from lib import params
from pathlib import Path
from util import cartesian_coord
import sys
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error
from INN.inn import INNConfig, create_model
from INN.hydro import setup_data_with_data
import INN.hydro as hydro

from get_speed import get_speed_from_model_predicts
from get_volume import extract_volume

from copy import deepcopy


class INNTester:

    def __init__(
        self,
        model_folder="../data/trained_models/INN/latest/",
        result_path="../data/results/INN/",
        save_to_file=True,
        debug=False,
    ):
        self.result_path = result_path
        self.save_to_file = save_to_file

        self.model_folder = model_folder
        self.debug = debug

    def set_input_data(self, a, w, simulation_subset):
        ## Get Config
        if a == 0 and w == 0:
            self.train_location = (
                f"../data/simulation_data/{simulation_subset}/combined_data.npy"
            )
        else:
            self.train_location = (
                f"../data/simulation_data/{simulation_subset}/a{a}_normw{w}_data.npy"
            )

        sys.path.append("./INN")
        self.config: INNConfig = INNConfig.from_file(
            f"{self.model_folder}/INNConfig.pkl"
        )

        self.x_dim = self.config.x_dim
        self.y_dim = self.config.y_dim
        self.z_dim = self.config.z_dim
        self.tot_dim = self.y_dim + self.config.z_dim

        self.a = a
        self.w = w
        self.simulation_subset = simulation_subset
        ###

        # Load settings
        self.settings = params.Settings(
            shuffle_data=False, train_location=self.train_location
        )

        # Load data
        self.unnormalized_data = params.Data(self.settings, self.train_location)
        self.train_data, self.train_labels, self.test_data, self.test_labels = (
            setup_data_with_data(deepcopy(self.unnormalized_data))
        )

        # Load model
        model = create_model(
            self.tot_dim,
            self.config.n_couple_layer,
            self.config.n_hid_layer,
            self.config.n_hid_dim,
        )
        latest_model_path = f"{self.model_folder}trained_model_weights.tf"
        model.load_weights(latest_model_path)

        self.network = model

    def get_data(self):
        if (
            not hasattr(self, "settings")
            or not hasattr(self, "test_labels")
            or not hasattr(self, "network")
            or not hasattr(self, "config")
        ):
            print("Make sure to run 'set_input_data' before trying to retrieve data!")
            return

        result_data = []

        for run_idx in range(0, len(self.test_labels) // 1024):
            x_data, x_pred, y_data, y_pred = self.__predict_data(run_idx)

            # print(x_pred[:, :self.x_dim].shape)
            # exit()

            # hydro.plot_results(x_data, x_pred, y_data, y_pred, self.x_dim, self.y_dim, title=f"Hydro", savefig=False)

            # Calculate speed
            speed, _ = get_speed_from_model_predicts(
                x_pred[:, : self.x_dim],
                self.test_labels[(1024 * run_idx) : (1024 + 1024 * run_idx)],
                self.unnormalized_data.test_timestamp[run_idx],
                window_size=1,
            )

            # Calculate volume
            vx_data = self.unnormalized_data.test_data[run_idx][:, ::2]
            vy_data = self.unnormalized_data.test_data[run_idx][:, 1::2]

            volume = extract_volume(
                x_pred[:, : self.x_dim],
                speed,
                vx_data,
                vy_data,
                self.test_labels[(1024 * run_idx) : (1024 + 1024 * run_idx)],
                0,
                real_volume=self.a,
            )

            if self.debug:
                print(
                    f" -- INN error -- \n\nForward error: {mean_squared_error(y_data[:, :self.y_dim], y_pred[:, :self.y_dim])}\nBackward error:{mean_squared_error(x_data[:, :self.x_dim], x_pred[:, :self.x_dim])}"
                )
                print(f"\n\nVolume: {volume}\nSpeed: {speed}\n")

            result_data.append(
                (
                    x_pred,
                    y_pred,
                    mean_squared_error(
                        y_data[:, : self.y_dim], y_pred[:, : self.y_dim]
                    ),
                    mean_squared_error(
                        x_data[:, : self.x_dim], x_pred[:, : self.x_dim]
                    ),
                    volume,
                    speed,
                )
            )
        self.result_data = np.array(result_data, dtype=object)
        # print(self.result_data[:, 2])

    def save_result_data(self):
        if not hasattr(self, "result_data"):
            print("First run get_data and set 'result_data' before trying to save it")

        np.save(
            Path(f"{self.result_path}{self.simulation_subset}_a{self.a}_w{self.w}.npy"),
            self.result_data,
        )
        del self.result_data

    def __predict_data(self, run_idx):
        data = self.test_data[(1024 * run_idx) : (1024 + 1024 * run_idx)]
        labels = self.test_labels[(1024 * run_idx) : (1024 + 1024 * run_idx)]

        z_dim = self.z_dim
        tot_dim = self.y_dim + z_dim
        pad_dim = tot_dim - self.x_dim

        # Test model
        pad_x = np.zeros((data.shape[0], pad_dim))
        pad_x = np.random.multivariate_normal(
            [0.0] * pad_dim, np.eye(pad_dim), data.shape[0]
        )

        z = np.random.multivariate_normal([1.0] * z_dim, np.eye(z_dim), labels.shape[0])
        y_data = np.concatenate([labels, z], axis=-1).astype("float32")
        x_pred = self.network.inverse(y_data).numpy()
        x_data = np.concatenate([data, pad_x], axis=-1).astype("float32")
        y_pred = self.network(x_data).numpy()

        return x_data, x_pred, y_data, y_pred


def generate_inn_options():
    n_couple_layer_options = [2, 4, 8]
    n_hid_layers_options = [2, 4, 8]
    n_hid_dim_options = [16, 32, 64, 128, 256]
    z_dim_options = [2, 16, 32, 64]

    return cartesian_coord(
        n_couple_layer_options, n_hid_layers_options, n_hid_dim_options, z_dim_options
    )
