from lib import params
import sys
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error
from INN.inn import INNConfig, create_model

from copy import deepcopy

class INNTester():
    def __init__(self, model_folder="../data/trained_models/INN/latest/", result_path="../data/results/", save_to_file=True):
        self.result_path = result_path
        self.save_to_file = save_to_file

        self.model_folder = model_folder

    def set_input_data(self, a, w, simulation_subset):
        ## Get Config
        self.train_location = f"../data/simulation_data/{simulation_subset}/a{a}_normw{w}_data.npy"

        sys.path.append("./INN")
        self.config: INNConfig = INNConfig.from_file("../data/trained_models/INN/latest/INNConfig.pkl")

        self.x_dim = self.config.x_dim
        self.y_dim = self.config.y_dim
        self.z_dim = self.config.z_dim
        self.tot_dim = self.y_dim + self.config.z_dim

        self.a = a
        self.w = w
        self.simulation_subset = simulation_subset
        ###


        # Load settings
        self.settings = params.Settings(shuffle_data=False, train_location=self.train_location)

        # Load data
        self.data = params.Data(self.settings, self.train_location)
        self.unnormalized_data = deepcopy(self.data)
        self.data.normalize()

        # Load model
        model = create_model(self.tot_dim, self.config.n_couple_layer, self.config.n_hid_layer, self.config.n_hid_dim)
        latest_model_path = f"{self.model_folder}trained_model_weights.tf"
        model.load_weights(latest_model_path)

        self.network = model

        # self.network = LSTM.LSTM_network(self.data, self.settings)
        # self.network.model = self.model

    def get_data(self):
        if not hasattr(self, 'settings') or not hasattr(self, 'data') or not hasattr(self, 'network') or not hasattr(self, 'config'):
            print("Make sure to run 'set_input_data' before trying to retrieve data!")
            return

        for run_idx in tqdm(range(0, len(self.data.test_labels))):
            x_data, x_pred, y_data, y_pred = self.__predict_data(run_idx)

            print(f" -- INN error -- \n\nForward error: {mean_squared_error(y_data[:, :self.y_dim], y_pred[:, :self.y_dim])}\nBackward error:{mean_squared_error(x_data[:, :self.x_dim], x_pred[:, :self.x_dim])}")



    def __predict_data(self, run_idx):

        data = self.data.test_data[run_idx]
        labels = self.data.test_labels[run_idx]

        print(data.shape)
        print(labels.shape)
        exit()

        z_dim = self.z_dim
        tot_dim = self.y_dim + z_dim
        pad_dim = tot_dim - self.x_dim

        # Test model
        pad_x = np.zeros((data.shape[0], pad_dim))
        pad_x = np.random.multivariate_normal([0.] * pad_dim, np.eye(pad_dim),
                                            data.shape[0])

        z = np.random.multivariate_normal([1.] * z_dim, np.eye(z_dim), labels.shape[0])
        y_data = np.concatenate([labels, z], axis=-1).astype('float32')
        x_pred = self.network.inverse(y_data).numpy()
        x_data = np.concatenate([data, pad_x], axis=-1).astype('float32')
        y_pred = self.network(x_data).numpy()

        return x_data, x_pred, y_data, y_pred
