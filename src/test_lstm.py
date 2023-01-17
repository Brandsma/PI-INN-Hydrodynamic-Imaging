from INN.test import run_test_on_model
import tensorflow as tf

from lib import params, LSTM
from LSTMTester import LSTMTester

import INN.hydro as hydro


def test_lstm():

    subsets = ["offset", "offset_inverse", "mult_path", "parallel", "far_off_parallel"]
    sensor_options = [1,3,8,64]

    for num_sensors in sensor_options:
        # TODO: Load model based on num_sensors
        # model_location = "window_size:16&stride:2&n_nodes:256&alpha:0.05&decay:1e-09&n_epochs:16&shuffle_data:True&data_split:0.8&dropout_ratio:0.0&ac_fun:tanh"
        model_location = f"window_size:16&stride:2&n_nodes:256&alpha:0.05&decay:1e-09&n_epochs:16&shuffle_data:True&data_split:0.8&dropout_ratio:0&ac_fun:tanh&num_sensors:{num_sensors}"
        model = tf.keras.models.load_model(
            f"../data/trained_models/LSTM/{model_location}"
        )
        for subset in subsets:

            print(f"Running {subset}...")

            train_location = f"../data/simulation_data/{subset}/combined.npy"

            # Load settings
            settings = params.Settings.from_model_location(model_location,
                                                        train_location)

            settings.num_sensors = num_sensors

            # Load data
            data = params.Data(settings, train_location)

            data.normalize()

            # Select a subset of sensors
            test_data = data.test_data[0:32]


            network = LSTM.LSTM_network(data, settings)
            network.model = model

            network.test(test_data,
                         data.test_labels,
                         dirname=f"../results/LSTM/{subset}/",
                         num_runs=32)

            hydro.plot_results_from_array(data.test_labels[0], network.pred, subset, num_sensors, savefig=True, savepath=f"../results/LSTM/{subset}")

def main():
    test_lstm()

if __name__ == '__main__':
    main()
