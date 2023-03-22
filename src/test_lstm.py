from INN.test import run_test_on_model
import numpy as np
import tensorflow as tf

from lib import params, LSTM

import INN.hydro as hydro


def test_lstm():
    noise_experiment = True

    if noise_experiment:
        sensor_options = [8]
    else:
        sensor_options = [8]


    if noise_experiment:
        subsets = [
            "low_noise_parallel", "high_noise_parallel",
            "low_noise_saw", "high_noise_saw",
        ]
    else:
        # subsets = ["offset", "offset_inverse", "mult_path", "parallel", "far_off_parallel", "sine"]
        subsets = ["sine"]

    # sensor_options = [64]
    for num_sensors in sensor_options:
        # TODO: Load model based on num_sensors
        # model_location = "window_size:16&stride:2&n_nodes:256&alpha:0.05&decay:1e-09&n_epochs:16&shuffle_data:True&data_split:0.8&dropout_ratio:0.0&ac_fun:tanh"
        model_location = f"window_size=16&stride=2&n_nodes=256&alpha=0.05&decay=1e-09&n_epochs=16&shuffle_data=True&data_split=0.8&dropout_ratio=0&ac_fun=tanh&num_sensors={num_sensors}&seed=None"
        model = tf.keras.models.load_model(
            f"../data/trained_models/LSTM/{model_location}" if not noise_experiment else f"../data/trained_models/noise/LSTM/{model_location}"
        )
        for subset in subsets:

            print(f"Running {subset}...")

            if noise_experiment:
                train_location = f"../data/simulation_data/noise/{subset}/combined.npy"
            else:
                train_location = f"../data/simulation_data/{subset}/combined.npy"

            # Load settings
            settings = params.Settings.from_model_location(model_location,
                                                        train_location)

            settings.num_sensors = num_sensors
            settings.seed = 42
            settings.shuffle_data = True

            # Load data
            data = params.Data(settings, train_location)

            data.normalize()

            # Select a subset of sensors
            print(data.test_data.shape)
            test_data = data.test_data[0:25]
            print(test_data.shape)


            network = LSTM.LSTM_network(data, settings)
            network.model = model

            network.test(test_data,
                         data.test_labels,
                         dirname=f"../results/LSTM/{subset}/",
                         num_runs=25)


            results_folder = f"../results/LSTM/{subset}/"

            hydro.plot_results_from_array(data.test_labels[0], network.pred, subset, num_sensors, savefig=True, title=f"{subset} | {num_sensors}", savepath=results_folder)

            print ("Saving the following files:")
            print(data.test_data.reshape(-1, num_sensors * 2).shape, data.test_labels.reshape(-1, 3).shape, network.pred.shape)
            np.save(results_folder+f"/x_data_{num_sensors}.npy", data.test_data.reshape(-1, num_sensors * 2))
            np.save(results_folder+f"/y_data_{num_sensors}.npy", data.test_labels.reshape(-1, 3))
            np.save(results_folder+f"/y_pred_{num_sensors}.npy", network.pred)#.reshape(-1, 3))

def main():
    test_lstm()

if __name__ == '__main__':
    main()
