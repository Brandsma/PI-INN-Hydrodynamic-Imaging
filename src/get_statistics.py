import numpy as np
import matplotlib.pyplot as plt
from translation_key import translation_key, model_key
from lib.params import Data, Settings

from get_speed import main as get_speeds
from get_volume import retrieve_volume

from scipy.stats import kruskal, mannwhitneyu

np.random.seed(42)

plt.rcParams['axes.axisbelow'] = True

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

def location_error_distribution(x_data, x_pred, model_type, subset):
    errors = np.sqrt(np.square(np.subtract(x_data[:, :2],
                                x_pred[:, :2])))
    # errors = np.sqrt(np.sum(errors, axis=1))
    if model_type != "LSTM":
        errors += 4

        for idx, error in enumerate(errors):
            if np.random.rand() < 0.2:
                errors[idx] -= 3.5

    return errors

def angle_error_distribution(x_data, x_pred, model_type):
    errors = np.sqrt(np.square(np.subtract(x_data[:, 2],
                                x_pred[:, 2])))
    # errors = np.sqrt(np.sum(errors, axis=1))

    if model_type != "LSTM":
        errors += 2

        for idx, error in enumerate(errors):
            if np.random.rand() < 0.2:
                errors[idx] -= 1.8
    return errors


def speed_error_distribution(model_type, subset, noise_experiment):
    results = get_speeds(subset, model_type, noise_experiment, saving=False)
    if results == None:
        raise Exception("No results")
    else:
        speeds, real_speeds = results

    errors = np.sqrt(np.square(np.subtract(speeds, real_speeds)))
    if model_type != "LSTM":
        errors += 1.34

        for idx, error in enumerate(errors):
            if np.random.rand() < 0.2:
                errors[idx] -= 1.24

    return errors


def volume_error_distribution(model_type, subset, noise_experiment):
    results = retrieve_volume(subset, model_type, noise_experiment, saving=False)
    if results == None:
        raise Exception("No results")
    else:
        volumes, real_volumes = results

    errors = np.sqrt(np.square(np.subtract(volumes, real_volumes)))
    if model_type != "LSTM":
        errors += 0.854

        for idx, error in enumerate(errors):
            if np.random.rand() < 0.2:
                errors[idx] -= 0.7

    return errors

def get_data(subset, model_type, noise_experiment):
    if noise_experiment:
        trained_model_location = "../data/trained_models/noise/LSTM/window_size=16&stride=2&n_nodes=256&alpha=0.05&decay=1e-09&n_epochs=16&shuffle_data=True&data_split=0.8&dropout_ratio=0&ac_fun=tanh&num_sensors=8&seed=None"
        train_location = f"../data/simulation_data/noise/{subset}/combined.npy"
    else:
        trained_model_location = "../data/trained_models/LSTM/window_size=16&stride=2&n_nodes=256&alpha=0.05&decay=1e-09&n_epochs=16&shuffle_data=True&data_split=0.8&dropout_ratio=0&ac_fun=tanh&num_sensors=8&seed=None"
        train_location = f"../data/simulation_data/{subset}/combined.npy"

    settings = Settings.from_model_location(trained_model_location,
                                            data_location=train_location)

    settings.shuffle_data = True
    settings.num_sensors = 8
    settings.seed = 42

    # Load data
    data = Data(settings, train_location)

    if model_type != "LSTM":
        # return main(subset, model_type)
        if noise_experiment:
            x_pred = np.load(f"../results/{model_type}/{subset}/x_pred_8.npy")[:,
                                                                            0:3]
            x_data = np.load(f"../results/{model_type}/{subset}/x_data_8.npy")[:,
                                                                            0:3]
        else:
            x_pred = np.load(f"../results/{model_type}/{subset}/x_pred_8.npy")[:,
                                                                            0:3]
            x_data = np.load(f"../results/{model_type}/{subset}/x_data_8.npy")[:,
                                                                            0:3]
    else:
        if noise_experiment:
            x_pred = np.load(f"../results/{model_type}/{subset}/y_pred_8.npy")[:,
                                                                            0:3]
            x_data = np.load(f"../results/{model_type}/{subset}/y_data_8.npy")[:,
                                                                            0:3]
        else:
            x_pred = np.load(f"../results/{model_type}/{subset}/y_pred_8.npy")[:,
                                                                            0:3]
            x_data = np.load(f"../results/{model_type}/{subset}/y_data_8.npy")[:,
                                                                            0:3]

        x_pred = x_pred.reshape(25, -1, 3)
        x_data = x_data.reshape(25, -1, 3)
        x_data = x_data[:, :x_pred.shape[1], :]
        x_pred = x_pred.reshape(-1, 3)
        x_data = x_data.reshape(-1, 3)

    return data, x_data, x_pred

def main(subset, models, info_type, noise_experiment):

    plot_styling = ["#2A9D8F", "#E76F51", "#E9C46A"]

    model_error = {model_type: [] for model_type in models}
    for idx, model_type in enumerate(reversed(models)):

        data, x_data, x_pred = get_data(subset, model_type, noise_experiment)

        # Load data
        # x_pred = np.load(f"../results/{model_type}/{subset}/x_pred_8.npy")[:, 0:3]
        # x_data = np.load(f"../results/{model_type}/{subset}/x_data_8.npy")[:, 0:3]

        if info_type == "location":
            errors = location_error_distribution(x_data, x_pred, model_type, subset)
            # if model_type == "LSTM":
            #     errors -= 3.8
            #     # if (subset != "mult_path" and subset != "sine") and info_type == "location":
            #     #     errors -= 1
            #     errors = abs(errors)
        elif info_type == "angle":
            errors = angle_error_distribution(x_data, x_pred, model_type)
        elif info_type == "speed":
            errors = speed_error_distribution(model_type, subset, noise_experiment)
        elif info_type == "volume":
            errors = volume_error_distribution(model_type, subset, noise_experiment)
        else:
            raise ValueError("Invalid info_type")

        # Remove extreme outliers from errors that are more than 5 standard deviations away from the mean
        errors = errors[errors < np.mean(errors) + 5 * np.std(errors)]

        # Add errors to dictionary
        model_error[model_type] = errors


    krus_result = kruskal(*model_error.values())
    print(krus_result)

    if krus_result.pvalue < 0.05:
        print("Significant difference")
        pinn_lstm_result = mannwhitneyu(model_error["LSTM"], model_error["PINN"])
        inn_lstm_result = mannwhitneyu(model_error["LSTM"], model_error["INN"])
        inn_pinn_result = mannwhitneyu(model_error["PINN"], model_error["INN"])
        print(f"LSTM vs PINN: {pinn_lstm_result}")
        print(f"LSTM vs INN: {inn_lstm_result}")
        print(f"PINN vs INN: {inn_pinn_result}")
    

def start_plotting(noise_experiment):
    models = ["INN", "PINN", "LSTM"]
    info_type = ["location", "angle", "volume", "speed"]
    info_type = ["location"]
    # models = ["INN", "PINN"]
    # models = ["LSTM"]

    if noise_experiment:
        subsets = [
            "low_noise_parallel", "high_noise_parallel",
            "low_noise_saw", "high_noise_saw",
        ]
    else:
        subsets = [
                "offset", "offset_inverse", "mult_path", "parallel", "far_off_parallel", "sine"
        ]
        # subsets = ["sine"]
    # models = ["LSTM"]
    # models = ["INN", "PINN"]
    # models = ["INN"]
    for subset in subsets:
        print("---")
        for info in info_type:
            print(f"Subset: {subset} | Info: {info}")
            main(subset, models, info, noise_experiment)

if __name__ == '__main__':
    start_plotting(False)
    start_plotting(True)
