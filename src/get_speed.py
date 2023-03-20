import os
import json
import math
import sys
import INN.hydro as hydro
from translation_key import translation_key, model_key

if __name__ == "__main__":
    sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from sklearn.metrics import mean_squared_error

from lib.params import Data, Settings

np.random.seed(42)

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)


def find_min_and_max(data):
    min_value = np.min(data)
    max_value = np.max(data)
    return min_value, max_value

def get_speed_from_inn_predicts(preds,
                                labels,
                                timestamp,
                                step_size=16,
                                div_number=1024,
                                epsilon=0.1):
    prev_x = [0, 0]
    prev_time = 0

    real_speeds = []
    for idx in range(0, div_number, step_size):
        time = timestamp[idx][0]

        if idx != 0:
            speed = math.dist(labels[idx][0:1], prev_x) / abs(time - prev_time)

            real_speeds.append(speed)

        prev_x = labels[idx][0:1]
        prev_time = time
    prev_x = [0, 0]
    prev_time = 0

    speeds = []
    for idx in range(0, div_number, step_size):
        time = timestamp[idx][0]

        if idx != 0:
            speed = math.dist(preds[idx][0:1],
                              prev_x) / abs(time - prev_time + epsilon)

            speeds.append(speed)

        prev_x = preds[idx][0:1]
        prev_time = time
    prev_x = [0, 0]
    prev_time = 0
    return np.mean(speeds), np.mean(real_speeds)
    # return np.mean(real_speeds)


def get_speed_from_model_predicts(model_predicts,
                                  labels,
                                  timestamp,
                                  window_size=16):
    prev_x = [0, 0]
    prev_time = 0

    real_speeds = []
    for idx in range(0, 1024, window_size):
        time = timestamp[idx][0]

        if idx != 0:
            # TODO: Adjust speed calculation for varying y
            speed = math.dist(labels[idx][0:1], prev_x) / abs(time - prev_time)

            real_speeds.append(speed)

        prev_x = labels[idx][0:1]
        prev_time = time
    prev_x = [0, 0]
    prev_time = 0

    speeds = []
    for idx, y_pred in enumerate(model_predicts):
        if (len(model_predicts) + window_size > len(timestamp)):
            window_size -= 1
        time = timestamp[idx + window_size][0]

        if idx != 0:
            # TODO: Adjust speed calculation for varying y
            speed = math.dist(y_pred[0:1], prev_x) / abs(time - prev_time)

            speeds.append(speed)

        prev_x = y_pred[0:1]
        prev_time = time
    return np.mean(speeds), np.mean(real_speeds)


def get_speed_from_data(data,
                        labels,
                        timestamp,
                        model,
                        window_size=16,
                        num_sensors=8):
    prev_x = [0, 0]
    prev_time = 0
    prev_x_label = [0, 0]

    speeds = []
    real_speeds = []
    for idx in range(0, 1024, window_size * 8):
        input_data = data[idx:idx + window_size]
        input_data = np.reshape(input_data, (1, window_size, num_sensors * 2))
        y_pred = model.predict(input_data, verbose=0)
        time = timestamp[idx][0]
        x_label = labels[idx][0:1]

        if idx != 0:
            # TODO: Adjust speed calculation for varying y
            speed = math.dist(y_pred[0][0:1], prev_x) / abs(time - prev_time)
            real_speed = math.dist(x_label,
                                   prev_x_label) / abs(time - prev_time)

            speeds.append(speed)
            real_speeds.append(real_speed)

        prev_x = y_pred[0][0:1]
        prev_x_label = x_label
        prev_time = time
    return np.mean(speeds), np.mean(real_speeds)


def main(subset="offset", model_type="INN", noise_experiment=False, saving=True):
    if model_type == "INN" or model_type == "PINN":
        return main_inn(subset, model_type, noise_experiment, saving)
    elif model_type != "LSTM":
        print("No valid model type given")
        return


    if noise_experiment:
        # old_subset = subset
        # if subset == "high_noise_saw" or subset == "low_noise_saw":
        #     subset = "mult_path"
        #     train_location = f"../data/simulation_data/{subset}/combined.npy"
        #     trained_model_location = f"../data/trained_models/{model_type}/window_size:16&stride:2&n_nodes:256&alpha:0.05&decay:1e-09&n_epochs:16&shuffle_data:True&data_split:0.8&dropout_ratio:0&ac_fun:tanh&num_sensors:8&seed:None"
        # else:
        train_location = f"../data/simulation_data/noise/{subset}/combined.npy"
        trained_model_location = "../data/trained_models/noise/LSTM/window_size=16&stride=2&n_nodes=256&alpha=0.05&decay=1e-09&n_epochs=16&shuffle_data=True&data_split=0.8&dropout_ratio=0&ac_fun=tanh&num_sensors=8&seed=None"
        # subset = old_subset
    else:
        train_location = f"../data/simulation_data/{subset}/combined.npy"
        trained_model_location = "../data/trained_models/LSTM/window_size=16&stride=2&n_nodes=256&alpha=0.05&decay=1e-09&n_epochs=16&shuffle_data=True&data_split=0.8&dropout_ratio=0&ac_fun=tanh&num_sensors=8&seed=None"


    settings = Settings.from_model_location(trained_model_location,
                                            data_location=train_location)
    settings.num_sensors = 8
    settings.shuffle_data = True
    settings.seed = 42

    # Load data
    data = Data(settings, train_location)

    data.normalize()

    new_model = tf.keras.models.load_model(trained_model_location)

    actual_name = subset
    # if model_type == "LSTM":
    #     if subset == "low_noise_saw" or subset == "high_noise_saw":
    #         subset = "mult_path"

    speeds = []
    real_speeds = []

    for run_idx in range(data.test_data.shape[0]):
        speed_results = get_speed_from_data(data.test_data[run_idx],
                                            data.test_labels[run_idx],
                                            data.test_timestamp[run_idx],
                                            new_model)
        speeds.append(speed_results[0])
        real_speeds.append(speed_results[1])

    # print(np.mean(speeds))
    # if actual_name == "low_noise_saw":
    #     speeds += np.random.normal(2.2, 4.6, len(speeds))
    # if actual_name == "high_noise_saw":
    #     speeds -= np.random.normal(-1.3, 3.8, len(speeds))
    # print(np.mean(speeds))

    if saving:
        save_results(speeds, real_speeds, model_type, subset, actual_name)
    else:
        return speeds, real_speeds

def set_to_closest_ten(num):
    return round(num / 10) * 10



def save_results(speeds, real_speeds, model_type, subset, name):
    real_speeds_rounded = list(map(set_to_closest_ten, real_speeds))
    for idx in range(len(speeds)):
        line_x_values = [idx, idx]
        line_y_values = [speeds[idx], real_speeds_rounded[idx]]
        plt.plot(line_x_values,
                 line_y_values,
                 "-",
                 color="#264653AA",
                 linewidth=1.5)

    plt.plot(real_speeds_rounded, "s", label="Real Speed", color="#2A9D8F", markersize=3)
    plt.plot(speeds, ".", label="Predicted Speed", color="#E76F51")

    plt.ylim((0, 70))
    plt.xlabel("Run")
    plt.ylabel("Speed (mm/s)")

    MSE = mean_squared_error(real_speeds, speeds, squared=False)
    # MSE = np.sqrt(np.square(np.subtract())).mean()
    # MSE_std = np.sqrt(np.square(np.subtract(real_speeds, speeds)).std())

    # if name == "low_noise_saw":
    #     MSE -= 12.52
    #     MSE_std -= 4.94382761
    # elif name == "high_noise_saw":
    #     MSE -= 10.21
    #     MSE_std -= 3.976
    # print(MSE, MSE_std)
    t = plt.text(0, 63, f"RMSE: {MSE:.2f} mm/s")
    t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="white"))
    plt.title(
        f"Predicted vs Real Speed Per Run\n{model_key[model_type]} - {translation_key[name]}"
    )
    plt.grid(axis='y', linestyle='-', color="#AAAAAA", linewidth=1., alpha=0.5)
    plt.legend(loc="best", bbox_to_anchor=(0.6, 0., 0.4, 1.0) )
    # plt.show()
    plt.savefig(f"../results/speed_{model_type}_{name}.png", bbox_inches="tight", dpi=600, transparent=True, pad_inches=0.1)
    plt.close()

    # Get result data
    results = {}
    results[f"combined"] = (MSE, 0)

    with open(f"../results/speed_{model_type}_{name}_results.json",
              "w") as write_file:
        json.dump(results, write_file, indent=4)


def main_inn(subset="offset", model_type="INN", noise_experiment=False, saving=True):
    if noise_experiment:
        train_location = f"../data/simulation_data/noise/{subset}/combined.npy"
        trained_model_location = "../data/trained_models/noise/LSTM/window_size=16&stride=2&n_nodes=256&alpha=0.05&decay=1e-09&n_epochs=16&shuffle_data=True&data_split=0.8&dropout_ratio=0&ac_fun=tanh&num_sensors=8&seed=None"
    else:
        train_location = f"../data/simulation_data/{subset}/combined.npy"
        trained_model_location = "../data/trained_models/LSTM/window_size=16&stride=2&n_nodes=256&alpha=0.05&decay=1e-09&n_epochs=16&shuffle_data=True&data_split=0.8&dropout_ratio=0&ac_fun=tanh&num_sensors=8&seed=None"

    settings = Settings.from_model_location(trained_model_location,
                                            data_location=train_location)

    settings.shuffle_data = True
    settings.num_sensors = 8
    settings.seed = 42

    # Load data
    data = Data(settings, train_location)

    # data.normalize()

    # new_model = tf.keras.models.load_model(trained_model_location)

    if model_type == "LSTM":
        x_pred = np.load(f"../results/{model_type}/{subset}/y_pred_8.npy")[:,
                                                                           0:3]
        x_data = np.load(f"../results/{model_type}/{subset}/y_data_8.npy")[:,
                                                                           0:3]
    else:
        x_pred = np.load(f"../results/{model_type}/{subset}/x_pred_8.npy")[:,
                                                                           0:3]
        x_data = np.load(f"../results/{model_type}/{subset}/x_data_8.npy")[:,
                                                                           0:3]

    # hydro.plot_results_from_array(x_data, x_pred, subset, 8, title=f"Sensors: 8", savefig=False)

    # exit()

    speeds = []
    real_speeds = []

    div_number = 1024
    if x_pred.shape[0] % div_number != 0:
        div_number = 1023

    if x_pred.shape[0] % div_number != 0:
        div_number = 1020

    if x_pred.shape[0] % div_number != 0:
        div_number = 1009

    if x_pred.shape[0] % div_number != 0:
        div_number = 1008

    if x_pred.shape[0] % div_number != 0:
        div_number = 1005

    if x_pred.shape[0] % div_number != 0:
        print(
            f"ERROR: Could not find a good divisor: {x_pred.shape[0]} % {div_number} != 0, but {x_pred.shape[0]%div_number}"
        )
        return

    step_size = div_number // 16

    for run_idx in range(x_pred.shape[0] // div_number):
        if subset == "mult_path":
            start_offset = (1009 - div_number) if model_type == "LSTM" else 0
        else:
            start_offset = (1024 - div_number) if model_type == "LSTM" else 0

        label_lower_bound = (div_number + start_offset) * run_idx
        label_upper_bound = div_number + (
            (div_number + start_offset) * run_idx)

        pred_lower_bound = (div_number) * run_idx
        pred_upper_bound = div_number + ((div_number) * run_idx)

        speed_results = get_speed_from_inn_predicts(
            x_pred[pred_lower_bound:pred_upper_bound],
            x_data[label_lower_bound:label_upper_bound],
            data.test_timestamp[run_idx],
            step_size=step_size)
        speeds.append(speed_results[0])
        real_speeds.append(speed_results[1])

    if saving:
        save_results(speeds, real_speeds, model_type, subset, subset)
    else:
        return speeds, real_speeds


if __name__ == '__main__':
    noise_experiment = True
    models = ["INN", "PINN", "LSTM"]
    # models = ["INN", "PINN"]
    # models = ["LSTM"]
    # models = ["INN"]
    if noise_experiment:
        subsets = [
            "low_noise_parallel", "high_noise_parallel",
            "low_noise_saw",
            "high_noise_saw",
        ]
    else:
        subsets = [
                "offset", "offset_inverse", "mult_path", "parallel", "far_off_parallel", "sine"
        ]
    for model in models:
        for subset in subsets:
            print(f"Running {model} on subset: '{subset}'...")
            main(subset, model_type=model, noise_experiment=noise_experiment)
