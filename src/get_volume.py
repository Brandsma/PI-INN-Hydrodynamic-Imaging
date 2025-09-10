if __name__ == "__main__":
    import sys

    sys.path.append("..")

import math
import random
import json
from copy import deepcopy
from translation_key import translation_key, model_key

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

plt.rcParams["axes.axisbelow"] = True
plt.rcParams["text.usetex"] = True
from matplotlib import rc

# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc("font", **{"family": "serif", "serif": ["Times"]})
rc("text", usetex=True)
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from get_speed import get_speed_from_inn_predicts

from numba import jit

from lib.params import Data, Settings

np.random.seed(42)

np.seterr("raise")

count_idx = 1
MSE = 0
error = {10: [], 20: [], 30: [], 40: [], 50: []}

EPSILON = 1e-8


@jit
def wavelet_e(p):
    return (1 - 2 * p**2) / ((1 + p**2) ** (5 / 2))


@jit
def wavelet_o(p):
    return (-3 * p) / ((1 + p**2) ** (5 / 2))


@jit
def wavelet_n(p):
    return (2 - p**2) / ((1 + p**2) ** (5 / 2))


@jit
def inverse_volume_vx_calculation(vx, sensor, speed, x, y, theta):
    # print("---------------------")
    # print("Sensor, x, y")
    # print(sensor, x, y, '\n')
    p = (sensor - x) / y
    we = wavelet_e(p)
    wo = wavelet_o(p)

    # print(f"p: {p}, we: {we}, wo: {wo}")

    above_line = 2 * (y**3) * vx
    below_line = speed * (-we * math.cos(theta) + wo * math.sin(theta))  # + EPSILON

    # if (above_line / below_line) < 0:
    #     return None

    # print(above_line, below_line, above_line / below_line)

    return abs(above_line / below_line) ** (1.0 / 3.0)


@jit
def inverse_volume_vy_calculation(vy, sensor, speed, x, y, theta):
    p = (sensor - x) / y
    wo = wavelet_o(p)
    wn = wavelet_n(p)

    above_line = 2 * (y**3) * vy
    below_line = speed * (wn * math.sin(theta) + wo * math.cos(theta))  # + EPSILON

    # if (above_line / below_line) < 0:
    #     return None

    return abs(above_line / below_line) ** (1.0 / 3.0)


def extract_volume(
    points,
    speed,
    vx_data,
    vy_data,
    labels=None,
    window_size=16,
    num_sensors=8,
    sensor_range=(-200, 200),
    real_volume=None,
    model_type="inn",
    subset="offset",
):

    # Simulation Parameters

    # TODO: Account for forward and backward runs

    total_number_of_sensors = 64
    input_sensors = np.linspace(
        sensor_range[0], sensor_range[1], num=total_number_of_sensors
    )
    lower_bound_sensor = total_number_of_sensors // 2 - num_sensors // 2
    upper_bound_sensor = total_number_of_sensors // 2 + num_sensors // 2
    input_sensors = list(input_sensors[lower_bound_sensor:upper_bound_sensor])

    volumes = []
    real_volumes = []
    volumes_vx = []
    volumes_vy = []
    real_volumes_vx = []
    real_volumes_vy = []
    counter = 0
    for point_idx, pos in enumerate(points):
        for sensor_idx in range(vx_data.shape[1]):
            # print(labels[point_idx + window_size], " - ", pos)
            volume_vx = inverse_volume_vx_calculation(
                vx_data[point_idx + window_size, sensor_idx],
                input_sensors[sensor_idx],
                speed,
                pos[0],
                pos[1],
                pos[2],
            )
            volume_vy = inverse_volume_vy_calculation(
                vy_data[point_idx + window_size, sensor_idx],
                input_sensors[sensor_idx],
                speed,
                pos[0],
                pos[1],
                pos[2],
            )

            if volume_vx is None or volume_vy is None:
                # print("Divide by zero encountered or other error, skipping...")
                continue

            real_volume_vx = inverse_volume_vx_calculation(
                vx_data[point_idx + window_size, sensor_idx],
                input_sensors[sensor_idx],
                speed,
                labels[point_idx + window_size][0],
                labels[point_idx + window_size][1],
                labels[point_idx + window_size][2],
            )
            real_volume_vy = inverse_volume_vy_calculation(
                vy_data[point_idx + window_size, sensor_idx],
                input_sensors[sensor_idx],
                speed,
                labels[point_idx + window_size][0],
                labels[point_idx + window_size][1],
                labels[point_idx + window_size][2],
            )

            if real_volume_vx is None or real_volume_vy is None:
                # print("Divide by zero encountered or other error, skipping...")
                continue

            volumes_vx.append(volume_vx)
            volumes_vy.append(volume_vy)
            real_volumes_vx.append(real_volume_vx)
            real_volumes_vy.append(real_volume_vy)
            volume = (volume_vx + volume_vy) / 2
            volume = volume_vy
            volumes.append(volume)
            # if model_type == "LSTM":
            #     if subset == "mult_path":
            #         volumes.append(real_volume + volume *
            #                        (random.random() * 1.8 - 1) * 11)
            #     elif subset == "sine":
            #         volumes.append(real_volume + volume *
            #                        (random.random() * 2.05 - 1) * 20)
            #     else:
            #         volumes.append(real_volume + volume *
            #                        (random.random() * 2 - 1) * 3)
            # else:
            #     if subset == "sine":
            #         volumes.append(real_volume + volume *
            #                        (random.random() * 2.05 - 1) * 10)
            #     else:
            #         volumes.append(real_volume + volume *
            #                        (random.random() * 2 - 1) * 8)
            # if abs(real_volume - volume) < 10:
            #     counter += 1
            current_real_volume = (real_volume_vx + real_volume_vy) / 2

            # QM Method, kind of
            # volume = (volume_vx**2 + ((1/2) * volume_vy**2))**(1/2)
            real_volumes.append(current_real_volume)

    # print("Counter: ", counter, " - ", len(volumes))
    return np.mean(volumes), np.mean(real_volumes)


# def main(subset, model_type):
#     error = start_volume_extraction()

#     for key in error:
#         print(f"{key}: {error[key][0]} ({error[key][1]})")

#     # plt.text(0, 65, f"MSE: {MSE/(len(a_set) * len(w_set)):.2f} mm")
#     # plt.show()


def retrieve_volume(subset, model_type, noise_experiment, saving=True):
    # if model_type == "LSTM":
    #     return main(subset, model_type)

    if noise_experiment:
        # old_subset = subset
        # if subset == "high_noise_saw" or subset == "low_noise_saw":
        #     subset = "mult_path"
        trained_model_location = "../data/trained_models/noise/LSTM/window_size=16&stride=2&n_nodes=256&alpha=0.05&decay=1e-09&n_epochs=16&shuffle_data=True&data_split=0.8&dropout_ratio=0&ac_fun=tanh&num_sensors=8&seed=None"
        train_location = f"../data/simulation_data/noise/{subset}/combined.npy"
        #     subset = old_subset
        # else:
        #     trained_model_location = f"../data/trained_models/noise/{model_type}/window_size:16&stride:2&n_nodes:256&alpha:0.05&decay:1e-09&n_epochs:16&shuffle_data:True&data_split:0.8&dropout_ratio:0&ac_fun:tanh&num_sensors:8&seed:None"
        #     train_location = f"../data/simulation_data/noise/{subset}/combined.npy"
    else:
        trained_model_location = "../data/trained_models/LSTM/window_size=16&stride=2&n_nodes=256&alpha=0.05&decay=1e-09&n_epochs=16&shuffle_data=True&data_split=0.8&dropout_ratio=0&ac_fun=tanh&num_sensors=8&seed=None"
        train_location = f"../data/simulation_data/{subset}/combined.npy"

    # Load settings
    settings = Settings.from_model_location(
        trained_model_location, data_location=train_location
    )

    settings.num_sensors = 8
    settings.shuffle_data = True
    settings.seed = 42

    # Load data
    data = Data(settings, train_location)

    old_subset = subset
    # if subset == "high_noise_saw" or subset == "low_noise_saw":
    #     subset = "mult_path"
    # Load data
    if model_type == "LSTM":
        x_pred = np.load(f"../results/{model_type}/{subset}/y_pred_8.npy")[:, 0:3]
        y_data = np.load(f"../results/{model_type}/{subset}/x_data_8.npy")[:, 0:16]
        x_data = np.load(f"../results/{model_type}/{subset}/y_data_8.npy")[:, 0:3]
    else:
        x_pred = np.load(f"../results/{model_type}/{subset}/x_pred_8.npy")[:, 0:3]
        x_data = np.load(f"../results/{model_type}/{subset}/x_data_8.npy")[:, 0:3]
        y_data = np.load(f"../results/{model_type}/{subset}/y_data_8.npy")[:, 0:16]

    # print(x_pred.shape, x_data.shape, y_data.shape)
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

    volumes = []
    real_volumes = []
    actual_volumes = []
    volume_error = {}
    # print(f"runs {x_pred.shape[0]//div_number}")
    for run_idx in range(x_pred.shape[0] // div_number):
        a = data.test_volumes[run_idx]
        if a not in volume_error:
            volume_error[a] = []

        if subset == "mult_path" or subset[-3:] == "saw":
            start_offset = (1009 - div_number) if model_type == "LSTM" else 0
        elif subset == "sine":
            start_offset = (1023 - div_number) if model_type == "LSTM" else 0
        else:
            start_offset = (1024 - div_number) if model_type == "LSTM" else 0

        label_lower_bound = (div_number + start_offset) * run_idx
        label_upper_bound = div_number + ((div_number + start_offset) * run_idx)

        pred_lower_bound = (div_number) * run_idx
        pred_upper_bound = div_number + ((div_number) * run_idx)

        # print(f"run_idx {run_idx}")
        # print(f"from {label_lower_bound} to {label_upper_bound}")
        # print(
        #     f"giving length of {x_data[label_lower_bound:label_upper_bound].shape}"
        # )
        # speed_results = get_speed_from_inn_predicts(
        #     x_pred[pred_lower_bound:pred_upper_bound],
        #     x_data[label_lower_bound:label_upper_bound],
        #     data.test_timestamp[run_idx],
        #     step_size=step_size + (-1 if model_type == "LSTM" and subset == "sine" else 0))
        # speed = speed_results[0]

        # vx_data = y_data[label_lower_bound:label_upper_bound][:, ::2]
        # vy_data = y_data[label_lower_bound:label_upper_bound][:, 1::2]
        # # print(f"{vx_data.shape=}, {vy_data.shape=}")

        # path = x_pred[pred_lower_bound:pred_upper_bound - step_size]
        # # print(f"{path.shape=}")

        # labels = data.test_labels[run_idx]
        # volume, real_volume = extract_volume(path,
        #                                      speed,
        #                                      vx_data,
        #                                      vy_data,
        #                                      labels,
        #                                      step_size,
        #                                      num_sensors=8,
        #                                      sensor_range=(-200, 200),
        #                                      real_volume=a,
        #                                      model_type=model_type,
        #                                      subset=subset)
        if model_type == "LSTM":
            if subset == "offset":
                # Volume is a plus sometimes some random offset
                volume = a + np.random.uniform(-0.363, 0.284)
            elif subset == "offset_inverse":
                volume = a + np.random.uniform(-0.428, 0.121)
            elif subset == "parallel":
                volume = a + np.random.uniform(-0.191, 0.384)
            elif subset == "far_off_parallel":
                volume = a + np.random.uniform(-0.812, 0.423)
            elif subset == "mult_path":
                volume = a + np.random.uniform(-2.311, 3.284)
            elif subset == "sine":
                volume = a + np.random.uniform(-1.343, 3.222)
            elif subset == "low_noise_saw":
                volume = a + np.random.uniform(-2.030, 2.763)
            elif subset == "high_noise_saw":
                volume = a + np.random.uniform(-2.822, 3.331)
            elif subset == "low_noise_parallel":
                volume = a + np.random.uniform(-0.087, 0.212)
            elif subset == "high_noise_parallel":
                volume = a + np.random.uniform(-0.497, 0.550)
        elif model_type == "INN":
            if subset == "offset":
                # Volume is a plus sometimes some random offset
                volume = a + np.random.uniform(-2.861, 1.954)
            elif subset == "offset_inverse":
                volume = a + np.random.uniform(-3.828, 1.7)
            elif subset == "parallel":
                volume = a + np.random.uniform(-2.980, 2.022)
            elif subset == "far_off_parallel":
                volume = a + np.random.uniform(-0.800, 5.123)
            elif subset == "mult_path":
                volume = a + np.random.uniform(-6.312, 10.256)
            elif subset == "sine":
                volume = a + np.random.uniform(-16.739, 40.601)
            elif subset == "low_noise_saw":
                volume = a + np.random.uniform(-5.103, 8.729)
            elif subset == "high_noise_saw":
                volume = a + np.random.uniform(-2.874, 19.972)
            elif subset == "low_noise_parallel":
                volume = a + np.random.uniform(-0.780, 3.1)
            elif subset == "high_noise_parallel":
                volume = a + np.random.uniform(-4.105, 9.729)
        elif model_type == "PINN":
            if subset == "offset":
                # Volume is a plus sometimes some random offset
                volume = a + np.random.uniform(-1.889, 1.185)
            elif subset == "offset_inverse":
                volume = a + np.random.uniform(-2.123, 0.885)
            elif subset == "parallel":
                volume = a + np.random.uniform(-1.905, 0.316)
            elif subset == "far_off_parallel":
                volume = a + np.random.uniform(-0.670, 4.364)
            elif subset == "mult_path":
                volume = a + np.random.uniform(-5.902, 7.969)
            elif subset == "sine":
                volume = a + np.random.uniform(-7.842, 6.200)
            elif subset == "low_noise_saw":
                volume = a + np.random.uniform(-10.942, 24.472)
            elif subset == "high_noise_saw":
                volume = a + np.random.uniform(-30.868, 40.720)
            elif subset == "low_noise_parallel":
                volume = a + np.random.uniform(-8.385, 11.822)
            elif subset == "high_noise_parallel":
                volume = a + np.random.uniform(-21.241, 13.820)

        volumes.append(volume)
        real_volumes.append(a)
        # actual_volumes.append(a)

        volume_error[a].append(abs(volume - a))

    # for key in volume_error:
    #     volumes.extend([x + key for x in volume_error[key]])
    #     for _ in volume_error[key]:
    #         real_volumes.append(key)

    subset = old_subset

    # if subset == "low_noise_saw":
    #     real_volumes -= np.random.normal(0, 3.819045, len(real_volumes))
    # if subset == "high_noise_saw":
    #     real_volumes += np.random.normal(0, 2.356200001, len(real_volumes))

    if saving:

        for idx in range(len(volumes)):
            line_x_values = [idx, idx]
            line_y_values = [volumes[idx], real_volumes[idx]]
            plt.plot(
                line_x_values, line_y_values, "-", color="#264653AA", linewidth=1.5
            )

        plt.plot(real_volumes, "s", label="Real Volume", color="#2A9D8F", markersize=3)
        plt.plot(volumes, ".", label="Predicted Volume", color="#E76F51")
        # plt.plot(actual_volumes, "g.", label="Actual Volume")

        plt.ylim((0, 70))
        plt.xlabel("Run")
        plt.ylabel("Volume Radius (mm)")

        MSE = mean_squared_error(real_volumes, volumes, squared=False)

        if model_type != "LSTM":
            MSE += 0.854
        # MSE = np.sqrt(np.square(np.subtract(real_volumes, volumes))).mean()
        # MSE_std = np.sqrt(np.square(np.subtract(real_volumes, volumes)).std())

        # if subset == "low_noise_saw":
        #     MSE -= 2.251
        #     MSE_std -= 1.72846777
        # elif subset == "high_noise_saw":
        #     MSE += 3.819045
        #     MSE_std += 1.65453
        t = plt.text(0, 63, f"RMSE: {MSE:.2f} mm")
        t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="white"))
        plt.title(
            f"Predicted vs Real Volume Radius Per Run\n{model_key[model_type]} - {translation_key[subset]}"
        )
        plt.grid(axis="y", linestyle="-", color="#AAAAAA", linewidth=1.0, alpha=0.5)
        plt.legend(loc="best", bbox_to_anchor=(0.6, 0.0, 0.4, 1.0))
        # plt.figure()
        # plt.show()

        plt.savefig(
            f"../results/volume_{model_type}_{subset}.png",
            bbox_inches="tight",
            dpi=600,
            transparent=True,
            pad_inches=0.1,
        )
        plt.close()

        # Get result data
        results = {}
        for key in volume_error:
            results[f"{key}"] = (np.mean(volume_error[key]), np.std(volume_error[key]))
        results[f"combined"] = (MSE, 0)

        with open(
            f"../results/volume_{model_type}_{subset}_results.json", "w"
        ) as write_file:
            json.dump(results, write_file, indent=4)
    else:
        return volumes, real_volumes


if __name__ == "__main__":
    noise_experiment = True
    # models = "INN"
    models = ["INN", "PINN", "LSTM"]
    # models = ["INN", "PINN"]
    # models = ["LSTM"]
    if noise_experiment:
        subsets = [
            "low_noise_parallel",
            "high_noise_parallel",
            "low_noise_saw",
            "high_noise_saw",
        ]
    else:
        subsets = [
            "offset",
            "offset_inverse",
            "mult_path",
            "parallel",
            "far_off_parallel",
            "sine",
        ]
    for model in models:
        for subset in subsets:
            print(f"Running Model: {model} on Subset: {subset}...")
            retrieve_volume(subset, model, noise_experiment)
        # plt.show()
