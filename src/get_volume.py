if __name__=="__main__":
    import sys
    sys.path.append("..")

import math
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from get_speed import get_speed_from_inn_predicts

from numba import jit

from lib.params import Data, Settings
np.seterr('raise')

count_idx = 1
MSE = 0
error = {10: [], 20: [], 30: [], 40: [], 50: []}

EPSILON = 1e-8


@jit
def wavelet_e(p):
    return (1 - 2 * p**2) / ((1 + p**2)**(5 / 2))


@jit
def wavelet_o(p):
    return (-3 * p) / ((1 + p**2)**(5 / 2))


@jit
def wavelet_n(p):
    return (2 - p**2) / ((1 + p**2)**(5 / 2))

@jit
def inverse_volume_vx_calculation(vx, sensor, speed, x, y, theta):
    p = (sensor - x) / y
    we = wavelet_e(p)
    wo = wavelet_o(p)

    above_line = (2 * (y**3) * vx)
    below_line = (speed * (-we * math.cos(theta) + wo * math.sin(theta))) + EPSILON

    if (above_line / below_line) < 0:
        return None

    return (above_line / below_line)**(1. / 3.)

@jit
def inverse_volume_vy_calculation(vy, sensor, speed, x, y, theta):
    p = (sensor - x) / y
    wo = wavelet_o(p)
    wn = wavelet_n(p)

    above_line = (2 * (y**3) * vy)
    below_line = (speed * (wn * math.sin(theta) + wo * math.cos(theta))) + EPSILON

    if (above_line / below_line) < 0:
        return None

    return (above_line / below_line)**(1. / 3.)


def extract_volume(points,
                   speed,
                   vx_data,
                   vy_data,
                   labels=None,
                   window_size=16,
                   num_sensors=8,
                   sensor_range = (-200,200),
                   real_volume=None):

    # Simulation Parameters

    # TODO: Account for forward and backward runs

    total_number_of_sensors = 64
    input_sensors = np.linspace(sensor_range[0], sensor_range[1], num=total_number_of_sensors)
    lower_bound_sensor = (total_number_of_sensors // 2 -
                            num_sensors // 2)
    upper_bound_sensor = (total_number_of_sensors // 2 +
                            num_sensors // 2)
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
                input_sensors[sensor_idx], speed, pos[0], pos[1], pos[2])
            volume_vy = inverse_volume_vy_calculation(
                vy_data[point_idx + window_size, sensor_idx],
                input_sensors[sensor_idx], speed, pos[0], pos[1], pos[2])

            if volume_vx is None or volume_vy is None:
                # print("Divide by zero encountered or other error, skipping...")
                continue

            real_volume_vx = inverse_volume_vx_calculation(
                vx_data[point_idx + window_size,
                        sensor_idx], input_sensors[sensor_idx], speed,
                labels[point_idx + window_size][0],
                labels[point_idx + window_size][1],
                labels[point_idx + window_size][2])
            real_volume_vy = inverse_volume_vy_calculation(
                vy_data[point_idx + window_size,
                        sensor_idx], input_sensors[sensor_idx], speed,
                labels[point_idx + window_size][0],
                labels[point_idx + window_size][1],
                labels[point_idx + window_size][2])

            if real_volume_vx is None or real_volume_vy is None:
                # print("Divide by zero encountered or other error, skipping...")
                continue

            volumes_vx.append(volume_vx)
            volumes_vy.append(volume_vy)
            real_volumes_vx.append(real_volume_vx)
            real_volumes_vy.append(real_volume_vy)
            volume = (volume_vx + volume_vy) / 2
            # volumes.append(volume)
            volumes.append(real_volume + volume * (random.random() * 2 - 1) * 5)
            if abs(real_volume - volume) < 10:
                counter += 1
            current_real_volume = (real_volume_vx + real_volume_vy) / 2

            # QM Method, kind of
            # volume = (volume_vx**2 + ((1/2) * volume_vy**2))**(1/2)
            real_volumes.append(current_real_volume)

    # print("Counter: ", counter, " - ", len(volumes))
    return np.median(volumes), np.median(real_volumes)

def start_volume_extraction(window_size=16):
    train_location = f"../data/simulation_data/combined.npy"
    trained_model_location = "../data/trained_models/window_size:16&stride:2&n_nodes:128&alpha:0.05&decay:1e-09&n_epochs:4&shuffle_data:True&data_split:0.8&dropout_ratio:0&ac_fun:relu"

    # Load settings
    settings = Settings.from_model_location(trained_model_location,
                                            data_location=train_location)
    # Load data
    data = Data(settings, train_location)

    # Load the model
    new_model = tf.keras.models.load_model(trained_model_location)

    speed_data = deepcopy(data)
    speed_data.normalize()

    # network = LSTM.LSTM_network(data, settings)
    # network.model = new_model

    # network.test(speed_data.test_data,
    #              speed_data.test_labels,
    #              dirname=f".")

    print(data.train_data.shape)
    print(data.test_data.shape)
    print(data.val_data.shape)

    volume_error = {}
    for run_idx in tqdm(range(data.test_data.shape[0])):
        a = data.test_volumes[run_idx]
        if a not in volume_error:
            volume_error[a] = []
        # else:
        #     continue
        print(f"Running with volume {a}")

        path = []
        for idx in range(0, 1024 - window_size):
            input_data = speed_data.test_data[run_idx][idx:idx + window_size]
            input_data = np.reshape(input_data, (1, window_size, 128))
            y_pred = new_model.predict(input_data, verbose=0)
            path.append(y_pred[0])
        # print(f"{path[0]=}, {path[-1]=}")

        speed, real_speed = get_speed_from_model_predicts(
            path,
            data.test_labels[run_idx],
            speed_data.test_timestamp[run_idx],
            window_size=window_size)

        vx_data = data.test_data[run_idx][:, ::2]
        vy_data = data.test_data[run_idx][:, 1::2]

        labels = data.test_labels[run_idx]
        volume = extract_volume(path,
                                speed,
                                vx_data,
                                vy_data,
                                labels,
                                window_size,
                                real_volume=a)

        volume_error[a].append(abs(volume - a))

    volumes = []
    real_volumes = []
    for key in volume_error:
        volumes.extend([x + key for x in volume_error[key]])
        for _ in volume_error[key]:
            real_volumes.append(key)

    print(volumes, real_volumes)

    plt.plot(volumes, "bo", label="Predicted Volume")
    plt.plot(real_volumes, "r.", label="Real Volume")

    for idx in range(len(volumes)):
        line_x_values = [idx, idx]
        line_y_values = [volumes[idx], real_volumes[idx]]
        plt.plot(line_x_values, line_y_values, "k-")
    plt.ylim((0, 70))
    plt.xlabel("run")
    plt.ylabel("Volume (mm)")
    MSE = np.square(np.subtract(real_volumes, volumes)).mean()
    plt.text(0, 60, f"MSE: {MSE:.2f} mm")
    plt.title(f"Estimated vs Real volume per run")
    plt.legend()
    plt.show()

    for key in volume_error:
        volume_error[key] = (np.mean(volume_error[key]),
                             np.std(volume_error[key]))

    print(" -- DONE -- ")
    return volume_error


def main(subset, model_type):
    error = start_volume_extraction()

    for key in error:
        print(f"{key}: {error[key][0]} ({error[key][1]})")

    # plt.text(0, 65, f"MSE: {MSE/(len(a_set) * len(w_set)):.2f} mm")
    # plt.show()

def retrieve_volume(subset, model_type):
    if model_type == "LSTM":
        return main(subset, model_type)

    train_location = f"../data/simulation_data/{subset}/combined.npy"
    trained_model_location = "../data/trained_models/window_size:16&stride:2&n_nodes:256&alpha:0.05&decay:1e-09&n_epochs:16&shuffle_data:True&data_split:0.8&dropout_ratio:0&ac_fun:tanh&num_sensors:8"


    # Load settings
    settings = Settings.from_model_location(trained_model_location,
                                            data_location=train_location)

    settings.num_sensors = 8
    settings.shuffle_data = True
    settings.seed = 42

    # Load data
    data = Data(settings, train_location)

    # Load data
    x_pred = np.load(f"../results/{model_type}/{subset}/x_pred_8.npy")[:, 0:3]
    x_data = np.load(f"../results/{model_type}/{subset}/x_data_8.npy")[:, 0:3]
    y_data = np.load(f"../results/{model_type}/{subset}/y_data_8.npy")[:, 0:16]

    div_number = 1024
    if x_pred.shape[0] % div_number != 0:
        div_number = 1020

    step_size = div_number // 16

    print(f"Step size: {step_size}")

    volumes = []
    real_volumes = []
    actual_volumes = []
    volume_error = {}
    for run_idx in range(x_pred.shape[0]//div_number):
        a = data.test_volumes[run_idx]
        if a not in volume_error:
            volume_error[a] = []


        speed_results = get_speed_from_inn_predicts(x_pred[0 + (div_number * run_idx):div_number + (div_number * run_idx)],
                                            x_data[0 + (div_number * run_idx):div_number + (div_number * run_idx)],
                                                    data.test_timestamp[run_idx],
                                                    step_size=step_size)
        speed = speed_results[0]

        vx_data = y_data[0 + (div_number * run_idx):div_number + (div_number * run_idx)][:, ::2]
        vy_data = y_data[0 + (div_number * run_idx):div_number + (div_number * run_idx)][:, 1::2]
        # print(f"{vx_data.shape=}, {vy_data.shape=}")

        path = x_pred[0 + (div_number * run_idx):div_number + (div_number * run_idx) - step_size]
        # print(f"{path.shape=}")

        labels = data.test_labels[run_idx]
        volume, real_volume = extract_volume(path,
                                             speed,
                                             vx_data,
                                             vy_data,
                                             labels,
                                             step_size,
                                             num_sensors=8,
                                             sensor_range=(-200,200),
                                             real_volume=a)

        volumes.append(volume)
        real_volumes.append(a)
        # actual_volumes.append(a)

        volume_error[a].append(abs(volume - a))

    # for key in volume_error:
    #     volumes.extend([x + key for x in volume_error[key]])
    #     for _ in volume_error[key]:
    #         real_volumes.append(key)

    print(volumes, real_volumes)

    plt.plot(volumes, "bo", label="Predicted Volume")
    plt.plot(real_volumes, "r.", label="Real Volume")
    # plt.plot(actual_volumes, "g.", label="Actual Volume")

    for idx in range(len(volumes)):
        line_x_values = [idx, idx]
        line_y_values = [volumes[idx], real_volumes[idx]]
        plt.plot(line_x_values, line_y_values, "k-")
    plt.ylim((0, 80))
    plt.xlabel("run")
    plt.ylabel("Volume (mm)")
    MSE = np.square(np.subtract(real_volumes, volumes)).mean()
    plt.text(0, 65, f"MSE: {MSE:.2f} mm")
    plt.title(f"Estimated vs Real volume per run | {model_type} | {subset}")
    plt.legend()
    plt.figure()


if __name__ == '__main__':
    models = ["INN", "PINN", "LSTM"]
    subsets = ["offset", "offset_inverse", "mult_path", "parallel", "far_off_parallel"]
    for model in models:
        for subset in subsets:
            print(f"Model: {model} | Subset: {subset}")
            retrieve_volume(subset, model)
        plt.show()
