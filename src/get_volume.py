import math
from tqdm import tqdm
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from get_angle import get_angle_from_data
from get_speed import get_speed_from_data
from lib.params import Data, Settings

count_idx = 1
MSE = 0
error = {10: [], 20: [], 30: [], 40: [], 50: []}


def read_inputs(train_loc):
    n_nodes = 100
    n_epochs = 30
    window_size = 16
    stride = 2
    alpha = 0.05
    decay = 1e-9
    shuffle_data = True
    data_split = 0.8
    dropout = 0
    ac_fun = "relu"
    return n_nodes, n_epochs, window_size, stride, \
        alpha, decay, shuffle_data, data_split, dropout, train_loc, ac_fun


def wavelet_e(p):
    return (1 - 2 * p**2) / ((1 + p**2)**(5 / 2))


def wavelet_o(p):
    return (-3 * p) / ((1 + p**2)**(5 / 2))


def wavelet_n(p):
    return (2 - p**2) / ((1 + p**2)**(5 / 2))


# TODO: Update both inverse volume calculation for a theta that is not zero
def inverse_volume_vx_calculation(vx, sensor, speed, x, y, theta):
    p = (sensor - x) / y
    we = wavelet_e(p)
    wo = wavelet_o(p)

    return abs((2 * y**3 *
            vx) / (speed *
                   (-we * math.cos(theta) + wo * math.sin(theta))))**(1/3)


def inverse_volume_vy_calculation(vy, sensor, speed, x, y, theta):
    p = (sensor - x) / y
    wo = wavelet_o(p)
    wn = wavelet_n(p)

    # print((2 * y**3 * vy) / (speed * (wn * math.sin(theta) + wo * math.cos(theta)))**(1 / 3))
    # print((2 * y**3 * vy))
    # print((speed * (wn * math.sin(theta) + wo * math.cos(theta))))
    # print((wn * math.sin(theta) + wo * math.cos(theta)))
    # print((wo * math.cos(theta)))
    # print((wn * math.sin(theta)))

    # TODO: is taking the absolute absolutely fine?
    return abs((2 * y**3 *
            vy) / (speed *
                   (wn * math.sin(theta) + wo * math.cos(theta))))**(1/3)


def extract_volume(points, speed, theta, vx_data, vy_data, window_size):

    # Simulation Parameters

    # TODO: Account for forward and backward runs

    sensor_range = (-200, 200)

    number_of_sensors = 64
    input_sensors = list(
        np.linspace(sensor_range[0], sensor_range[1], num=number_of_sensors))

    volumes = []
    volumes_vx = []
    volumes_vy = []
    for point_idx, pos in enumerate(points):
        for sensor_idx in range(vx_data.shape[1]):
            volume_vx = inverse_volume_vx_calculation(
                vx_data[point_idx+window_size, sensor_idx], input_sensors[sensor_idx],
                speed, pos[0], pos[1], theta)
            volume_vy = inverse_volume_vy_calculation(
                vy_data[point_idx+window_size, sensor_idx], input_sensors[sensor_idx],
                speed, pos[0], pos[1], theta)
            volumes_vx.append(volume_vx)
            volumes_vy.append(volume_vy)
            volume = (volume_vx + volume_vy) / 2
            volumes.append(volume)

    return np.mean(volumes)


def start_volume_extraction(window_size=16):
    train_location = f"../data/simulation_data/combined.npy"
    trained_model_location = "../data/trained_models/window_size:16&stride:2&n_nodes:128&alpha:0.05&decay:1e-09&n_epochs:4&shuffle_data:True&data_split:0.8&dropout_ratio:0&ac_fun:relu"

    # Load settings
    settings = Settings.from_model_location(trained_model_location, data_location=train_location)
    # Load data
    data = Data(settings, train_location)
    
    # Load the model
    new_model = tf.keras.models.load_model(trained_model_location)

    speed_data = deepcopy(data)
    speed_data.normalize()

    volume_error = {}
    for run_idx in tqdm(range(data.test_data.shape[0])):
    #for run_idx in tqdm(range(2)):
        a = data.test_volumes[run_idx]
        if a not in volume_error:
            volume_error[a] = []

        path = []
        for idx in range(0, 1024-window_size):
            input_data = speed_data.test_data[run_idx][idx:idx+window_size]
            input_data = np.reshape(input_data, (1, 16, 128))
            y_pred = new_model.predict(input_data, verbose=0)
            path.append(y_pred[0])
        speed = get_speed_from_data(speed_data.test_data[run_idx],
                                     speed_data.test_labels[run_idx],
                                     speed_data.test_timestamp[run_idx],
                                     new_model)[0]
        angle = get_angle_from_data(speed_data.test_data[run_idx],
                                     speed_data.test_labels[run_idx],
                                     new_model)[0]

        vx_data = data.test_data[run_idx][:, ::2]
        vy_data = data.test_data[run_idx][:, 1::2]

        volume = extract_volume(path, speed, angle, vx_data, vy_data, window_size)
        
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
        volume_error[key] = np.mean(volume_error[key])

    print(" -- DONE -- ")
    return volume_error


def main():
    error = start_volume_extraction()

    for key in error:
        print(f"{key}: {error[key]}")

    # plt.text(0, 65, f"MSE: {MSE/(len(a_set) * len(w_set)):.2f} mm")
    # plt.show()


if __name__ == '__main__':
    main()
