import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from get_speed_from_location import get_speed_from_data
from lib.params import Data, Settings


def read_inputs(train_loc):
    n_nodes = 100
    n_epochs = 30
    window_size = 16
    stride = 2
    alpha = 0.05
    decay = 1e-9
    shuffle_data = False
    data_split = 0.8
    dropout = 0
    # train_loc = "../data/simulation_data/combined.npy"
    # train_loc = "../data/simulation_data/a10_normw10_theta0.npy"
    # train_loc = "../data/simulation_data/a20_normw10_theta0.npy"
    # train_loc = "../data/simulation_data/a30_normw10_theta0.npy"
    # train_loc = "../data/simulation_data/a40_normw10_theta0.npy"
    # train_loc = "../data/simulation_data/a50_normw10_theta0.npy"
    # train_loc = "../data/simulation_data/a10_normw10_theta0.npy"
    # train_loc = "../data/simulation_data/a10_normw20_theta0.npy"
    # train_loc = "../data/simulation_data/a10_normw30_theta0.npy"
    # train_loc = "../data/simulation_data/a10_normw40_theta0.npy"
    # train_loc = "../data/simulation_data/a10_normw50_theta0.npy"
    ac_fun = "relu"
    return n_nodes, n_epochs, window_size, stride, \
        alpha, decay, shuffle_data, data_split, dropout, train_loc, ac_fun


def wavelet_e(p):
    return (1 - 2 * p**2) / ((1 + p**2)**(5 / 2))


def wavelet_o(p):
    return (-3 * p) / ((1 + p**2)**(5 / 2))


def wavelet_n(p):
    return (2 - p**2) / ((1 + p**2)**(5 / 2))


def inverse_volume_vx_calculation(vx, sensor, speed, x, y, theta):
    p = (sensor - x) / y
    we = wavelet_e(p)

    return abs((2 * y**3 * vx) / (speed * we * math.cos(theta)))**(1 / 3)


def inverse_volume_vy_calculation(vy, sensor, speed, x, y, theta):
    p = (sensor - x) / y
    wo = wavelet_o(p)

    # TODO: is taking the absolute absolutely fine?
    return abs((2 * y**3 * vy) / (speed * wo * math.cos(theta)))**(1 / 3)


def extract_volume(speed, vx_data, vy_data):

    # Simulation Parameters

    # TODO: Account for forward and backward runs

    sensor_range = (-200, 200)
    x_range = (-500, 500)

    number_of_sensors = 64
    number_of_x_steps = 1024
    input_sensors = list(
        np.linspace(sensor_range[0], sensor_range[1], num=number_of_sensors))
    x_input = list(np.linspace(x_range[0], x_range[1], num=number_of_x_steps))

    volumes = []
    volumes_vx = []
    volumes_vy = []
    for position in range(vx_data.shape[0]):
        plt.plot(vx_data[position, :])
        x = x_input[position]
        for sensor_idx in range(vx_data.shape[1]):
            volume_vx = inverse_volume_vx_calculation(
                vx_data[position, sensor_idx], input_sensors[sensor_idx],
                speed, x, 25, 0)
            volume_vy = inverse_volume_vy_calculation(
                vy_data[position, sensor_idx], input_sensors[sensor_idx],
                speed, x, 25, 0)
            volumes_vx.append(volume_vx)
            volumes_vy.append(volume_vy)
            volume = (volume_vx + volume_vy) / 2
            volumes.append(volume)

    return np.mean(volumes)


def start_volume_extraction():
    train_loc = f"../data/simulation_data/combined.npy"
    (n_nodes, n_epochs, window_size, stride, alpha, decay, \
     shuffle_data, data_split, dropout_ratio, train_location, ac_fun) =  \
        read_inputs(train_loc)

    # Load settings
    settings = Settings(window_size, stride, n_nodes, \
                        alpha, decay, n_epochs, shuffle_data, data_split, dropout_ratio, \
               train_location, ac_fun)
    # Load data
    data = Data(settings, train_location)

    new_model = tf.keras.models.load_model(
        '../data/trained_models/win16_stride2_epochs30_dropout0_latest')

    for run_idx in range(16):
        speeds = get_speed_from_data(data.test_data[run_idx],
                                     data.test_labels[run_idx],
                                     data.test_timestamp[run_idx], new_model)
        speed = speeds[0]

        vx_data = data.test_data[run_idx][:, ::2]
        vy_data = data.test_data[run_idx][:, 1::2]

        volume = extract_volume(speed, vx_data, vy_data)
        print(f"Real speed: {speeds[1]} mm/s")
        print(f"Speed: {speed} mm/s")
        print(f"Volume: {volume} mm")

        print('--')

    print(" -- DONE -- ")


def main():
    start_volume_extraction()


if __name__ == '__main__':
    main()
