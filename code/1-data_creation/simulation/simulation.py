import math
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def wavelet_e(p):
    return (1 - 2 * p**2) / ((1 + p**2)**(5 / 2))


def wavelet_o(p):
    return (-3 * p) / ((1 + p**2)**(5 / 2))


def wavelet_n(p):
    return (2 - p**2) / ((1 + p**2)**(5 / 2))


def v_x(s, x, y, theta, a, norm_w):
    p = (s - x) / y
    C = (norm_w * a**3) / (2 * y**3)
    return C * (wavelet_o(p) * math.sin(theta) -
                wavelet_e(p) * math.cos(theta))


def v_y(s, x, y, theta, a, norm_w):
    p = (s - x) / y
    C = (norm_w * a**3) / (2 * y**3)
    return C * (wavelet_n(p) * math.sin(theta) -
                wavelet_o(p) * math.cos(theta))


def simulate(theta,
             a,
             norm_w,
             sensor_range=(-200, 200),
             number_of_sensors=64,
             x_range=(-500, 500),
             y_range=(0, 500),
             number_of_x_steps=1024,
             number_of_y_steps=1,
             simulation_area_offset=25,
             number_of_runs=128,
             add_noise=True,
             noise_power=1.5e-5,
             backward_and_forward_runs=True,
             folder_path="../../../data/"):

    input_sensors = list(
        np.linspace(sensor_range[0], sensor_range[1], num=number_of_sensors))
    x_input = list(np.linspace(x_range[0], x_range[1], num=number_of_x_steps))
    y_input = list(
        np.linspace(y_range[0] + simulation_area_offset,
                    y_range[1] + simulation_area_offset,
                    num=number_of_y_steps))
    print("y_input", y_input)
    time_step = abs(x_input[1] - x_input[0]) / norm_w
    start_time = 0

    all_data = []
    all_labels = []
    all_timestamp = []
    for _ in tqdm(range(number_of_runs)):
        if backward_and_forward_runs:
            x_input = list(reversed(x_input))
        time = start_time
        data = []
        labels = []
        timestamp = []
        for y_idx, y in enumerate(y_input):
            data.append([])
            labels.append([])
            timestamp.append([])
            for x_idx, x in enumerate(x_input):
                data[y_idx].append([])
                labels[y_idx].append([])
                timestamp[y_idx].append([])
                for input_sensor in input_sensors:
                    # NOTE: the x and y coordinates are different than the array coordinates
                    data[y_idx][x_idx].append(
                        v_x(input_sensor, x, y + 1, theta, a, norm_w))
                    data[y_idx][x_idx].append(
                        v_y(input_sensor, x, y + 1, theta, a, norm_w))

                labels[y_idx][x_idx].append(x)
                labels[y_idx][x_idx].append(y + 1)
                timestamp[y_idx][x_idx].append(time)
                time += time_step

                if add_noise:
                    data[y_idx][x_idx] += np.random.normal(
                        0, noise_power, len(data[y_idx][x_idx]))
        all_data.append(data)
        all_labels.append(labels)
        all_timestamp.append(timestamp)

    data_path = folder_path + f"a{a}_normw{norm_w}_theta{theta}.npy"
    labels_path = folder_path + f"a{a}_normw{norm_w}_theta{theta}_labels.npy"
    timestamp_path = folder_path + f"a{a}_normw{norm_w}_theta{theta}_timestamp.npy"

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    all_timestamp = np.array(all_timestamp)

    all_data = np.reshape(all_data, (all_data.shape[0], all_data.shape[1] *
                                     all_data.shape[2], all_data.shape[3]))
    all_labels = np.reshape(all_labels,
                            (all_labels.shape[0], all_labels.shape[1] *
                             all_labels.shape[2], all_labels.shape[3]))
    all_timestamp = np.reshape(
        all_timestamp, (all_timestamp.shape[0], all_timestamp.shape[1] *
                        all_timestamp.shape[2], all_timestamp.shape[3]))

    # x_pos = 511
    # indices_to_plot = np.arange(x_pos % 2, len(all_data[0, x_pos]), 2)
    # to_plot = np.take(all_data, indices_to_plot, axis=2)[0, x_pos]
    # plt.plot(to_plot)
    # x_pos = 512
    # indices_to_plot = np.arange(x_pos % 2, len(all_data[0, x_pos]), 2)
    # to_plot = np.take(all_data, indices_to_plot, axis=2)[0, x_pos]
    # plt.plot(to_plot)
    # plt.show()

    print(all_data.shape)
    print(all_labels.shape)
    print(all_timestamp.shape)
    np.save(data_path, all_data)
    np.save(labels_path, all_labels)
    np.save(timestamp_path, all_timestamp)


def main():
    # w_set = [
    #     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
    # ]
    # a_set = [
    #     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
    # ]
    # w_set = [1, 2, 3, 4, 5]
    w_set = [10]
    # w_set = [10, 20, 30, 40, 50]
    # a_set = [10]
    a_set = [10, 20, 30, 40, 50]

    theta = 0
    count = 0
    print("w_set", w_set, "a_set", a_set)
    exit()
    for norm_w in w_set:
        for a in a_set:
            print(f"Simulating set {count}/{len(w_set)*len(a_set)}...")
            simulate(theta, a, norm_w)
            count += 1


if __name__ == "__main__":
    main()
