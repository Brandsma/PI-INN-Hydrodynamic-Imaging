import math
import os
import shutil
from pathlib import Path

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


def simulate_new(theta,
                 a,
                 norm_w,
                 sensor_range=(0, 64),
                 x_range=(-32, 32),
                 y_range=(2, 3),
                 number_of_x_steps=1024,
                 number_of_runs=120,
                 sampling_rate=1,
                 folder_path="../../../data/"):

    input_sensors = list(
        np.arange(x_range[0], x_range[1], (x_range[1] - x_range[0]) /
                  (sensor_range[1] * sampling_rate)))
    x_input = list(
        np.arange(x_range[0], x_range[1],
                  (abs(x_range[0]) + abs(x_range[1])) / number_of_x_steps))
    all_data = []
    all_labels = []
    for _ in tqdm(range(number_of_runs)):
        data = []
        labels = []
        for y in range(y_range[0], y_range[1]):
            data.append([])
            labels.append([])
            for x_idx, x in enumerate(x_input):
                data[y - y_range[0]].append([])
                labels[y - y_range[0]].append([])
                for input_sensor in input_sensors:
                    # NOTE: the x and y coordinates are different than the array coordinates
                    data[y - y_range[0]][x_idx].append(
                        v_x(input_sensor, x, y + 1, theta, a, norm_w))
                    data[y - y_range[0]][x_idx].append(
                        v_y(input_sensor, x, y + 1, theta, a, norm_w))

                labels[y - y_range[0]][x_idx].append(x)
                labels[y - y_range[0]][x_idx].append(y + 1)

        all_data.append(data)
        all_labels.append(labels)

    data_path = folder_path + f"a{a}_normw{norm_w}_theta{theta}.npy"
    labels_path = folder_path + f"a{a}_normw{norm_w}_theta{theta}_labels.npy"

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    all_data = np.reshape(all_data, (all_data.shape[0], all_data.shape[1] *
                                     all_data.shape[2], all_data.shape[3]))
    all_labels = np.reshape(all_labels,
                            (all_labels.shape[0], all_labels.shape[1] *
                             all_labels.shape[2], all_labels.shape[3]))

    print(all_data.shape)
    print(all_labels.shape)
    np.save(data_path, all_data)
    np.save(labels_path, all_labels)


def main():
    # w_set = [
    #     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
    # ]
    # a_set = [
    #     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
    # ]
    w_set = [1]
    a_set = [1]
    theta = 0
    count = 0
    for norm_w in w_set:
        for a in a_set:
            print(f"Simulating set {count}/{len(w_set)*len(a_set)}...")
            simulate_new(theta, a, norm_w)
            count += 1


if __name__ == "__main__":
    main()
