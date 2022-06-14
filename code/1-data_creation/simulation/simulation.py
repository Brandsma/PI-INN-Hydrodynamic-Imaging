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
                 sensor_range=(0, 1024),
                 x_range=(0, 1000),
                 y_range=(0, 3),
                 sampling_rate=1,
                 folder_path="../../../data/"):

    sensors = list(range(sensor_range[0], sensor_range[1] * sampling_rate))
    input_sensors = list(
        np.arange(x_range[0], x_range[1],
                  x_range[1] / (sensor_range[1] * sampling_rate)))
    data = []
    labels = []
    
    for sensor in tqdm(sensors):
        data.append([])
        labels.append([])
        for y in range(y_range[0], y_range[1]):
            data[sensor].append([])
            labels[sensor].append([])
            for x in range(x_range[0], x_range[1]):
                # input_sensor = (
                #     sensor +
                #     (x_range[1] /
                #      (sensor_range[1] * sampling_rate))) / sampling_rate
                input_sensor = input_sensors[sensor]

                # NOTE: the x and y coordinates (sensor also) are different than the array coordinates
                data[sensor][y].append(
                    (v_x(input_sensor, x + 1, y + 1, theta, a, norm_w),
                     v_y(input_sensor, x + 1, y + 1, theta, a, norm_w)))
                    
                labels[sensor][y].append((x + 1, y + 1))

    data_path = folder_path + f"a{a}_normw{norm_w}_theta{theta}.npy"
    labels_path = folder_path + f"a{a}_normw{norm_w}_theta{theta}_labels.npy"

    data = np.array(data)
    labels = np.array(labels)
    print(data.shape)
    print(labels.shape)
    np.save(data_path, data)
    np.save(labels_path, labels)


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
