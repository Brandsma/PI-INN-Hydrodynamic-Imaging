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


def save_simulation_data(vx_data, vy_data, path):
    if Path(path).exists():
        shutil.rmtree(Path(path))
        os.makedirs(Path(path))
    else:
        os.makedirs(Path(path))

    vx_data.to_csv(Path(path + "simdata_vx.csv"), index=False)
    vy_data.to_csv(Path(path + "simdata_vy.csv"), index=False)


def simulate(theta,
             a,
             norm_w,
             sensor_range=(1, 33),
             x_range=(1, 21),
             y_range=(1, 11),
             sampling_rate=1,
             folder_path="../../data/"):
    sensors = list(range(sensor_range[0],
                         sensor_range[1]))  # equidistantly spaced

    vx_data = pd.DataFrame(columns=sensors,
                           index=np.arange(
                               (y_range[1] - 1) * (x_range[1] - 1)),
                           dtype=np.float32)
    vy_data = pd.DataFrame(columns=sensors,
                           index=np.arange(
                               (y_range[1] - 1) * (x_range[1] - 1)),
                           dtype=np.float32)

    for sensor in tqdm(sensors):
        for y in range(y_range[0], y_range[1]):
            for x in range(x_range[0], x_range[1]):
                input_sensor = sensor / sampling_rate

                vx_data[sensor][(y - 1) * (x_range[1] - 1) + (x - 1)] = v_x(
                    input_sensor, x, y, theta, a, norm_w)

                vy_data[sensor][(y - 1) * (x_range[1] - 1) + (x - 1)] = v_y(
                    input_sensor, x, y, theta, a, norm_w)

    # TODO: Ideally we don't have to do this, it changes the indexed position!!
    vx_data = vx_data.dropna()
    vy_data = vy_data.dropna()

    data_path = folder_path + f"a{a}_normw{norm_w}_theta{theta}/"

    save_simulation_data(vx_data, vy_data, data_path)


def simulate_new(theta,
                 a,
                 norm_w,
                 sensor_range=(0, 64),
                 x_range=(0, 100),
                 y_range=(0, 10),
                 sampling_rate=10,
                 folder_path="../../data/"):

    sensors = list(range(sensor_range[0], sensor_range[1] * sampling_rate))
    input_sensors = list(
        np.arange(x_range[0], x_range[1],
                  x_range[1] / (sensor_range[1] * sampling_rate)))
    data = []

    for sensor in tqdm(sensors):
        data.append([])
        for y in range(y_range[0], y_range[1]):
            data[sensor].append([])
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

    data_path = folder_path + f"a{a}_normw{norm_w}_theta{theta}.npy"

    data = np.array(data)
    print(data.shape)
    np.save(data_path, data)


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
