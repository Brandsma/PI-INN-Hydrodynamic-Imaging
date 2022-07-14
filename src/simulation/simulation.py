import math

import numpy as np
from lib.logger import setup_logger
from tqdm import tqdm

log = setup_logger(__name__)

## Velocity Profiles ##


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


## Simulation ##


def simulate(theta=0,
             a=10,
             norm_w=10,
             sensor_range=(-200, 200),
             number_of_sensors=64,
             x_range=(-500, 500),
             y_range=(0, 500),
             number_of_x_steps=1024,
             number_of_y_steps=1,
             simulation_area_offset=75,
             number_of_runs=32,
             add_noise=True,
             noise_power=1.5e-5,
             forward_and_backward_runs=False,
             folder_path="../../data/simulation/"):

    input_sensors = list(
        np.linspace(sensor_range[0], sensor_range[1], num=number_of_sensors))
    x_input = list(np.linspace(x_range[0], x_range[1], num=number_of_x_steps))
    y_input = list(
        np.linspace(y_range[0] + simulation_area_offset,
                    y_range[1] + simulation_area_offset,
                    num=number_of_y_steps))
    time_step = abs(x_input[1] - x_input[0]) / norm_w
    start_time = 0

    all_data = []
    all_labels = []
    all_timestamp = []
    for _ in tqdm(range(number_of_runs)):
        if forward_and_backward_runs:
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

    data_path = folder_path / f"a{a}_normw{norm_w}_theta{theta}.npy"
    labels_path = folder_path / f"a{a}_normw{norm_w}_theta{theta}_labels.npy"
    timestamp_path = folder_path / f"a{a}_normw{norm_w}_theta{theta}_timestamp.npy"

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

    log.debug(all_data.shape)
    log.debug(all_labels.shape)
    log.debug(all_timestamp.shape)
    np.save(data_path, all_data)
    np.save(labels_path, all_labels)
    np.save(timestamp_path, all_timestamp)
