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


def calculate_path(points, num_steps, simulation_area_offset=75):
    subpath_steps = num_steps // (len(points) - 1)
    path = []

    for idx in range(len(points) - 1):
        x_input = list(
            np.linspace(points[idx][0], points[idx + 1][0], num=subpath_steps))

        y_input = list(
            np.linspace(points[idx][1] + simulation_area_offset,
                        points[idx + 1][1] + simulation_area_offset,
                        num=subpath_steps))
        subpath = list(zip(x_input, y_input))
        path.extend(subpath)

    return path


def calculate_angle(start_point, terminal_point):
    dir_vector = np.array(terminal_point) - np.array(start_point)
    if dir_vector[0] == 0:
        # TODO: Is this correct? It seems logical that if the y-position
        # does not change it is parallel to the x-axis
        return 90
    return np.arctan(dir_vector[1] / dir_vector[0]) * (180 / math.pi)


## Simulation ##


def simulate(theta=0,
             a=10,
             norm_w=10,
             sensor_range=(-200, 200),
             number_of_sensors=64,
             points=[(-500, 0), (500, 0)],
             number_of_steps=1024,
             simulation_area_offset=75,
             number_of_runs=32,
             add_noise=True,
             noise_power=1.5e-5,
             forward_and_backward_runs=False,
             folder_path="../../data/simulation/"):

    input_sensors = list(
        np.linspace(sensor_range[0], sensor_range[1], num=number_of_sensors))

    path = calculate_path(points,
                          number_of_steps,
                          simulation_area_offset=simulation_area_offset)

    time_step = np.linalg.norm(np.array(path[0]) - np.array(path[1])) / norm_w
    start_time = 0

    all_data = []
    all_labels = []
    all_timestamp = []
    for _ in tqdm(range(number_of_runs)):
        if forward_and_backward_runs:
            path = list(reversed(path))
        time = start_time
        data = []
        labels = []
        timestamp = []
        for path_idx, (x, y) in enumerate(path):

            # TODO: Update theta
            if path_idx != (len(path) - 1):
                theta = calculate_angle([x, y], path[path_idx + 1])
            data.append([])
            labels.append([])
            timestamp.append([])
            for input_sensor in input_sensors:
                # NOTE: the x and y coordinates are different than the array coordinates
                data[path_idx].append(
                    v_x(input_sensor, x, y + 1, theta, a, norm_w))
                data[path_idx].append(
                    v_y(input_sensor, x, y + 1, theta, a, norm_w))

            labels[path_idx].append(x)
            labels[path_idx].append(y + 1)
            timestamp[path_idx].append(time)
            time += time_step

            if add_noise:
                data[path_idx] += np.random.normal(0, noise_power,
                                                   len(data[path_idx]))
        all_data.append(data)
        all_labels.append(labels)
        all_timestamp.append(timestamp)

    data_path = folder_path / f"a{a}_normw{norm_w}_theta{theta}.npy"
    labels_path = folder_path / f"a{a}_normw{norm_w}_theta{theta}_labels.npy"
    timestamp_path = folder_path / f"a{a}_normw{norm_w}_theta{theta}_timestamp.npy"

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    all_timestamp = np.array(all_timestamp)

    log.debug(all_data.shape)
    log.debug(all_labels.shape)
    log.debug(all_timestamp.shape)
    np.save(data_path, all_data)
    np.save(labels_path, all_labels)
    np.save(timestamp_path, all_timestamp)
