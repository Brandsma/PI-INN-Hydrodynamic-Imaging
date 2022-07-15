import os
import sys

if __name__ == "__main__":
    sys.path.insert(1, os.path.join(sys.path[0], '..'))

import math

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from lib.params import Data, Settings


def calculate_angle(start_point, terminal_point):
    if type(start_point) is not np.ndarray:
        start_point = np.array(start_point)
    if type(terminal_point) is not np.ndarray:
        terminal_point = np.array(terminal_point)

    dir_vector = terminal_point - start_point
    if dir_vector[0] == 0:
        # TODO: Is this correct? It seems logical that if the x-position
        # does not change it is parallel to the y-axis
        return 90
    return np.arctan(dir_vector[1] / dir_vector[0]) * (180 / math.pi)


def get_angle_from_data(data, labels, model, window_size=16):
    prev_xy = 0
    prev_xy_label = 0

    angles = []
    real_angles = []
    for idx in range(0, 1024, window_size * 8):
        input_data = data[idx:idx + window_size]
        input_data = np.reshape(input_data, (1, window_size, 128))
        y_pred = model.predict(input_data, verbose=0)
        xy_label = labels[idx]

        if idx != 0:
            angle = calculate_angle(prev_xy, y_pred[0])
            real_angle = calculate_angle(prev_xy_label, xy_label)
            # print(prev_xy_label, xy_label)

            angles.append(angle)
            real_angles.append(real_angle)

        prev_xy = y_pred[0]
        prev_xy_label = xy_label
    return np.mean(angles), np.mean(real_angles)


def main():
    train_location = "../data/simulation_data/combined.npy"
    trained_model_location = "../data/trained_models/window_size:16&stride:2&n_nodes:128&alpha:0.05&decay:1e-09&n_epochs:4&shuffle_data:True&data_split:0.8&dropout_ratio:0&ac_fun:relu"

    settings = Settings.from_model_location(trained_model_location,
                                            data_location=train_location)

    # Load data
    data = Data(settings, train_location)

    data.normalize()

    new_model = tf.keras.models.load_model(trained_model_location)

    angles = []
    real_angles = []

    for run_idx in tqdm(range(data.test_data.shape[0])):
        angle_results = get_angle_from_data(data.test_data[run_idx],
                                            data.test_labels[run_idx],
                                            new_model)
        angles.append(angle_results[0])
        real_angles.append(angle_results[1])

    plt.plot(angles, "bo", label="Predicted Speed")
    plt.plot(real_angles, "r.", label="Real Speed")

    for idx in range(len(angles)):
        line_x_values = [idx, idx]
        line_y_values = [angles[idx], real_angles[idx]]
        plt.plot(line_x_values, line_y_values, "k-", linestyle="-")
    plt.ylim((-180, 180))
    plt.xlabel("run")
    plt.ylabel("Angle (degrees)")
    MSE = np.square(np.subtract(real_angles, angles)).mean()
    plt.text(0, 60, f"MSE: {MSE:.2f} mm/s")
    plt.title(f"Estimated vs Real angle per run")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
