import os
import sys

import numpy as np
import scipy.io as sio
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from lib.params import Data, Settings


def read_inputs():
    n_nodes = 100
    n_epochs = 30
    window_size = 16
    stride = 2
    alpha = 0.05
    decay = 1e-9
    shuffle_data = True
    data_split = 0.8
    dropout = 0
    train_loc = "../data/simulation_data/combined.npy"
    ac_fun = "relu"
    return n_nodes, n_epochs, window_size, stride, \
        alpha, decay, shuffle_data, data_split, dropout, train_loc, ac_fun


def get_speed_from_data(data, labels, timestamp, model, window_size=16):
    prev_x = 0
    prev_time = 0
    prev_x_label = 0

    speeds = []
    real_speeds = []
    for idx in range(0, 1024, window_size * 8):
        input_data = data[idx:idx + window_size]
        input_data = np.reshape(input_data, (1, window_size, 128))
        y_pred = model.predict(input_data)
        time = timestamp[idx][0]
        # print(time)
        x_label = labels[idx][0]
        # print(input_data)
        # print(time)
        # print(y_pred)

        if idx != 0:
            # prev_x = labels[0][idx + window_size - 1:idx + window_size][0][0]
            # y_pred[0][0] = labels[0][idx + window_size:idx + 16][0][0]
            speed = abs(y_pred[0][0] - prev_x) / abs(time - prev_time)
            real_speed = abs(x_label - prev_x_label) / abs(time - prev_time)

            # print("real loc1:", x_label)
            # print("real loc0:", prev_x_label)
            # print("loc1:", y_pred[0][0])
            # print("loc0:", prev_x)
            # print("delta meter:", (y_pred[0][0] - prev_x),
            #       "delta seconds:", (time - prev_time))
            # print("Speed:", speed)
            # print("Real Speed:", real_speed)
            # print(
            #     "---------------------------------------------------------"
            # )
            speeds.append(speed)
            real_speeds.append(real_speed)

        prev_x = y_pred[0][0]
        prev_x_label = x_label
        prev_time = time
    return np.mean(speeds), np.mean(real_speeds)


def main():
    (n_nodes, n_epochs, window_size, stride, alpha, decay, \
     shuffle_data, data_split, dropout_ratio, train_location, ac_fun) =  \
        read_inputs()

    # Load settings
    settings = Settings(window_size, stride, n_nodes, \
                        alpha, decay, n_epochs, shuffle_data, data_split, dropout_ratio, \
               train_location, ac_fun)
    # Load data
    data = Data(settings, train_location)

    new_model = tf.keras.models.load_model(
        '../data/trained_models/peregrine/win16_stride2_epochs120_dropout0_latest')
    new_model.summary()

    speeds = []
    real_speeds = []
    # run_idx = 8

    for run_idx in tqdm(range(data.test_data.shape[0])):
        speed_results = get_speed_from_data(data.test_data[run_idx],
                                            data.test_labels[run_idx],
                                            data.test_timestamp[run_idx],
                                            new_model)
        speeds.append(speed_results[0])
        real_speeds.append(speed_results[1])

    plt.plot(speeds, "bo", label="Predicted Speed")
    plt.plot(real_speeds, "r.", label="Real Speed")

    for idx in range(len(speeds)):
        line_x_values = [idx, idx]
        line_y_values = [speeds[idx], real_speeds[idx]]
        plt.plot(line_x_values, line_y_values, "k-", linestyle="-")
    plt.ylim((0, 70))
    plt.xlabel("run")
    plt.ylabel("Speed (mm/s)")
    MSE = np.square(np.subtract(real_speeds, speeds)).mean()
    plt.text(0, 60, f"MSE: {MSE:.2f} mm/s")
    plt.title(f"Estimated vs Real speed per run")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
