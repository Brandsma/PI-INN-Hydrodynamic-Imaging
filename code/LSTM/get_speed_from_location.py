import os
import sys

import numpy as np
import scipy.io as sio
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.params import Data, Settings


def read_inputs():
    n_nodes = 100
    n_epochs = 30
    window_size = 16
    stride = 2
    alpha = 0.05
    decay = 1e-9
    shuffle_data = False
    data_split = 0.8
    dropout = 0
    train_loc = "../../data/a10_theta0.npy"
    ac_fun = "relu"
    return n_nodes, n_epochs, window_size, stride, \
        alpha, decay, shuffle_data, data_split, dropout, train_loc, ac_fun


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
        './trained_models/win16_stride2_epochs30_dropout0_latest')
    new_model.summary()
    prev_x = 0
    prev_time = 0
    prev_x_label = 0

    speeds = []
    real_speeds = []
    # run_idx = 8

    for run_idx in range(data.test_data.shape[0]):

        for idx in tqdm(range(0, 1024, window_size * 32)):
            input_data = data.test_data[run_idx][idx:idx + window_size]
            input_data = np.reshape(input_data, (1, window_size, 128))
            y_pred = new_model.predict(input_data)
            time = data.test_timestamp[run_idx][idx][0]
            # print(time)
            x_label = data.test_labels[run_idx][idx][0]
            # print(input_data)
            # print(time)
            # print(y_pred)

            if idx != 0:

                # prev_x = data.test_labels[0][idx + window_size - 1:idx + window_size][0][0]
                # y_pred[0][0] = data.test_labels[0][idx + window_size:idx + 16][0][0]
                speed = abs(y_pred[0][0] - prev_x) / abs(time - prev_time)
                real_speed = abs(x_label - prev_x_label) / abs(time -
                                                               prev_time)

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

    plt.plot(speeds, label="Predicted Speed")
    plt.plot(real_speeds, label="Real Speed")
    plt.ylim((0, 70))
    plt.xlabel("Window")
    plt.ylabel("Speed (m/s)")
    MSE = np.square(np.subtract(real_speeds, speeds)).mean()
    plt.text(0, 60, f"MSE: {MSE:.2f}")
    plt.title(
        f"Estimated vs Real speed per sensor reading window, run: {run_idx}")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
