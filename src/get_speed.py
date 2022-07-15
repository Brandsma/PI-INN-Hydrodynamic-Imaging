import os
import sys

if __name__ == "__main__":
    sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from lib.params import Data, Settings


def get_speed_from_data(data, labels, timestamp, model, window_size=16):
    prev_x = 0
    prev_time = 0
    prev_x_label = 0

    speeds = []
    real_speeds = []
    for idx in range(0, 1024, window_size * 8):
        input_data = data[idx:idx + window_size]
        input_data = np.reshape(input_data, (1, window_size, 128))
        y_pred = model.predict(input_data, verbose=0)
        time = timestamp[idx][0]
        x_label = labels[idx][0]

        if idx != 0:
            speed = abs(y_pred[0][0] - prev_x) / abs(time - prev_time)
            real_speed = abs(x_label - prev_x_label) / abs(time - prev_time)

            speeds.append(speed)
            real_speeds.append(real_speed)

        prev_x = y_pred[0][0]
        prev_x_label = x_label
        prev_time = time
    return np.mean(speeds), np.mean(real_speeds)


def main():
    train_location = "../data/simulation_data/combined.npy"
    trained_model_location = "../data/trained_models/window_size:16&stride:2&n_nodes:128&alpha:0.05&decay:1e-09&n_epochs:4&shuffle_data:True&data_split:0.8&dropout_ratio:0&ac_fun:relu"

    settings = Settings.from_model_location(trained_model_location,
                                            data_location=train_location)

    # Load data
    data = Data(settings, train_location)

    data.normalize()

    new_model = tf.keras.models.load_model(trained_model_location)

    speeds = []
    real_speeds = []

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
        plt.plot(line_x_values, line_y_values, "k-")
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
