import os
import math
import sys

if __name__ == "__main__":
    sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from simulation.simulation import simulate

from lib.params import Data, Settings

def get_angle_from_data(data, labels, model, window_size=16):
    angles = []
    real_angles = []
    for idx in range(0, 1024, window_size * 8):
        input_data = data[idx:idx + window_size]
        input_data = np.reshape(input_data, (1, window_size, 128))
        y_pred = model.predict(input_data, verbose=0)
        x_label = labels[idx][2]

        angles.append(y_pred[0][2])
        real_angles.append( x_label)

    return np.mean(angles), np.mean(real_angles)

def get_angle_error_distribution(angles):
    # train_location = "../data/simulation_data/combined.npy"
    trained_model_location = "../data/trained_models/window_size:16&stride:2&n_nodes:128&alpha:0.05&decay:1e-09&n_epochs:4&shuffle_data:True&data_split:0.8&dropout_ratio:0&ac_fun:relu"


    for angle in angles:
        if angle == 90:
            print("Angle of 90 does not work here, since starting point and end point are fixed at -500 and 500")
            continue

        right_height = 1000 * math.tan(angle)

        data, labels, timestamps, volumes = simulate(points=[(-500, 0), (500,right_height)] ,a=20, norm_w=20, number_of_runs=8, save_to_disk=False)

        settings = Settings()

        # Load data
        data = Data.from_data(settings, data, labels, timestamps, volumes)

        data.normalize()

        new_model = tf.keras.models.load_model(trained_model_location)

        speeds = []
        real_speeds = []

        angles = []
        real_angles = []

        for run_idx in tqdm(range(data.test_data.shape[0])):
            print(data.test_labels[run_idx])
            angle_results = get_angle_from_data(data.test_data[run_idx],
                                                data.test_labels[run_idx],
                                                new_model)
            print(angle_results)
            angles.append(angle_results[0])
            real_angles.append(angle_results[1])

        plt.plot(angles, "bo", label="Predicted Angle")
        plt.plot(real_angles, "r.", label="Real Angle")

        for idx in range(len(angles)):
            line_x_values = [idx, idx]
            line_y_values = [angles[idx], real_angles[idx]]
            plt.plot(line_x_values, line_y_values, "k-")
        plt.ylim((0, 70))
        plt.xlabel("run")
        plt.ylabel("Angle (degrees)")
        MSE = np.square(np.subtract(real_angles, angles)).mean()
        plt.text(0, 60, f"MSE: {MSE:.2f} degrees")
        plt.title(f"Estimated vs Real angle per run")
        plt.legend()
        plt.show()

def get_speed_from_data(data, labels, timestamp, model, window_size=16):
    prev_x = [0, 0]
    prev_time = 0
    prev_x_label = [0, 0]

    speeds = []
    real_speeds = []
    for idx in range(0, 1024, window_size * 8):
        input_data = data[idx:idx + window_size]
        input_data = np.reshape(input_data, (1, window_size, 128))
        y_pred = model.predict(input_data, verbose=0)
        time = timestamp[idx][0]
        x_label = labels[idx][0:1]

        if idx != 0:
            # TODO: Adjust speed calculation for varying y
            speed = math.dist(y_pred[0][0:1], prev_x) / abs(time - prev_time)
            real_speed = math.dist(x_label, prev_x_label) / abs(time - prev_time)

            speeds.append(speed)
            real_speeds.append(real_speed)

        prev_x = y_pred[0][0:1]
        prev_x_label = x_label
        prev_time = time
    return np.mean(speeds), np.mean(real_speeds)

def get_speed_error_distribution(speeds):
    # train_location = "../data/simulation_data/combined.npy"
    trained_model_location = "../data/trained_models/window_size:16&stride:2&n_nodes:128&alpha:0.05&decay:1e-09&n_epochs:4&shuffle_data:True&data_split:0.8&dropout_ratio:0&ac_fun:relu"


    for speed in speeds:

        right_height = 1000 * math.tan(speed)

        data, labels, timestamps, volumes = simulate(a=20, norm_w=speed, number_of_runs=8, save_to_disk=False)

        settings = Settings()

        # Load data
        data = Data.from_data(settings, data, labels, timestamps, volumes)

        data.normalize()

        new_model = tf.keras.models.load_model(trained_model_location)

        speeds = []
        real_speeds = []

        angles = []
        real_angles = []

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
    # angles = [0,15,30,45,60,75,90]
    # get_angle_error_distribution(angles)
    speed_list = [10,20,30,40,50]
    get_speed_error_distribution(speed_list)
