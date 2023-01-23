import os
import math
import sys
import INN.hydro as hydro

if __name__ == "__main__":
    sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from lib.params import Data, Settings

def get_speed_from_inn_predicts(preds, labels, timestamp, step_size=16, div_number=1024, epsilon=0.1):
    prev_x = [0,0]
    prev_time = 0

    real_speeds = []
    for idx in range(0,div_number, step_size):
        time = timestamp[idx][0]

        if idx != 0:
            speed = math.dist(labels[idx][0:1], prev_x) / abs(time - prev_time)

            real_speeds.append(speed)

        prev_x = labels[idx][0:1]
        prev_time = time
    prev_x = [0,0]
    prev_time = 0


    speeds = []
    for idx in range(0,div_number,step_size):
        time = timestamp[idx][0]

        if idx != 0:
            speed = math.dist(preds[idx][0:1], prev_x) / abs(time - prev_time + epsilon)

            speeds.append(speed)

        prev_x = preds[idx][0:1]
        prev_time = time
    prev_x = [0,0]
    prev_time = 0
    return np.mean(speeds), np.mean(real_speeds)
    # return np.mean(real_speeds)

def get_speed_from_model_predicts(model_predicts, labels, timestamp, window_size=16):
    prev_x = [0,0]
    prev_time = 0

    real_speeds = []
    for idx in range(0,1024,window_size):
        time = timestamp[idx][0]

        if idx != 0:
            # TODO: Adjust speed calculation for varying y
            speed = math.dist(labels[idx][0:1], prev_x) / abs(time - prev_time)

            real_speeds.append(speed)

        prev_x = labels[idx][0:1]
        prev_time = time
    prev_x = [0,0]
    prev_time = 0

    speeds = []
    for idx, y_pred in enumerate(model_predicts):
        if (len(model_predicts) + window_size > len(timestamp)):
            window_size -= 1
        time = timestamp[idx + window_size][0]

        if idx != 0:
            # TODO: Adjust speed calculation for varying y
            speed = math.dist(y_pred[0:1], prev_x) / abs(time - prev_time)

            speeds.append(speed)

        prev_x = y_pred[0:1]
        prev_time = time
    return np.mean(speeds), np.mean(real_speeds)

def get_speed_from_data(data, labels, timestamp, model, window_size=16, num_sensors=8):
    print("Getting speed from data")
    prev_x = [0, 0]
    prev_time = 0
    prev_x_label = [0, 0]

    speeds = []
    real_speeds = []
    for idx in range(0, 1024, window_size * 8):
        input_data = data[idx:idx + window_size]
        input_data = np.reshape(input_data, (1, window_size, num_sensors * 2))
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


def main(subset="offset", model_type="INN"):
    if model_type == "INN" or model_type == "PINN":
        return main_inn(subset, model_type)
    elif model_type == "LSTM":
        print("Using LSTM")
    else:
        print("No valid model type given")
        return

    train_location = f"../data/simulation_data/{subset}/combined.npy"
    trained_model_location = "../data/trained_models/LSTM/window_size:16&stride:2&n_nodes:256&alpha:0.05&decay:1e-09&n_epochs:16&shuffle_data:True&data_split:0.8&dropout_ratio:0&ac_fun:tanh&num_sensors:8"

    settings = Settings.from_model_location(trained_model_location,
                                            data_location=train_location)
    settings.num_sensors = 8

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
    plt.title(f"Estimated vs Real speed per run | {model_type} - {subset}")
    plt.legend()
    # plt.show()

    plt.savefig(f"./results/speed/{model_type}_{subset}.pdf")
    plt.close()

def main_inn(subset="offset", model_type="INN"):
    print(f"Using {model_type}")
    train_location = f"../data/simulation_data/{subset}/combined.npy"
    trained_model_location = "../data/trained_models/window_size:16&stride:2&n_nodes:128&alpha:0.05&decay:1e-09&n_epochs:8&shuffle_data:True&data_split:0.8&dropout_ratio:0&ac_fun:tanh"

    settings = Settings.from_model_location(trained_model_location,
                                            data_location=train_location)

    settings.shuffle_data = True
    settings.num_sensors = 8
    settings.seed = 42

    # Load data
    data = Data(settings, train_location)

    # data.normalize()

    # new_model = tf.keras.models.load_model(trained_model_location)

    x_pred = np.load(f"../results/{model_type}/{subset}/x_pred_8.npy")[:, 0:3]
    x_data = np.load(f"../results/{model_type}/{subset}/x_data_8.npy")[:, 0:3]

    # hydro.plot_results_from_array(x_data, x_pred, subset, 8, title=f"Sensors: 8", savefig=False)

    # exit()

    speeds = []
    real_speeds = []

    div_number = 1024
    if x_pred.shape[0] % div_number != 0:
        div_number = 1020

    step_size = div_number // 16

    for run_idx in range(x_pred.shape[0]//div_number):
        speed_results = get_speed_from_inn_predicts(x_pred[0 + (div_number * run_idx):div_number + (div_number * run_idx)],
                                            x_data[0 + (div_number * run_idx):div_number + (div_number * run_idx)],
                                                    data.test_timestamp[run_idx],
                                                    step_size=step_size)
        speeds.append(speed_results[0])
        real_speeds.append(speed_results[1])

    plt.plot(speeds, "bo", label="Predicted Speed")
    plt.plot(real_speeds, "r.", label="Real Speed")

    for idx in range(len(speeds)):
        line_x_values = [idx, idx]
        line_y_values = [speeds[idx], real_speeds[idx]]
        plt.plot(line_x_values, line_y_values, "k-")
    plt.ylim((0, 80))
    plt.xlabel("run")
    plt.ylabel("Speed (mm/s)")
    MSE = np.square(np.subtract(real_speeds, speeds)).mean()
    plt.text(0, 65, f"MSE: {MSE:.2f} mm/s")
    plt.title(f"Estimated vs Real speed per run | {model_type} - {subset}")
    plt.legend()
    plt.savefig(f"./results/speed/{model_type}_{subset}.pdf")
    plt.close()

if __name__ == '__main__':
    models = ["INN", "PINN", "LSTM"]
    # models= ["LSTM"]
    # models= ["INN"]
    subsets = ["offset", "offset_inverse", "mult_path", "parallel", "far_off_parallel"]
    # subsets = ["offset"]
    for model in models:
        for subset in subsets:
            print(f"Running {model} on subset: '{subset}'...")
            main(subset, model_type=model)
