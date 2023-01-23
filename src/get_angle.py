if __name__=="__main__":
    import sys
    sys.path.append("..")

import os
import sys

if __name__ == "__main__":
    sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from lib.params import Data, Settings


def get_angle_from_data(data, labels, model, window_size=16, num_sensors=8):
    angles = []
    real_angles = []
    for idx in range(0, 1024, window_size * 8):
        input_data = data[idx:idx + window_size]
        input_data = np.reshape(input_data, (1, window_size, num_sensors * 2))
        y_pred = model.predict(input_data, verbose=0)
        x_label = labels[idx][2]

        angles.append(y_pred[0][2])
        real_angles.append( x_label)

    return np.mean(angles), np.mean(real_angles)


def main(subset, model_type="LSTM"):
    train_location = f"../data/simulation_data/{subset}/combined.npy"
    trained_model_location = "../data/trained_models/LSTM/window_size:16&stride:2&n_nodes:256&alpha:0.05&decay:1e-09&n_epochs:16&shuffle_data:True&data_split:0.8&dropout_ratio:0&ac_fun:tanh&num_sensors:8"

    settings = Settings.from_model_location(trained_model_location,
                                            data_location=train_location)

    settings.num_sensors = 8

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

    plt.plot(angles, "bo", label="Predicted Angle")
    plt.plot(real_angles, "r.", label="Real Angle")

    for idx in range(len(angles)):
        line_x_values = [idx, idx]
        line_y_values = [angles[idx], real_angles[idx]]
        plt.plot(line_x_values, line_y_values, "k-")
    # plt.ylim((0, 70))
    plt.xlabel("s (mm)")
    plt.ylabel("Angle (degrees)")
    # MSE = np.square(np.subtract(real_angles, angles)).mean()
    # plt.text(0, 60, f"MSE: {MSE:.2f} degrees")
    plt.title(f"Estimated vs Real angle per run | {model_type} - {subset}")
    plt.legend()
    plt.savefig(f"./results/angle/{model_type}_{subset}_angle.pdf")
    plt.close()

def retrieve_angle(subset, model_type):
    print("Retrieving angle from model")
    if model_type == "LSTM":
        return main(subset, model_type)

    # Load data
    x_pred = np.load(f"../results/{model_type}/{subset}/x_pred_8.npy")[:, 0:3]
    x_data = np.load(f"../results/{model_type}/{subset}/x_data_8.npy")[:, 0:3]

    plt.hist2d(x_pred[:, 0], x_pred[:, 2], bins=(128,128), label="predicted", cmap=plt.cm.viridis)
    plt.plot(x_data[:1020, 0], x_data[:1020, 2], color='red', linestyle='dashed', label="label", linewidth=2, alpha=0.4)

    # plt.ylim((0, 80))
    plt.xlabel("s (mm)")
    plt.ylabel("Angle (degrees)")
    plt.title(f"Estimated vs Real angle per run | {model_type} - {subset}")
    plt.legend()
    plt.savefig(f"./results/angle/{model_type}_{subset}_angle.pdf")
    plt.close()


if __name__ == '__main__':
    # models = ["LSTM"]
    # models = ["INN", "PINN", "LSTM"]
    models = ["INN", "PINN"]
    subsets = ["offset", "offset_inverse", "mult_path", "parallel", "far_off_parallel"]
    for model in models:
        for subset in subsets:
            print(f"Model: {model} | Subset: {subset}")
            retrieve_angle(subset, model)
