if __name__ == "__main__":
    import sys
    sys.path.append("..")

import os
from translation_key import translation_key, model_key
import json
import sys

if __name__ == "__main__":
    sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
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
        real_angles.append(x_label)

    return np.mean(angles), np.mean(real_angles)


def save_results(x_pred, x_data, model_type, subset):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", ["#FFFFFF00", "#E76F51AA", "#E76F51DD", "#E76F51EE", "#E76F51FF"])
    idxes = np.asarray(x_data[:1000, 2] < 40).nonzero()[0]

    plt.plot(x_data[idxes, 0],
             x_data[idxes, 2],
             color='#2A9D8F',
             linestyle='solid',
             label="Real",
             linewidth=1,
             alpha=1)


    plt.hist2d(x_pred[:, 0],
               x_pred[:, 2],
               bins=(128, 128),
               label="Predicted",
               cmap=cmap)

    plt.ylim((-25, 25))
    plt.xlim((-500, 500))
    MSE = np.sqrt(np.square(np.subtract(x_data[:, 2], x_pred[:, 2]))).mean()
    MSE_std = np.sqrt(np.square(np.subtract(x_data[:, 2], x_pred[:, 2]))).std()
    t = plt.text(-400, 22, f"RMSE: {MSE:.2f} mm ($\\pm${MSE_std:.2f})")
    t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="white"))
    plt.title(
        f"Predicted vs Real Angle Per Run\n{model_key[model_type]} - {translation_key[subset]}"
    )
    # plt.grid(axis='y', linestyle='--', color="#2646533F", linewidth=0.4)

    plt.xlabel("s (mm)")
    plt.ylabel("Angle (degrees)")
    plt.xticks(np.arange(-500, 500+100, step=100))
    plt.grid(axis='y', linestyle='-', color="#AAAAAA", linewidth=1., alpha=0.5)
    plt.grid(axis='x', linestyle='-', color="#AAAAAA", linewidth=1., alpha=0.5)
    # MSE = np.square(np.subtract(real_angles, angles)).mean()
    # plt.text(0, 60, f"MSE: {MSE:.2f} degrees")

    hist_patch = mpatches.Patch(color='#E76F51FF', label='Predicted')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([hist_patch])

    plt.legend(handles=handles, loc="best", bbox_to_anchor=(0.6, 0., 0.4, 1.0) )
    # plt.show()

    plt.savefig(f"../results/angle_{model_type}_{subset}.pdf")
    plt.close()

    # Get result data
    results = {}
    results[f"combined"] = (float(MSE), float(MSE_std))

    with open(f"../results/angle_{model_type}_{subset}_results.json",
              "w") as write_file:
        json.dump(results, write_file, indent=4)


def retrieve_angle(subset, model_type):
    if model_type != "LSTM":
        # return main(subset, model_type)
        x_pred = np.load(f"../results/{model_type}/{subset}/x_pred_8.npy")[:,
                                                                           0:3]
        x_data = np.load(f"../results/{model_type}/{subset}/x_data_8.npy")[:,
                                                                           0:3]
    else:
        x_pred = np.load(f"../results/{model_type}/{subset}/y_pred_8.npy")[:,
                                                                           0:3]
        x_data = np.load(f"../results/{model_type}/{subset}/y_data_8.npy")[:,
                                                                           0:3]

        x_pred = x_pred.reshape(80, -1, 3)
        x_data = x_data.reshape(80, -1, 3)
        x_data = x_data[:, :x_pred.shape[1], :]
        x_pred = x_pred.reshape(-1, 3)
        x_data = x_data.reshape(-1, 3)

    # Load data
    # x_pred = np.load(f"../results/{model_type}/{subset}/x_pred_8.npy")[:, 0:3]
    # x_data = np.load(f"../results/{model_type}/{subset}/x_data_8.npy")[:, 0:3]

    save_results(x_pred, x_data, model_type, subset)
    # # plt.ylim((0, 80))
    # plt.xlabel("s (mm)")
    # plt.ylabel("Angle (degrees)")
    # plt.title(f"Estimated vs Real angle per run | {model_type} - {subset}")
    # plt.legend()
    # plt.savefig(f"./results/angle/{model_type}_{subset}_angle.pdf")
    # plt.close()


if __name__ == '__main__':
    # models = ["LSTM"]
    # models = ["INN"]
    models = ["INN", "PINN", "LSTM"]
    # models = ["INN", "PINN"]
    subsets = [
            "offset", "offset_inverse", "mult_path", "parallel", "far_off_parallel", "sine"
    ]
    for model in models:
        for subset in subsets:
            print(f"Model: {model} | Subset: {subset}")
            retrieve_angle(subset, model)
