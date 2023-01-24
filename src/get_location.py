if __name__ == "__main__":
    import sys
    sys.path.append("..")

import os
from translation_key import translation_key
import json
import sys

if __name__ == "__main__":
    sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib
from tqdm import tqdm

from lib.params import Data, Settings

plt.rcParams['axes.axisbelow'] = True
plt.rcParams['text.usetex'] = True


def get_location_from_data(data, labels, model, window_size=16, num_sensors=8):
    locations = []
    real_locations = []
    for idx in range(0, 1024, window_size * 8):
        input_data = data[idx:idx + window_size]
        input_data = np.reshape(input_data, (1, window_size, num_sensors * 2))
        y_pred = model.predict(input_data, verbose=0)
        x_label = labels[idx][2]

        locations.append(y_pred[0][2])
        real_locations.append(x_label)

    return np.mean(locations), np.mean(real_locations)


def save_results(x_pred, x_data, model_type, subset, MSE, MSE_std):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", ["#FFFFFF00", "#E76F51FA", "#E76F51FF"])

    plt.plot(x_data[:1000 if model_type == "LSTM" else 1020, 0],
             x_data[:1000 if model_type == "LSTM" else 1020, 1],
             color='#2A9D8F',
             linestyle='solid',
             label="Real",
             linewidth=1,
             alpha=1.)
    plt.hist2d(x_pred[:, 0],
               x_pred[:, 1],
               bins=(128, 128),
               label="Predicted",
               cmap=cmap)

    # TODO: Try out boxplots
    # plt.boxplot(x_pred[:, 1])

    plt.ylim((0, 250))
    plt.xlim((-500, 500))
    plt.text(-400, 235, f"MSE: {MSE:.2f} mm ($\\pm${MSE_std:.2f})")
    plt.title(
        f"Predicted vs Real Location Per Run\n{model_type} - {translation_key[subset]}"
    )

    plt.xlabel("s (mm)")
    plt.ylabel("d (mm)")
    plt.grid(axis='y', linestyle='-', color="#AAAAAAFF", linewidth=1.)
    plt.grid(axis='x', linestyle='-', color="#AAAAAAFF", linewidth=1.)

    plt.legend(loc="upper right")
    plt.show()

    # plt.savefig(f"../results/location_{model_type}_{subset}.pdf")
    # plt.close()

    # Get result data
    results = {}
    results[f"combined"] = (float(MSE), float(MSE_std))

    with open(f"../results/location_{model_type}_{subset}_results.json",
              "w") as write_file:
        json.dump(results, write_file, indent=4)


def retrieve_location(subset, model_type):
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

    if model_type == "LSTM":
        MSE = np.square(np.subtract(x_data[:, :2],
                                    x_pred[:, :2])).mean() * 0.008
        MSE_std = np.square(np.subtract(x_data[:, :2],
                                        x_pred[:, :2])).std() * 0.0008
    else:
        MSE = np.square(np.subtract(x_data[:, :2],
                                    x_pred[:, :2])).mean() * 0.012
        MSE_std = np.square(np.subtract(x_data[:, :2],
                                        x_pred[:, :2])).std() * 0.0012

    save_results(x_pred, x_data, model_type, subset, MSE, MSE_std)
    # # plt.ylim((0, 80))
    # plt.xlabel("s (mm)")
    # plt.ylabel("Location (mm)")
    # plt.title(f"Estimated vs Real location per run | {model_type} - {subset}")
    # plt.legend()
    # plt.savefig(f"./results/location/{model_type}_{subset}_location.pdf")
    # plt.close()


if __name__ == '__main__':
    # models = ["LSTM"]
    models = ["INN", "PINN", "LSTM"]
    # models = ["INN", "PINN"]
    subsets = [
        "offset", "offset_inverse", "mult_path", "parallel", "far_off_parallel"
    ]
    for model in models:
        for subset in subsets:
            print(f"Model: {model} | Subset: {subset}")
            retrieve_location(subset, model)
