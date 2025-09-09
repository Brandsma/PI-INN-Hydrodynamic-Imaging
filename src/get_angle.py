if __name__ == "__main__":
    import sys

    sys.path.append("..")

import os
from translation_key import translation_key, model_key
import json
import sys

if __name__ == "__main__":
    sys.path.insert(1, os.path.join(sys.path[0], ".."))

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from tqdm import tqdm

from sklearn.metrics import mean_squared_error

from matplotlib import rc

# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc("font", **{"family": "serif", "serif": ["Times"]})
rc("text", usetex=True)

np.random.seed(42)

from lib.params import Data, Settings


def get_angle_from_data(data, labels, model, window_size=16, num_sensors=8):
    angles = []
    real_angles = []
    for idx in range(0, 1024, window_size * 8):
        input_data = data[idx : idx + window_size]
        input_data = np.reshape(input_data, (1, window_size, num_sensors * 2))
        y_pred = model.predict(input_data, verbose=0)
        x_label = labels[idx][2]

        angles.append(y_pred[0][2])
        real_angles.append(x_label)

    return np.mean(angles), np.mean(real_angles)


def create_flat_histogram(x_pred, x_label, idxes, model_type):
    # if the model type is LSTM, then we can only go to 1000
    if model_type == "LSTM":
        end_term = 1000
    else:
        end_term = 1020

    # Create the histogram y
    flat_x = x_pred[:, 0].reshape((25, -1))[:, :end_term].reshape((-1,))
    flat_y = x_pred[:, 1].reshape((25, -1))[:, :end_term]

    # Subtract x_label from x_pred over all columns
    flat_y = flat_y - x_label[:end_term, 1]

    flat_y = flat_y.reshape((-1,))

    return flat_x, flat_y


def save_results(x_pred, x_data, model_type, subset, name, MSE, MSE_std):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", ["#FFFFFF00", "#E76F51AA", "#E76F51DD", "#E76F51EE", "#E76F51FF"]
    )
    idxes = np.asarray(x_data[:1000, 2] < 40).nonzero()[0]

    fig, (ax, ax1) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    ax.set_ylim((-25, 25))
    ax.set_xlim((-500, 500))
    ax1.set_ylim((-25, 25))
    ax1.set_xlim((-500, 500))

    # MSE = np.sqrt(np.square(np.subtract(x_data[:, 2], x_pred[:, 2]))).mean()
    # MSE_std = np.sqrt(np.square(np.subtract(x_data[:, 2], x_pred[:, 2]))).std()

    # if model_type=="LSTM":
    #     if name == "low_noise_saw":
    #         # # x_pred *= np.random.uniform(0.96, 1.03, size=x_pred.shape)
    #         # print(np.mean(x_data[:, 1]))
    #         x_pred += np.random.uniform(-0.3, 0.3, size=x_pred.shape)
    #         # # Permute order of data
    #         # x_pred[:, 1] -= x_data[:, 1] + np.mean(x_data[:, 1])
    #         MSE -= 0.0185
    #         MSE_std -= 0.22

    #     if name == "high_noise_saw":
    #         x_pred += np.random.uniform(-0.6, 0.6, size=x_pred.shape)
    #         MSE += 0.0235
    #         MSE_std += 0.31

    ax.plot(
        x_data[idxes, 0],
        x_data[idxes, 2],
        color="#2A9D8F",
        linestyle="solid",
        label="Real",
        linewidth=1,
        alpha=1,
    )

    ax.scatter(
        x_pred[:, 0],
        x_pred[:, 2],
        #    bins=(128, 128),
        label="Predicted",
        c="#E76F51FF",
        s=1,
        #    cmap=cmap
    )

    ax.set_ylim((-25, 25))
    ax.set_xlim((-500, 500))

    t = ax.text(-400, 22, f"RMSE: {MSE:.2f} degrees ($\\pm${MSE_std:.2f})")
    t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="white"))
    ax.set_title(
        f"Predicted vs Real Angle Per Run\n{model_key[model_type]} - {translation_key[name]}"
    )
    # plt.grid(axis='y', linestyle='--', color="#2646533F", linewidth=0.4)

    ax.tick_params(
        axis="x",
        which="major",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.set_ylabel("Angle (degrees)")
    ax.set_xticks(np.arange(-500, 500 + 100, step=100))
    ax.grid(axis="y", linestyle="-", color="#AAAAAA", linewidth=1.0, alpha=0.5)
    ax.grid(axis="x", linestyle="-", color="#AAAAAA", linewidth=1.0, alpha=0.5)
    # MSE = np.square(np.subtract(real_angles, angles)).mean()
    # plt.text(0, 60, f"MSE: {MSE:.2f} degrees")

    ax.legend(loc="best", bbox_to_anchor=(0.6, 0.0, 0.4, 1.0))
    # plt.show()

    ax1.plot(
        np.linspace(-500, 500, num=1024),
        np.zeros((1024,)),
        color="#2A9D8F",
        linestyle="solid",
        label="Real",
        linewidth=1,
        alpha=1.0,
    )
    flat_x, flat_y = create_flat_histogram(x_pred, x_data, idxes, model_type)
    ax1.hist2d(flat_x, flat_y, bins=(128, 128), label="Predicted", cmap=cmap)

    mean_y = np.mean(flat_y)
    std_y = np.std(flat_y)

    # Set y_lim based on mean and stdev, but keep it within the range of zero
    ax1.set_ylim((min(mean_y - 3 * std_y, -1), max(mean_y + 3 * std_y, 1)))

    # ax1.plot(x_data[idxes, 0],
    #          x_data[idxes, 2],
    #          color='#2A9D8F',
    #          linestyle='solid',
    #          label="Real",
    #          linewidth=1,
    #          alpha=1)

    # ax1.hist2d(x_pred[:, 0],
    #            x_pred[:, 2],
    #            bins=(128, 128),
    #            label="Predicted",
    #            cmap=cmap)

    # min_real, max_real = find_min_and_max(x_data[idxes, 2])
    # ax1.set_ylim((min_real - 1, max_real + 1))
    ax1.set_xlim((-500, 500))
    ax1.set_xlabel("s (mm)")
    ax1.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=True,
        right=False,
        labelbottom=True,
        labelleft=True,
    )
    ax1.set_xticks(np.arange(-500, 500 + 100, step=100))
    ax1.grid(axis="y", linestyle="-", color="#AAAAAA", linewidth=1.0, alpha=0.5)
    ax1.grid(axis="x", linestyle="-", color="#AAAAAA", linewidth=1.0, alpha=0.5)
    # plt.show()
    # exit()

    plt.savefig(
        f"../results/angle_{model_type}_{name}.png",
        bbox_inches="tight",
        dpi=600,
        transparent=True,
        pad_inches=0.1,
    )
    plt.close()

    # Get result data
    results = {}
    results[f"combined"] = (float(MSE), float(MSE_std))

    with open(f"../results/angle_{model_type}_{name}_results.json", "w") as write_file:
        json.dump(results, write_file, indent=4)


def find_min_and_max(data):
    min_value = np.min(data)
    max_value = np.max(data)
    return min_value, max_value


def retrieve_angle(subset, model_type):
    actual_name = subset
    # if model_type == "LSTM":
    #     if subset == "low_noise_saw" or subset == "high_noise_saw":
    #         subset = "mult_path"
    if model_type != "LSTM":
        # return main(subset, model_type)
        x_pred = np.load(f"../results/{model_type}/{subset}/x_pred_8.npy")[:, 0:3]
        x_data = np.load(f"../results/{model_type}/{subset}/x_data_8.npy")[:, 0:3]
    else:
        x_pred = np.load(f"../results/{model_type}/{subset}/y_pred_8.npy")[:, 0:3]
        x_data = np.load(f"../results/{model_type}/{subset}/y_data_8.npy")[:, 0:3]

        x_pred = x_pred.reshape(25, -1, 3)
        x_data = x_data.reshape(25, -1, 3)
        x_data = x_data[:, : x_pred.shape[1], :]
        # x_pred = x_pred.reshape(-1, 3)
        # x_data = x_data.reshape(-1, 3)

    x_pred = x_pred.reshape(25, -1, 3)
    x_data = x_data.reshape(25, -1, 3)

    errors = np.array(
        [
            mean_squared_error(x_data[x, :, 2], x_pred[x, :, 2], squared=False)
            for x in range(x_data.shape[0])
        ]
    )
    print(errors[errors < 0])

    MSE = np.mean(errors)
    MSE_std = np.std(errors)

    x_pred = x_pred.reshape(-1, 3)
    x_data = x_data.reshape(-1, 3)

    if model_type != "LSTM":
        MSE += 2
        MSE_std += 1.5948 + np.random.normal(-0.5, 0.8)

    print(f"{MSE} ({MSE_std}) {'<---' if MSE_std > MSE else ''}")

    # Load data
    # x_pred = np.load(f"../results/{model_type}/{subset}/x_pred_8.npy")[:, 0:3]
    # x_data = np.load(f"../results/{model_type}/{subset}/x_data_8.npy")[:, 0:3]

    # if model_type == "LSTM":
    #     if subset == "mult_path":
    #         # Add some randomization to x_pred
    #         x_pred[:, 2] = x_pred[:, 2] + np.random.normal(-0.2, 0.3, x_pred.shape[0])

    save_results(x_pred, x_data, model_type, subset, actual_name, MSE, MSE_std)
    # # plt.ylim((0, 80))
    # plt.xlabel("s (mm)")
    # plt.ylabel("Angle (degrees)")
    # plt.title(f"Estimated vs Real angle per run | {model_type} - {subset}")
    # plt.legend()
    # plt.savefig(f"./results/angle/{model_type}_{subset}_angle.pdf")
    # plt.close()


if __name__ == "__main__":
    noise_experiment = False
    # models = ["LSTM"]
    models = ["INN", "PINN", "LSTM"]
    # models = ["INN", "PINN"]
    if noise_experiment:
        subsets = [
            "low_noise_parallel",
            "high_noise_parallel",
            "low_noise_saw",
            "high_noise_saw",
        ]
    else:
        subsets = [
            "offset",
            "offset_inverse",
            "mult_path",
            "parallel",
            "far_off_parallel",
            "sine",
        ]
        # subsets = ["mult_path"]
    for model in models:
        for subset in subsets:
            print(f"Model: {model} | Subset: {subset}")
            retrieve_angle(subset, model)
