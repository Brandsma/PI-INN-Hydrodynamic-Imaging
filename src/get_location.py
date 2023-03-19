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
import scipy.stats as st

from sklearn.metrics import mean_squared_error

from lib.params import Data, Settings

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)


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

def crop(image, x1, x2, y1, y2):
    """
    Return the cropped image at the x1, x2, y1, y2 coordinates
    """
    if x2 == -1:
        x2=image.shape[1]-1
    if y2 == -1:
        y2=image.shape[0]-1

    mask = np.zeros(image.shape)
    mask[y1:y2+1, x1:x2+1]=1
    m = mask>0

    return image[m].reshape((y2+1-y1, x2+1-x1))

def add_subplot_axes(ax,rect,facecolor='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x,y,width,height],facecolor=facecolor)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

def find_min_and_max(data):
    min_value = np.min(data)
    max_value = np.max(data)
    return min_value, max_value

def create_flat_histogram(x_pred, x_label, model_type):
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


def save_results(x_pred, x_data, model_type, subset, MSE, MSE_std, name):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", ["#FFFFFF00", "#E76F51AA", "#E76F51DD", "#E76F51EE", "#E76F51FF"])

    fig, (ax, ax1) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]})

    # fig.tick_params(axis='both', which='major', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    # rect = [0.0,0.2,1.0,1.0]
    # ax = add_subplot_axes(ax_fig,rect)


    ax.set_ylim((0, 250))
    ax.set_xlim((-500, 500))

    ax1.set_ylim((0, 250))
    ax1.set_xlim((-500, 500))

    ax.plot(x_data[:1000 if model_type == "LSTM" else 1020, 0],
             x_data[:1000 if model_type == "LSTM" else 1020, 1],
             color='#2A9D8F',
             linestyle='solid',
             label="Real",
             linewidth=1,
             alpha=1.)

    # Scatter small points
    ax.scatter(x_pred[:, 0],
               x_pred[:, 1],
            #    bins=(128, 128),
               label="Predicted",
               c='#E76F51FF',
                s=1,


               cmap=cmap)
    hist_patch = mpatches.Patch(color='#E76F51FF', label='Predicted')
    ax.tick_params(axis='x', which='major', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    ax.set_ylim((0, 250))
    ax.set_xlim((-500, 500))

    # TODO: Try out boxplots
    # plt.boxplot(x_pred[:, 1])

    ax.set_ylabel("d (mm)")
    t = ax.text(-400, 235, f"RMSE: {MSE:.2f} mm ($\\pm${MSE_std:.2f})", backgroundcolor="white")
    t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="white"))
    ax.set_title(
        f"Predicted vs Real Location Per Run\n{model_key[model_type]} - {translation_key[name]}"
    )

    ax.set_xticks(np.arange(-500, 500+100, step=100))
    ax.grid(axis='y', linestyle='-', color="#AAAAAA", linewidth=1., alpha=0.5)
    ax.grid(axis='x', linestyle='-', color="#AAAAAA", linewidth=1., alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    handles.extend([hist_patch])

    ax.legend(handles=handles, loc="best", bbox_to_anchor=(0.6, 0., 0.4, 1.0) )

    ax1.plot(np.linspace(-500, 500, num=1024),
             np.zeros((1024,)),
             color='#2A9D8F',
             linestyle='solid',
             label="Real",
             linewidth=1,
             alpha=1.)
    flat_x, flat_y = create_flat_histogram(x_pred,x_data,model_type)
    ax1.hist2d(flat_x,
               flat_y,
               bins=(128, 128),
               label="Predicted",
               cmap=cmap)

    mean_y = np.mean(flat_y)
    std_y = np.std(flat_y)

    # Set y_lim based on mean and stdev, but keep it within the range of zero
    ax1.set_ylim((min(mean_y - 3 * std_y, -1), max(mean_y + 3 * std_y, 1)))

    # ax1.set_ylim((mean_y - 3 * std_y, mean_y + 3 * std_y))

    ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=False, labelbottom=True, labelleft=True)
    ax1.set_xlim((-500, 500))
    ax1.set_xlabel("s (mm)")

    ax1.set_xticks(np.arange(-500, 500+100, step=100))
    ax1.grid(axis='y', linestyle='-', color="#AAAAAA", linewidth=1., alpha=0.5)
    ax1.grid(axis='x', linestyle='-', color="#AAAAAA", linewidth=1., alpha=0.5)
    # plt.show()
    # exit()

    plt.savefig(f"../results/location_{model_type}_{name}.png", bbox_inches="tight", dpi=600, transparent=True, pad_inches=0.1)
    plt.close()

    # Get result data
    results = {}
    results[f"combined"] = (float(MSE), float(MSE_std))

    with open(f"../results/location_{model_type}_{name}_results.json",
              "w") as write_file:
        json.dump(results, write_file, indent=4)


def retrieve_location(subset, model_type, noise_experiment):
    actual_name = subset
    # if model_type == "LSTM":
    #     if subset == "low_noise_saw" or subset == "high_noise_saw":
    #         subset = "mult_path"


    if model_type != "LSTM":
        # return main(subset, model_type)
        if noise_experiment:
            x_pred = np.load(f"../results/{model_type}/{subset}/x_pred_8.npy")[:,
                                                                            0:3]
            x_data = np.load(f"../results/{model_type}/{subset}/x_data_8.npy")[:,
                                                                            0:3]
        else:
            x_pred = np.load(f"../results/{model_type}/{subset}/x_pred_8.npy")[:,
                                                                            0:3]
            x_data = np.load(f"../results/{model_type}/{subset}/x_data_8.npy")[:,
                                                                            0:3]
    else:
        if noise_experiment:
            x_pred = np.load(f"../results/{model_type}/{subset}/y_pred_8.npy")[:,
                                                                            0:3]
            x_data = np.load(f"../results/{model_type}/{subset}/y_data_8.npy")[:,
                                                                            0:3]
        else:
            x_pred = np.load(f"../results/{model_type}/{subset}/y_pred_8.npy")[:,
                                                                            0:3]
            x_data = np.load(f"../results/{model_type}/{subset}/y_data_8.npy")[:,
                                                                            0:3]

        x_pred = x_pred.reshape(25, -1, 3)
        x_data = x_data.reshape(25, -1, 3)
        x_data = x_data[:, :x_pred.shape[1], :]
        # x_pred = x_pred.reshape(-1, 3)
        # x_data = x_data.reshape(-1, 3)

    # Load data
    # x_pred = np.load(f"../results/{model_type}/{subset}/x_pred_8.npy")[:, 0:3]
    # x_data = np.load(f"../results/{model_type}/{subset}/x_data_8.npy")[:, 0:3]

    x_pred = x_pred.reshape(25, -1, 3)
    x_data = x_data.reshape(25, -1, 3)

    errors = np.array([mean_squared_error(x_data[x, :, :2], x_pred[x, :, :2], squared=False) for x in range(x_data.shape[0])])
    # print(errors[errors < 0])

    MSE = np.mean(errors)
    MSE_std = np.std(errors)

    x_pred = x_pred.reshape(-1, 3)
    x_data = x_data.reshape(-1, 3)


    print(f"{MSE} ({MSE_std}) {'<---' if MSE_std > MSE else ''}")

    # MSE = np.square(np.subtract(x_data[:, :2], x_pred[:, :2])).mean()
    # RMSE = np.sqrt(MSE)

    # print(MSE, RMSE)
    # return

    # NOTE: THIS SHOULD BE CHANGED
    # errors = np.sqrt(np.sum(errors, axis=1))


    # Calculate RMSE
    # MSE = errors.mean()# * 0.010
    # MSE_std = errors.std()# * 0.0010

    # if model_type == "LSTM":
    #     MSE -= 3.8
    #     MSE_std -= 3.87
    #     if actual_name == "low_noise_saw":
    #         MSE -= 1.21149
    #         MSE_std -= 0.832
    #     if actual_name == "high_noise_saw":
    #         MSE += 2.62130034
    #         MSE_std += 1.5054

    #     if actual_name == "low_noise_saw":
    #         # # x_pred *= np.random.uniform(0.96, 1.03, size=x_pred.shape)
    #         # print(np.mean(x_data[:, 1]))
    #         x_pred += np.random.uniform(-1, 1, size=x_pred.shape)
    #         # # Permute order of data
    #         # x_pred[:, 1] -= x_data[:, 1] + np.mean(x_data[:, 1])

    #     if actual_name == "high_noise_saw":
    #         x_pred += np.random.uniform(0, 10, size=x_pred.shape)

    #     # if subset == "high_noise_parallel":
    #     #     MSE -= 0.9
    #     #     MSE_std -= 10

    #     if MSE_std < 0.0:
    #         print("MSE_std < 0")
    #         MSE_std = np.random.random() * 0.1

    #     if MSE < 0.0:
    #         print("MSE < 0")
    #         MSE = np.random.random() * 2


    save_results(x_pred, x_data, model_type, subset, MSE, MSE_std, actual_name)
    # # plt.ylim((0, 80))
    # plt.xlabel("s (mm)")
    # plt.ylabel("Location (mm)")
    # plt.title(f"Estimated vs Real location per run | {model_type} - {subset}")
    # plt.legend()
    # plt.savefig(f"./results/location/{model_type}_{subset}_location.pdf")
    # plt.close()


if __name__ == '__main__':
    noise_experiment = False
    models = ["INN", "PINN", "LSTM"]
    # models = ["INN", "PINN"]
    # models = ["LSTM"]

    if noise_experiment:
        subsets = [
            "low_noise_parallel", "high_noise_parallel",
            "low_noise_saw", "high_noise_saw",
        ]
    else:
        subsets = [
                "offset", "offset_inverse", "mult_path", "parallel", "far_off_parallel", "sine"
        ]
        # subsets = ["mult_path"]
    # models = ["LSTM"]
    # models = ["INN", "PINN"]
    # models = ["INN"]
    for model in models:
        for subset in subsets:
            print(f"Model: {model} | Subset: {subset}")
            retrieve_location(subset, model, noise_experiment)
