import numpy as np
import tensorflow as tf
import os

from matplotlib import pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import train_test_split

from lib import params

def add_noise(train_data, train_labels, test_data, test_labels, run):
    # NOTE: reversing the data and labels here
    if run == -1:
        train_data = np.reshape(train_data[0:32], (-1, train_data.shape[2]))
        train_labels = np.reshape(train_labels[0:32], (-1, train_labels.shape[2]))
    elif run >= 0:
        train_data = train_data[run]
        train_labels = train_labels[run]



    if run == -1:
        test_data = np.reshape(test_data[0:32], (-1, test_data.shape[2]))
        test_labels = np.reshape(test_labels[0:32], (-1, test_labels.shape[2]))
    elif run >= 0:
        test_data = test_data[run]
        test_labels = test_labels[run]


    ## Train noise
    train_data_noise = np.random.normal(0, .005, train_data.shape)
    train_labels_noise = np.random.normal(0, .005, train_labels.shape)

    train_data = train_data + train_data_noise
    train_labels = train_labels + train_labels_noise



    # Number of sensors
    train_labels = train_labels[:, 61:67]




    ## Test noise
    test_data_noise = np.random.normal(0, .005, test_data.shape)
    test_labels_noise = np.random.normal(0, .005, test_labels.shape)

    test_data = test_data + test_data_noise
    test_labels = test_labels + test_labels_noise

    # Number of sensors
    test_labels = test_labels[:, 61:67]

    return train_data, train_labels, test_data, test_labels

def setup_data_with_data(data, run=-1):
    data.normalize()

    train_labels = data.train_data
    train_data = data.train_labels

    test_labels = data.test_data
    test_data = data.test_labels

    train_data, train_labels, test_data, test_labels = add_noise(train_data, train_labels, test_data, test_labels, run)

    return train_data, train_labels, test_data, test_labels

def setup_data(subset="all", shuffle_data=True, run=-1, a=0, w=0):
    if subset == "all":
        subset = 'combined_groups'

    if a != 0 and w != 0:
        train_location = f"../data/simulation_data/{subset}/a{a}_normw{w}_data.npy"
    else:
        train_location = f"../data/simulation_data/{subset}/combined.npy"


    # Load data
    settings = params.Settings(shuffle_data=shuffle_data, train_location=train_location)
    data = params.Data(settings, train_location)
    data.normalize()

    train_labels = data.train_data
    train_data = data.train_labels

    test_labels = data.test_data
    test_data = data.test_labels

    train_data, train_labels, test_data, test_labels = add_noise(train_data, train_labels, test_data, test_labels, run)

    return train_data, train_labels, test_data, test_labels

#TODO Check this
def pde(vx_x, vy_y, x, y, W=30):
    return vx_x + vy_y + ((3 * (W * x + W * y)) / ((x**2 + y**2)**(5 / 2)))

def interior_loss(model, x_data, x_dim, y_dim):
    # Physics informed loss
    # TODO: Possibly add this in the forward loss calculation
    # NOTE: also see https://github.com/deepmorzaria/Physics-Informed-Neural-Network-PINNs---TF-2.0/blob/master/PINNs_2.ipynb
    with tf.GradientTape() as tape:
        tape.watch(x_data)
        pde_y_out = model(x_data)

    # Jacobian calculation
    # NOTE: requires independence between batches (so no Batch Normalization can be applied)
    j = tape.batch_jacobian(pde_y_out, x_data)
    vx_x = j[:, :x_dim, :y_dim][:, 0, 0::2]
    vy_y = j[:, :x_dim, :y_dim][:, 1, 1::2]

    # print(f"{vx_x.shape=} {vy_y.shape=}")
    # print(f"{x_data.shape=}")
    # print(f"{tf.reshape(x_data[:, 0], (-1,1)).shape=}")
    # exit()

    # TODO: Check this PDE function, it might make no sense at all
    pde_loss_output = pde(vx_x, vy_y, tf.reshape(x_data[:, 0], (-1,1)), tf.reshape(x_data[:, 1], (-1,1)))
    # Use the residuals of the PDE (anything other than zero is an error)
    return tf.math.reduce_mean(tf.math.square(pde_loss_output))


def plot_results_from_array(x_data, x_pred, y_data, y_pred, x_dim, y_dim, test_idx=0, title="", savefig=False, savepath="./results"):
    print(f"{'Saving' if savefig else 'Showing'} figures...")
    if savefig:
        os.makedirs(savepath, exist_ok=True)


    # Determine data to plot
    input_sensors = np.linspace(-200, 200, num=8)
    x = x_pred[:, 0]
    y = x_pred[:, 1]
    label_x = x_data[:1024, 0]
    label_y = x_data[:1024, 1]

    # Plot the predicted hist vs the labels
    plt.hist2d(x, y, bins=(128, 128), label="predicted", cmap=plt.cm.viridis)
    plt.plot(label_x, label_y, color='orange', linestyle='solid', label="label", linewidth=2)

    # Plot the sensor array
    plt.plot(input_sensors, [min(y)] * len(input_sensors), 'go')

    # plt.legend()
    plt.show()
    exit()



    if savefig:
        plt.savefig(f"{savepath}/{title}_forward")
        plt.close()

    if not savefig:
        plt.show()

def plot_results(x_data, x_pred, y_data, y_pred, x_dim, y_dim, run_idx, title="", savefig=False, savepath="./results"):
    print(f"{'Saving' if savefig else 'Showing'} figures...")

    if savefig:
        os.makedirs(savepath, exist_ok=True)

    plt.figure()
    plt.title(
        f"Backward Process - ($V_x$, $V_y$ -> x,y,$\\theta$) {title}")
    plt.plot(x_pred[(1024 * run_idx):((run_idx + 1) * 1023), 0], label="predicted x")
    plt.plot(x_pred[(1024 * run_idx):((run_idx + 1) * 1023), 1], label="predicted y")
    plt.plot(x_pred[(1024 * run_idx):((run_idx + 1) * 1023), 2], label="predicted $\\theta$")
    plt.plot(x_data[(1024 * run_idx):((run_idx + 1) * 1023), 0], label="label x")
    plt.plot(x_data[(1024 * run_idx):((run_idx + 1) * 1023), 1], label="label y")
    plt.plot(x_data[(1024 * run_idx):((run_idx + 1) * 1023), 2], label="label $\\theta$")
    plt.xlabel("Measurement")
    plt.ylabel("x | y")
    plt.legend()
    if savefig:
        plt.savefig(f"{savepath}/{title}_backward")
        plt.close()

    plt.figure()
    plt.title(f"Forward Process - (x,y,$\\theta$ -> $V_x$, $V_y$) {title}")
    plt.plot(y_pred[(1024 * run_idx):((run_idx + 1) * 1024), :y_dim], label="predicted")
    plt.plot(y_data[(1024 * run_idx):((run_idx + 1) * 1024), :y_dim], label="label")
    plt.xticks(np.linspace(0, 2048, num=8),
               map(round, np.linspace(-1, 1, num=8), [2] * 8))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    if savefig:
        plt.savefig(f"{savepath}/{title}_forward")
        plt.close()

    if not savefig:
        plt.show()
