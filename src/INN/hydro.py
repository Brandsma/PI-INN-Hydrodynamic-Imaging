import numpy as np
import tensorflow as tf
import os

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

def setup_data(subset="all", shuffle_data=True, run=-1):
    loaded_data = None
    loaded_labels = None
    if subset == "all":
        loaded_data = np.load('../data/simulation_data/combined.npy')
        loaded_labels = np.load(
            '../data/simulation_data/combined_labels.npy')
    else:
        loaded_data = np.load(f'../data/simulation_data/{subset}/combined.npy')
        loaded_labels = np.load(
            f'../data/simulation_data/{subset}/combined_labels.npy')


    # NOTE: reversing the data and labels here
    data = loaded_labels
    labels = loaded_data

    if run == -1:
        data = np.reshape(data[0:32], (-1, data.shape[2]))
        labels = np.reshape(labels[0:32], (-1, labels.shape[2]))
    elif run >= 0:
        data = data[run]
        labels = labels[run]


    data_noise = np.random.normal(0, .005, data.shape)
    labels_noise = np.random.normal(0, .005, labels.shape)

    data = data + data_noise
    labels = labels + labels_noise
    labels = labels[:, 63:65]

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, shuffle=shuffle_data, random_state=42)

    # print(train_data.shape)
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

def plot_results(x_data, x_pred, y_data, y_pred, x_dim, y_dim, title="", savefig=False, savepath="./results"):
    print(f"{'Saving' if savefig else 'Showing'} figures...")

    if savefig:
        os.makedirs(savepath, exist_ok=True)

    plt.figure()
    plt.title(
        f"Backward Process - ($V_x$, $V_y$ -> x,y,$\\theta$) {title}")
    plt.plot(x_pred[:, 0], label="predicted x")
    plt.plot(x_pred[:, 1], label="predicted y")
    plt.plot(x_pred[:, 2], label="predicted $\\theta$")
    plt.plot(x_data[:, 0], label="label x")
    plt.plot(x_data[:, 1], label="label y")
    plt.plot(x_data[:, 2], label="label $\\theta$")
    plt.xlabel("Measurement")
    plt.ylabel("x | y")
    plt.legend()
    if savefig:
        plt.savefig(f"{savepath}/{title}_backward")
        plt.close()

    plt.figure()
    plt.title(f"Forward Process - (x,y,$\\theta$ -> $V_x$, $V_y$) {title}")
    plt.plot(y_pred[:, :y_dim], label="predicted")
    plt.plot(y_data[:, :y_dim], label="label")
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
