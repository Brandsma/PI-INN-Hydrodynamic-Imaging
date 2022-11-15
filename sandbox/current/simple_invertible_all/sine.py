import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

def solution(x):
    return np.sin(np.pi * x)

def solution_derivative(x):
    return np.cos(np.pi * x)

def setup_data(shuffle_data=True):
    ## SETUP DATA ##
    data = np.linspace(-1, 1, num=2048).reshape((-1, 1))
    # labels = np.linspace(1, -1, num=2048).reshape((-1, 1))
    labels = solution(data).reshape((-1,1))
    polarity = np.array([1] * 1024 + [-1] * 1024).reshape((-1,1))
    delta_labels = solution_derivative(data).reshape((-1,1))
    labels = np.hstack((labels,polarity,delta_labels))
    # labels = np.hstack((labels,delta_labels))

    data_noise = np.random.normal(0, .005, data.shape)
    labels_noise = np.random.normal(0, .005, labels.shape)

    data = data + data_noise
    labels = labels + labels_noise

    labels = data
    data = labels

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, shuffle=shuffle_data, random_state=42)


    return train_data, train_labels, test_data, test_labels

def pde(s_xx, x):
    return -s_xx - np.pi ** 2 * tf.sin(np.pi * x)

def interior_loss(model, x_data, x_dim, y_dim):
    # Physics informed loss
    # TODO: Possibly add this in the forward loss calculation
    # NOTE: also see https://github.com/deepmorzaria/Physics-Informed-Neural-Network-PINNs---TF-2.0/blob/master/PINNs_2.ipynb
    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            t1.watch(x_data)
            t2.watch(x_data)
            pde_y_out = model(x_data)
        grad = t1.gradient(pde_y_out, x_data)

    # Jacobian calculation
    # NOTE: requires independence between batches (so no Batch Normalization can be applied)
    j = t2.batch_jacobian(grad, x_data)
    s_xx = tf.reshape(j[:, :x_dim, :y_dim][:, 0, 0], (-1,1))

    # print(f"{vx_x.shape=} {vy_y.shape=}")
    # print(f"{x_data.shape=}")
    # print(f"{tf.reshape(x_data[:, 0], (-1,1)).shape=}")
    # exit()

    # TODO: Check this PDE function, it might make no sense at all
    pde_loss_output = pde(s_xx, x_data)
    # Use the residuals of the PDE (anything other than zero is an error)
    return tf.math.reduce_mean(tf.math.square(pde_loss_output))

def plot_results(x_data, x_pred, y_data, y_pred, title="IMAGE TITLE"):
    print("Showing figures...")

    plt.figure()
    plt.title(f"Backward Process - {title}")
    plt.plot(x_pred[:, 0], label="predicted")
    plt.plot(x_data[:, 0], label="label")
    plt.xticks(np.linspace(0,2048, num=8), map(round, np.linspace(-1,1,num=8), [2] * 8))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    # plt.savefig(f"./results/gridsearch/backward/backward-couple{n_couple_layer}-layer{n_hid_layer}-dim{n_hid_dim}")
    # plt.savefig(f"./results/zdim2/backward/backward-z{z_dim}")
    # plt.close()

    plt.figure()
    plt.title(f"Forward Process - {title}")
    plt.plot(y_pred[:, -1], label="predicted derivative")
    plt.plot(y_data[:, -1], label="label derivative")
    plt.plot(y_pred[:, -2], label="predicted polarity")
    plt.plot(y_data[:, -2], label="label polarity")
    plt.plot(y_pred[:, -3], label="predicted value")
    plt.plot(y_data[:, -3], label="label value")
    plt.xticks(np.linspace(0,2048, num=8), map(round, np.linspace(-1,1,num=8), [2] * 8))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    # plt.savefig(f"./results/gridsearch/forward/forward-couple{n_couple_layer}-layer{n_hid_layer}-dim{n_hid_dim}")
    # plt.savefig(f"./results/zdim2/forward/forward-z{z_dim}")
    # plt.close()

    # plt.figure()
    # plt.title("Backward Process Distribution")
    # plt.plot(x_pred[:, 1:-1])
    plt.show()
