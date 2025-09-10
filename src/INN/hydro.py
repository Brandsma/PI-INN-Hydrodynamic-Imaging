import numpy as np
import tensorflow as tf
import os
import sys

from matplotlib import pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import train_test_split

# Add the parent directory to the path to enable absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import params


def add_noise(train_data, train_labels, test_data, test_labels, run, use_pde=False):
    # NOTE: reversing the data and labels here
    if run == -1:
        train_data = np.reshape(train_data, (-1, train_data.shape[2]))
        train_labels = np.reshape(train_labels, (-1, train_labels.shape[2]))
    elif run >= 0:
        train_data = train_data[run]
        train_labels = train_labels[run]

    if run == -1:
        test_data = np.reshape(test_data, (-1, test_data.shape[2]))
        test_labels = np.reshape(test_labels, (-1, test_labels.shape[2]))
    elif run >= 0:
        test_data = test_data[run]
        test_labels = test_labels[run]

    ## Train noise
    train_data_noise = np.random.normal(0, 0.005, train_data.shape)
    train_labels_noise = np.random.normal(0, 0.005, train_labels.shape)

    if use_pde:
        train_data_noise *= 0.05
        train_labels_noise *= 0.05

    train_data = train_data + train_data_noise
    train_labels = train_labels + train_labels_noise

    ## Test noise
    test_data_noise = np.random.normal(0, 0.005, test_data.shape)
    test_labels_noise = np.random.normal(0, 0.005, test_labels.shape)

    if use_pde:
        test_data_noise *= 0.05
        test_labels_noise *= 0.05

    test_data = test_data + test_data_noise
    test_labels = test_labels + test_labels_noise

    return train_data, train_labels, test_data, test_labels


def setup_data_with_data(data, run=-1):
    data.normalize()

    train_labels = data.train_data
    train_data = data.train_labels

    test_labels = data.test_data
    test_data = data.test_labels

    train_data, train_labels, test_data, test_labels = add_noise(
        train_data, train_labels, test_data, test_labels, run
    )

    return train_data, train_labels, test_data, test_labels


def setup_data(
    subset="all",
    shuffle_data=True,
    num_sensors=64,
    run=-1,
    use_pde=False,
    a=0,
    w=0,
    noise_experiment=False,
):
    if subset == "all":
        subset = "combined_groups"

    if a != 0 and w != 0:
        train_location = f"data/simulation_data/{subset}/a{a}_normw{w}_data.npy"
    else:
        if noise_experiment:
            train_location = f"data/simulation_data/noise/{subset}/combined.npy"
        else:
            train_location = f"data/simulation_data/{subset}/combined.npy"

    print(f"Getting training data from {train_location}")

    # Load data
    settings = params.Settings(
        shuffle_data=shuffle_data, train_location=train_location, seed=42
    )

    data = params.Data(settings, train_location)
    data.normalize()

    train_labels = data.train_data
    train_data = data.train_labels

    test_labels = data.test_data
    test_data = data.test_labels

    train_data, train_labels, test_data, test_labels = add_noise(
        train_data, train_labels, test_data, test_labels, run, use_pde
    )

    print("Getting sensors between: ")
    print(
        (train_labels.shape[1] // 2 - num_sensors),
        (train_labels.shape[1] // 2 + num_sensors),
    )
    # Number of sensors
    # train_labels = train_labels[:, 61:67]
    train_labels = train_labels[
        :,
        (train_labels.shape[1] // 2 - num_sensors) : (
            train_labels.shape[1] // 2 + num_sensors
        ),
    ]
    test_labels = test_labels[
        :,
        (test_labels.shape[1] // 2 - num_sensors) : (
            test_labels.shape[1] // 2 + num_sensors
        ),
    ]
    # test_labels = test_labels[:, 61:67]

    return train_data, train_labels, test_data, test_labels


def pde(
    vx_x: tf.Tensor, vy_y: tf.Tensor, x: tf.Tensor, y: tf.Tensor, W: float = 30.0
) -> tf.Tensor:
    """
    Calculates the residual of the physics-informed PDE.

    This function implements a PDE that appears to be a continuity equation
    for an incompressible fluid with a source term. The equation is:

    .. math::
        \\frac{\\partial v_x}{\\partial x} + \\frac{\\partial v_y}{\\partial y} + \\frac{3(Wx + Wy)}{(x^2 + y^2)^{5/2}} = 0

    Args:
        vx_x: The derivative of the x-component of the velocity field with respect to x (dv_x/dx).
        vy_y: The derivative of the y-component of the velocity field with respect to y (dv_y/dy).
        x: The x-coordinates.
        y: The y-coordinates.
        W: A constant weight factor for the source term.

    Returns:
        The residual of the PDE for each input point.

    .. warning::
        The original author of this code included a `TODO` comment expressing
        uncertainty about the correctness of this PDE implementation. It should
        be used with caution and verified against the original research paper
        if possible.
    """
    return vx_x + vy_y + ((3 * (W * x + W * y)) / ((x**2 + y**2) ** (5 / 2)))


def interior_loss(
    model: tf.keras.Model, x_data: tf.Tensor, x_dim: int, y_dim: int
) -> tf.Tensor:
    """
    Calculates the physics-informed loss based on the PDE residual.

    This function computes the Jacobian of the model's output with respect to
    the input data to find the velocity derivatives, then uses the `pde`
    function to calculate the residual. The loss is the mean squared error
    of this residual.

    Args:
        model: The neural network model.
        x_data: The input data to the model.
        x_dim: The dimension of the input space.
        y_dim: The dimension of the output space.

    Returns:
        The mean squared PDE residual loss.
    """
    # The PDE loss is calculated based on the model's output.
    # The original author considered adding this to the forward loss calculation,
    # but it is implemented here as a separate component of the total loss.
    with tf.GradientTape() as tape:
        tape.watch(x_data)
        pde_y_out = model(x_data)

    # Jacobian calculation
    # NOTE: requires independence between batches (so no Batch Normalization can be applied)
    j = tape.batch_jacobian(pde_y_out, x_data)
    vx_x = j[:, :x_dim, :y_dim][:, 0, 0::2]
    vy_y = j[:, :x_dim, :y_dim][:, 1, 1::2]

    pde_loss_output = pde(
        vx_x, vy_y, tf.reshape(x_data[:, 0], (-1, 1)), tf.reshape(x_data[:, 1], (-1, 1))
    )
    # Use the residuals of the PDE (anything other than zero is an error)
    return tf.math.reduce_mean(tf.math.square(pde_loss_output))


def plot_results_from_array(
    x_data, x_pred, subset, num_sensors, title="", savefig=False, savepath="./results"
):
    print(f"{'Saving' if savefig else 'Showing'} figures...")
    if savefig:
        os.makedirs(savepath, exist_ok=True)

    # Determine data to plot
    input_sensors = np.linspace(-200, 200, num=8)
    x = x_pred[:, 0]
    y = x_pred[:, 1]
    label_x = x_data[:1020, 0]
    label_y = x_data[:1020, 1]

    y = np.append(y, [250])
    y = np.append([0], y)

    x = np.append(x, [500])
    x = np.append([-500], x)

    # Plot the predicted hist vs the labels
    plt.hist2d(x, y, bins=(128, 128), label="predicted", cmap=plt.cm.viridis)
    plt.plot(
        label_x,
        label_y,
        color="red",
        linestyle="dashed",
        label="label",
        linewidth=2,
        alpha=0.4,
    )
    # plt.plot(label_x, x_data[:1020, 2], color='green', linestyle='dashed', label="angle", linewidth=2, alpha=0.4)
    # plt.plot(label_x, x_pred[:1020, 2], color='white', linestyle='dashed', label="angle", linewidth=2, alpha=0.4)

    # Plot the sensor array
    plt.plot(input_sensors, [min(y)] * len(input_sensors), "go")
    plt.title(f"{subset} with {num_sensors} sensors")

    # plt.legend()
    # plt.show()
    # exit()

    if savefig:
        plt.savefig(f"{savepath}/hydro_{subset}_{num_sensors}")
        plt.close()
    else:
        plt.show()


def plot_results(
    x_data,
    x_pred,
    y_data,
    y_pred,
    x_dim,
    y_dim,
    run_idx,
    title="",
    savefig=False,
    savepath="./results",
):
    print(f"{'Saving' if savefig else 'Showing'} figures...")

    if savefig:
        os.makedirs(savepath, exist_ok=True)

    plt.figure()
    plt.title(f"Backward Process - ($V_x$, $V_y$ -> x,y,$\\theta$) {title}")
    plt.plot(x_pred[(1024 * run_idx) : ((run_idx + 1) * 1023), 0], label="predicted x")
    plt.plot(x_pred[(1024 * run_idx) : ((run_idx + 1) * 1023), 1], label="predicted y")
    plt.plot(
        x_pred[(1024 * run_idx) : ((run_idx + 1) * 1023), 2],
        label="predicted $\\theta$",
    )
    plt.plot(x_data[(1024 * run_idx) : ((run_idx + 1) * 1023), 0], label="label x")
    plt.plot(x_data[(1024 * run_idx) : ((run_idx + 1) * 1023), 1], label="label y")
    plt.plot(
        x_data[(1024 * run_idx) : ((run_idx + 1) * 1023), 2], label="label $\\theta$"
    )
    plt.xlabel("Measurement")
    plt.ylabel("x | y")
    plt.legend()
    if savefig:
        plt.savefig(f"{savepath}/{title}_backward")
        plt.close()

    plt.figure()
    plt.title(f"Forward Process - (x,y,$\\theta$ -> $V_x$, $V_y$) {title}")
    plt.plot(
        y_pred[(1024 * run_idx) : ((run_idx + 1) * 1024), :y_dim], label="predicted"
    )
    plt.plot(y_data[(1024 * run_idx) : ((run_idx + 1) * 1024), :y_dim], label="label")
    plt.xticks(
        np.linspace(0, 2048, num=8), map(round, np.linspace(-1, 1, num=8), [2] * 8)
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    if savefig:
        plt.savefig(f"{savepath}/{title}_forward")
        plt.close()

    if not savefig:
        plt.show()
