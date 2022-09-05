# Import torch if using backend pytorch
# import torch
import math

import deepxde as dde
import numpy as np
import pandas as pd
import seaborn as sns
# Import tf if using backend tensorflow.compat.v1 or tensorflow
from deepxde.backend import tf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lib.params import Data, Settings
from util import Debugger

# GLOBAL VARIABLES

SENSOR = 32
SAMPLING_RATE = 1024

# Data


def load_gendata():
    vx_data = pd.read_csv('../data/a1_normw1_theta0/simdata_vx.csv')
    vy_data = pd.read_csv('../data/a1_normw1_theta0/simdata_vy.csv')

    return np.vstack((np.ravel(vx_data), np.ravel(vy_data))).T


def gen_testdata(x_range=(1, 21), y_range=(1, 11)):
    # TODO: Load actual data
    x = list(range(x_range[0], x_range[1]))
    y = list(range(y_range[0], y_range[1]))
    xx, yy = np.meshgrid(x, y)
    X = np.vstack((np.ravel(xx), np.ravel(yy))).T
    return X


# Solution
def wavelet_e(p):
    return (1 - 2 * p**2) / ((1 + p**2)**(5 / 2))


def wavelet_o(p):
    return (-3 * p) / ((1 + p**2)**(5 / 2))


def wavelet_n(p):
    return (2 - p**2) / ((1 + p**2)**(5 / 2))


def v_x(s, x, y, theta=0, a=10, norm_w=1):
    p = (s - x) / y
    C = (norm_w * a**3) / (2 * y**3)
    return C * (wavelet_o(p) * math.sin(theta) -
                wavelet_e(p) * math.cos(theta))


def v_y(s, x, y, theta=0, a=10, norm_w=1):
    p = (s - x) / y
    C = (norm_w * a**3) / (2 * y**3)
    return C * (wavelet_n(p) * math.sin(theta) -
                wavelet_o(p) * math.cos(theta))


@Debugger
def solution(x):
    x1, y1 = x[:, 0:1], x[:, 1:2]
    # TODO: Update this solution such that the sensor is also included
    return np.hstack((v_x(SENSOR / SAMPLING_RATE, x1,
                          y1), v_y(SENSOR / SAMPLING_RATE, x1, y1)))


def positional_solution(x):
    vx, vy = x[:, 0:1], x[:, 1:2]


# PDE

W = dde.Variable(19.0)


def pde(x, y):
    dvx_x = dde.grad.jacobian(y, x, i=0, j=0)
    dvy_y = dde.grad.jacobian(y, x, i=1, j=1)

    x1, y1 = x[:, 0:1], x[:, 1:2]
    # vx1, vy1 = y[:, 0:1], y[:, 1:2]
    # TODO: This should use the normalized (i.e. sensor-relative positions)
    return dvx_x + dvy_y + ((3 * (W * x1 + W * y1)) / ((x1**2 + y1**2)**(5 / 2)))

# MAIN


def print_x(x):
    if x[0] == -500:
        print(x)
    print("------------")
    return True


def main():
    Debugger.enabled = True

    settings = Settings(16, 1, 100, 0.05, 1e-09, 8, True, 0.8, 0.0,
                        "../../data/simulation_data/a20_normw20_data.npy", "relu")
    data = Data(settings, settings.train_location)

    # print(data.train_data.shape)
    xmin = [-500, 75]
    xmax = [500, 150]

    # Define the input data
    geom = dde.geometry.Rectangle(xmin=xmin, xmax=xmax)
    bc_vx_1 = dde.DirichletBC(
        geom,
        lambda x: 0,
        lambda x, on_boundary: x[0] == xmax[0],
        component=0,
    )
    bc_vx_2 = dde.DirichletBC(
        geom,
        lambda x: 0,
        lambda x, on_boundary: x[0] == xmin[0],
        component=0,
    )
    bc_vx_3 = dde.DirichletBC(
        geom,
        lambda x: 0,
        lambda x, on_boundary: x[1] == xmax[1],
        component=0,
    )
    bc_vx_4 = dde.DirichletBC(
        geom,
        lambda x: v_x(SENSOR, x[0], x[1], 4.28, 20, 20),
        lambda x, on_boundary: x[1] == xmin[1],
        component=0,
    )
    # -------------
    bc_vy_1 = dde.DirichletBC(
        geom,
        lambda x: 0,
        lambda x, on_boundary: x[0] == xmax[0],
        component=1,
    )
    bc_vy_2 = dde.DirichletBC(
        geom,
        lambda x: 0,
        lambda x, on_boundary: x[0] == xmin[0],
        component=1,
    )
    bc_vy_3 = dde.DirichletBC(
        geom,
        lambda x: 0,
        lambda x, on_boundary: x[1] == xmax[1],
        component=1,
    )
    bc_vy_4 = dde.DirichletBC(
        geom,
        lambda x: v_y(SENSOR, x[0], x[1], 4.28, 20, 20),
        lambda x, on_boundary: x[1] == xmin[1],
        component=1,
    )
    

    # TODO: ADD THE SPEED OR VOLUME OR WHATEVER DIRECTLY TO THE OUTPUT OF THE MODEL INSTEAD OF USING THE INVERSE CALCULATION METHOD

    # observed_data_vx = data.train_data[0, :, 0].reshape(data.train_data[0, :, 0].shape[0], 1)
    # observed_data_vy = data.train_data[0, :, 1].reshape(data.train_data[0, :, 1].shape[0], 1)
    # # print(x_input.shape, observed_data.shape)
    # observe_w_vx = dde.icbc.PointSetBC(coord_input, observed_data_vx, component=0)
    # observe_w_vy = dde.icbc.PointSetBC(coord_input, observed_data_vy, component=1)

    # Define the data together with the PDE
    data = dde.data.PDE(geom,
                        pde,
                        [bc_vx_1, bc_vx_2, bc_vx_3, bc_vx_4,
                            bc_vy_1, bc_vy_2, bc_vy_3, bc_vy_4],
                        num_domain=5000,
                        num_boundary=2000,
                        )

    # Define the neural network
    net = dde.nn.FNN([2] + [30] * 3 + [2], "tanh", "Glorot normal")

    # Create the PINN and train it
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, external_trainable_variables=[W])
    variable = dde.callbacks.VariableValue(
        [W], period=200, filename="variables.dat")
    losshistory, train_state = model.train(
        iterations=10000, callbacks=[variable])

    # Show the results of training
    y_train, y_test, best_y, best_ystd = dde.utils.external._pack_data(
        train_state)
    print(best_y.shape)
    # vx_output = best_y[:, 0].reshape((np.sqrt(best_y.shape[0]).astype(int), np.sqrt(best_y.shape[0]).astype(int)))
    # print(vx_output.shape)
    # print(train_state.X_test.shape)
    # ax = sns.heatmap(vx_output, linewidth=0.5)
    # ax = plt.axes(projection="3d")
    # ax.plot_surface(train_state.X_test[:, 0], train_state.X_test[:, 1], best_y[:, 0], cmap='viridis', edgecolor='none')
    # plt.show()
    plt.figure()
    x_input = np.array(list(np.linspace(-500, 500, num=64))).reshape(-1, 1)
    y_input = np.array(list(np.linspace(75, 150, num=64))).reshape(-1, 1)
    coordv = np.array(np.meshgrid(x_input, y_input))
    coordv = np.transpose(coordv.reshape(2, -1))
    true_vy_data = np.array([v_y(SENSOR, x, y, 4.28, 20, 20) for (x,y) in coordv]).reshape(-1, 1)

    ax = plt.axes(projection=Axes3D.name)
    ax.plot3D(
        coordv[:, 0],
        coordv[:, 1],
        true_vy_data[:, 0],
        ".",
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$V_y$")


    dde.utils.external.saveplot(losshistory, train_state)
    # dde.utils.external.save_best_state(
    #     train_state, "./train_data.txt", "./test_data.txt")


if __name__ == "__main__":
    main()
