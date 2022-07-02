# Import torch if using backend pytorch
# import torch
import math

import deepxde as dde
import numpy as np
import pandas as pd
# Import tf if using backend tensorflow.compat.v1 or tensorflow
from deepxde.backend import tf
from matplotlib import pyplot as plt

from util import Debugger

# GLOBAL VARIABLES

SENSOR = 10000
SAMPLING_RATE = 100

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


def v_x(s, x, y, theta=0, a=1, norm_w=1):
    p = (s - x) / y
    C = (norm_w * a**3) / (2 * y**3)
    return C * (wavelet_o(p) * math.sin(theta) -
                wavelet_e(p) * math.cos(theta))


def v_y(s, x, y, theta=0, a=1, norm_w=1):
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


def additional_term(x, y):
    # TODO: Make this depend on the speed variable instead of hardcoding it
    w_x = 1
    w_y = 1
    return ((3 * (w_x * x + w_y * y)) / ((x**2 + y**2)**(5 / 2)))


def pde(x, y):
    dvx_x = dde.grad.jacobian(y, x, i=0, j=0)
    dvy_y = dde.grad.jacobian(y, x, i=1, j=1)

    x1, y1 = x[:, 0:1], x[:, 1:2]
    vx1, vy1 = y[:, 0:1], y[:, 1:2]
    return dvx_x + dvy_y + additional_term(x1, y1)


# MAIN


def main():
    Debugger.enabled = True

    # Define the input data
    geom = dde.geometry.Rectangle(xmin=[1, 1], xmax=[10, 10])
    # TODO: Find the correct Boundary Conditions for V_xy to XY
    bc = dde.DirichletBC(
        geom,
        lambda x: 0,
        lambda _, on_boundary: on_boundary,
    )

    # Define the data together with the PDE
    data = dde.data.PDE(geom,
                        pde,
                        bc,
                        num_domain=6000,
                        num_boundary=150,
                        solution=solution)
    gen_data = load_gendata()

    # Add the simulated data as training points
    # data.add_anchors(gen_data)
    # data.resample_train_points()

    # Define the neural network
    net = dde.nn.FNN([2] + [32] * 3 + [2], "tanh", "Glorot normal")

    # Create the PINN and train it
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    model.train(epochs=15000)
    model.compile("L-BFGS")
    losshistory, train_state = model.train()

    # Show the results of training
    dde.postprocessing.plot_best_state(train_state)
    plt.show()

    # TEST the model
    # X = gen_testdata()
    # y_true = load_gendata()
    # y_true_single_sensor = list(
    #     zip(y_true[0].iloc[:, SENSOR].to_numpy(),
    #         y_true[1].iloc[:, SENSOR].to_numpy()))

    # y_pred = model.predict(X)
    # f = model.predict(X, operator=pde)

    # print("Mean residual:", np.mean(np.absolute(f)))
    # print("L2 relative error:",
    #       dde.metrics.l2_relative_error(y_true_single_sensor, y_pred))


if __name__ == "__main__":
    main()
