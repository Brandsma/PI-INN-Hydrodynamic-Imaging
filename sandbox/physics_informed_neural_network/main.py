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
from lib.params import Settings, Data

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


def main():
    Debugger.enabled = True

    settings = Settings(16, 1, 100, 0.05, 1e-09, 5, True, 0.8, 0.0, "../../data/simulation_data/a20_normw20_data.npy", "relu")
    data = Data(settings, settings.train_location)
    x_input = np.array(list(np.linspace(-500, 500, num=1024))).reshape(1024,1)
    y_input = list(np.linspace(75, 75, num=1024))
    coord_input = np.array(list(zip(x_input, y_input)))

    # print(data.train_data.shape)

    # Define the input data
    geom = dde.geometry.Rectangle(xmin=[-500,74.9], xmax=[500, 75.1])
    # TODO: Find the correct Boundary Conditions for V_xy to XY
    bc = dde.DirichletBC(
        geom,
        lambda x: 0,
        lambda _, on_boundary: on_boundary,
    )

    observed_data_vx = data.train_data[0, :, 0].reshape(data.train_data[0, :, 0].shape[0], 1)
    observed_data_vy = data.train_data[0, :, 1].reshape(data.train_data[0, :, 1].shape[0], 1)
    # print(x_input.shape, observed_data.shape)
    observe_w_vx = dde.icbc.PointSetBC(coord_input, observed_data_vx, component=0)
    observe_w_vy = dde.icbc.PointSetBC(coord_input, observed_data_vy, component=1)

    # Define the data together with the PDE
    data = dde.data.PDE(geom,
                        pde,
                        [bc, observe_w_vx, observe_w_vy],
                        num_domain=1024,
                        num_boundary=30,
                        anchors=coord_input)

    # Define the neural network
    net = dde.nn.FNN([2] + [30] * 3 + [2], "tanh", "Glorot normal")

    # Create the PINN and train it
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, external_trainable_variables=[W])
    variable = dde.callbacks.VariableValue([W], period = 600, filename="variables.dat")
    losshistory, train_state = model.train(iterations=60000, callbacks=[variable])

    # Show the results of training
    dde.utils.external.saveplot(losshistory, train_state)
    

if __name__ == "__main__":
    main()
