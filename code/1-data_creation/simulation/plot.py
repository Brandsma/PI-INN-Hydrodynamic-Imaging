import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_vp(data, positions=list(range(1, 32)), fname="./graph.png"):
    ax = plt.gca()
    for position in positions:
        ax.plot(data.iloc[:, position] + position)
    plt.savefig(fname)


def main():
    data = np.load('../../data/a1_normw1_theta0.npy')
    print(data.shape)
    # print("sensor list: ", data[0])
    # print("y list: ", data[0][0])
    print("vx, vy: ", data[0][0][0])
    # vx_data = pd.read_csv('../../data/a1_normw1_theta0/simdata_vx.csv')
    # vy_data = pd.read_csv('../../data/a1_normw1_theta0/simdata_vy.csv')
    # vx_gradient = pd.read_csv('./data/a1_normw1_theta0/simgradient_vx.csv')
    # vy_gradient = pd.read_csv('./data/a1_normw1_theta0/simgradient_vy.csv')
    ax = plt.gca()

    # vx
    plottable_data = [x[5][50] for x in data]
    ax.plot(plottable_data)
    plt.show()

    # plt.figure()
    # plot_vp(vx_data, fname='./vx_graph.png')
    # plt.figure()
    # plot_vp(vy_data, fname='./vy_graph.png')
    # plt.figure()
    # plot_vp(vx_gradient, fname='./vx_gradient_graph.png')
    # plt.figure()
    # plot_vp(vy_gradient, fname='./vy_gradient_graph.png')


if __name__ == '__main__':
    main()
