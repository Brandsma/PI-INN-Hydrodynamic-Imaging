import pandas as pd
from matplotlib import pyplot as plt


def plot_vp(data, sensors=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], fname="./graph.png"):
    ax = plt.gca()
    for sensor in sensors:
        ax.plot(data.iloc[:, sensor] + sensor)
    plt.savefig(fname)


def main():
    vx_data = pd.read_csv('./data/a1_normw1_theta0/simdata_vx.csv')
    vy_data = pd.read_csv('./data/a1_normw1_theta0/simdata_vy.csv')
    vx_gradient = pd.read_csv('./data/a1_normw1_theta0/simgradient_vx.csv')
    vy_gradient = pd.read_csv('./data/a1_normw1_theta0/simgradient_vy.csv')

    plt.figure()
    plot_vp(vx_gradient + vy_gradient, fname='./vxvy_gradient_graph.png')


if __name__ == '__main__':
    main()
