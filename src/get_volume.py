import math

import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
from scipy import integrate
from scipy.optimize import curve_fit

from get_speed_from_location import get_speed_from_data
from lib.params import Data, Settings


def read_inputs(train_loc):
    n_nodes = 100
    n_epochs = 30
    window_size = 16
    stride = 2
    alpha = 0.05
    decay = 1e-9
    shuffle_data = False
    data_split = 0.8
    dropout = 0
    # train_loc = "../data/simulation_data/combined.npy"
    # train_loc = "../data/simulation_data/a10_normw10_theta0.npy"
    # train_loc = "../data/simulation_data/a20_normw10_theta0.npy"
    # train_loc = "../data/simulation_data/a30_normw10_theta0.npy"
    # train_loc = "../data/simulation_data/a40_normw10_theta0.npy"
    # train_loc = "../data/simulation_data/a50_normw10_theta0.npy"
    # train_loc = "../data/simulation_data/a10_normw10_theta0.npy"
    # train_loc = "../data/simulation_data/a10_normw20_theta0.npy"
    # train_loc = "../data/simulation_data/a10_normw30_theta0.npy"
    # train_loc = "../data/simulation_data/a10_normw40_theta0.npy"
    # train_loc = "../data/simulation_data/a10_normw50_theta0.npy"
    ac_fun = "relu"
    return n_nodes, n_epochs, window_size, stride, \
        alpha, decay, shuffle_data, data_split, dropout, train_loc, ac_fun


def attempt1():
    x = np.array(
        [1.0, 11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0, 81.0, 91.0, 101.0])

    y = np.array([
        0.0001, 0.13, 0.93, 2.98, 6.90, 13.27, 22.71, 35.82, 53.18, 75.41,
        100.07
    ])

    def func(x, a, b, c):
        # print("b", b, "x", x)
        return a * x + b * x**2 + c

    plt.plot(y, x, 'b.', label="data")

    popt, pcov = curve_fit(func, y, x)
    print(popt)
    plt.plot(y,
             func(y, *popt),
             'r--',
             label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    plt.xlabel("a")
    plt.ylabel("S_s")
    plt.legend()
    plt.show()


def wavelet_e(p):
    return (1 - 2 * p**2) / ((1 + p**2)**(5 / 2))


def wavelet_o(p):
    return (-3 * p) / ((1 + p**2)**(5 / 2))


def wavelet_n(p):
    return (2 - p**2) / ((1 + p**2)**(5 / 2))


def inverse_volume_calculation(vx, sensor, speed, x, y, theta):
    p = (sensor - x) / y
    we = wavelet_e(p)

    return ((2 * y**2 * vx) / (speed * we * math.cos(theta)))


def v_x(s, x, y, theta, a, norm_w):
    p = (s - x) / y
    C = (norm_w * a**3) / (2 * y**3)
    return C * (wavelet_o(p) * math.sin(theta) -
                wavelet_e(p) * math.cos(theta))


def extract_volume(speed, vx_data):

    # Simulation Parameters
    sensor_range = (-200, 200)
    number_of_sensors = 64
    input_sensors = list(
        np.linspace(sensor_range[0], sensor_range[1], num=number_of_sensors))

    x_range = (-500, 500)
    number_of_x_steps = 1024
    x_input = list(np.linspace(x_range[0], x_range[1], num=number_of_x_steps))

    volumes = []
    for position in range(vx_data.shape[0]):
        x = x_input[position]
        for sensor_idx in range(vx_data.shape[1]):
            volume = inverse_volume_calculation(vx_data[position, sensor_idx],
                                                input_sensors[sensor_idx],
                                                speed, x, 25, 0)
            volumes.append(volume)

    return np.mean(volumes)
    # print(speed, vx_data.shape)
    # result_to_plot = vx_data[0, :]
    # for idx in range(vx_data.shape[0]):
    #     if idx == 0:
    #         continue
    #     result_to_plot = result_to_plot + vx_data[idx, :]
    # plt.plot(result_to_plot)
    # plt.xlabel("Sensors")
    # # plt.ylim((0,-5))
    # # plt.show()
    # plt.savefig(f"../results/a{a}_w{w}_combined_vxdata_plot.png")

    # Note: Integration Technique
    # integrations = []
    # for vx_idx in range(vx_data.shape[0]):
    #     integrations.append( integrate.simpson(vx_data[vx_idx,:]) / speed)

    # mean_integration = np.array(integrations).mean()
    # print(mean_integration)
    # TODO: Transform this integration to the volume


def start_volume_extraction():
    train_loc = f"../data/simulation_data/combined.npy"
    (n_nodes, n_epochs, window_size, stride, alpha, decay, \
     shuffle_data, data_split, dropout_ratio, train_location, ac_fun) =  \
        read_inputs(train_loc)

    # Load settings
    settings = Settings(window_size, stride, n_nodes, \
                        alpha, decay, n_epochs, shuffle_data, data_split, dropout_ratio, \
               train_location, ac_fun)
    # Load data
    data = Data(settings, train_location)

    # new_model = tf.keras.models.load_model(
    #     '../data/trained_models/old_latest')

    # for run_idx in range(1):
    #     speeds = get_speed_from_data(data.test_data[run_idx], data.test_labels[run_idx], data.test_timestamp[run_idx], new_model)
    #     speed = speeds[0]
    speed = 10

    vx_data = data.test_data[0][:, ::2]

    volume = extract_volume(speed, vx_data)
    print(f"Volume: {volume} mm")

    print('--')
    # print(data.test_labels[run_idx])

    print(" -- DONE -- ")


def main():
    # a_set = [10, 20, 30, 40, 50]
    # w_set = [10, 20, 30, 40, 50]

    # for a in a_set:
    #     plt.figure()
    #     for w in w_set:
    #         print(f"Running a{a} and w{w} now...")
    start_volume_extraction()


if __name__ == '__main__':
    main()
