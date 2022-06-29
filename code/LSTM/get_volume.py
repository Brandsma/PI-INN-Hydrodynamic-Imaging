from scipy.optimize import curve_fit
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.params import Data, Settings
from get_speed_from_location import get_speed_from_data


def read_inputs():
    n_nodes = 100
    n_epochs = 30
    window_size = 16
    stride = 2
    alpha = 0.05
    decay = 1e-9
    shuffle_data = True
    data_split = 0.8
    dropout = 0
    train_loc = "../../data/a10_theta0.npy"
    ac_fun = "relu"
    return n_nodes, n_epochs, window_size, stride, \
        alpha, decay, shuffle_data, data_split, dropout, train_loc, ac_fun


def attempt1():
    x = np.array([1.0,
                  11.0,
                  21.0,
                  31.0,
                  41.0,
                  51.0,
                  61.0,
                  71.0,
                  81.0,
                  91.0,
                  101.0])

    y = np.array([0.0001,
              0.13,
              0.93,
              2.98,
              6.90,
              13.27,
              22.71,
              35.82,
              53.18,
              75.41,
              100.07])

    def func(x, a, b, c):
        # print("b", b, "x", x)
        return a * x + b * x**2 + c

    plt.plot(y,x, 'b.', label="data")

    popt, pcov = curve_fit(func, y, x)
    print(popt)
    plt.plot(y, func(y, *popt), 'r--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    plt.xlabel("a")
    plt.ylabel("S_s")
    plt.legend()
    plt.show()

def extract_volume(speed, vx_data):
    print(speed, vx_data.shape)
    plt.plot(vx_data[:,0])
    plt.show()

    return "Working on it..."

def main():
    (n_nodes, n_epochs, window_size, stride, alpha, decay, \
     shuffle_data, data_split, dropout_ratio, train_location, ac_fun) =  \
        read_inputs()

    # Load settings
    settings = Settings(window_size, stride, n_nodes, \
                        alpha, decay, n_epochs, shuffle_data, data_split, dropout_ratio, \
               train_location, ac_fun)
    # Load data
    data = Data(settings, train_location)

    new_model = tf.keras.models.load_model(
        './trained_models/win16_stride2_epochs30_dropout0_latest')

    run_idx = 0
    speeds = get_speed_from_data(data.test_data[run_idx], data.test_labels[run_idx], data.test_timestamp[run_idx], new_model)
    speed = speeds[0][0]

    vx_data = np.reshape(data.test_data[run_idx], (1024,2,64))[:,0,:]

    print(extract_volume(speed, vx_data))

    print(" -- DONE -- ")



if __name__ == '__main__':
    main()


