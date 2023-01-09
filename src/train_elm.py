import datetime as dt
import math
import os
import sys
from itertools import chain, islice

from lib import params
import numpy as np
import pandas as pd
import scipy.io as sio
import tqdm
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from ELM.os_elm import OS_ELM

# Code adapted from https://github.com/otenim/TensorFlow-OS-ELM.


def mse(A, B):
    return np.linalg.norm(A - B)


# load all data from MATLAB file
def load_data(file_location, train_test_ratio):
    matfile_name = sio.whosmat(file_location)[0][0]
    loaded_file = sio.loadmat(file_location)[matfile_name]

    # Split into labels and data
    # if first name is not label, switch them
    if loaded_file.dtype.names[0].find('lab') == -1:
        data_name, label_name = loaded_file.dtype.names
    else:
        label_name, data_name = loaded_file.dtype.names
    all_data = loaded_file[data_name][0]
    all_labels = loaded_file[label_name][0]

    n_entries = len(all_data)

    # Generate a random permutation of indices
    perm = np.random.permutation(n_entries)

    # Determine indices for train, validation and test sets
    train_idx = int(train_test_ratio * n_entries)

    # Take subsets by using the permutation array as index
    train_data = all_data[perm[0:train_idx]]
    test_data = all_data[perm[train_idx:]]

    train_labels = all_labels[perm[0:train_idx]]
    test_labels = all_labels[perm[train_idx:]]

    return train_data, train_labels, test_data, test_labels


# load all data from CSV file
def load_data_csv(file_location, train_test_ratio):
    # Load data in shape data[sensor][position (1d indexed)]
    print(file_location)
    data = params.Data(params.Settings(), file_location)
    # vx_data = data.train_data[0][:, ::2]
    # vy_data = data.train_data[0][:, 1::2]

    # # TODO: Get width from data instead
    # WIDTH = 20

    # # Format all_data into data[position_idx][sensor][vx or vy], contains velocity profile data
    # # Formats all labels to data[position_idx][x or y], contains positional data
    # all_data = []
    # all_labels = []
    # for v_idx in range(len(vx_data)):
    #     all_data.append(
    #         list(
    #             islice(
    #                 chain.from_iterable(
    #                     zip(vx_data.iloc[v_idx].tolist(),
    #                         vy_data.iloc[v_idx].tolist())), 0, None)))
    #     all_labels.append([(v_idx % WIDTH) + 1, (v_idx // WIDTH) + 1])


    run_idx = 0
    train_data = data.train_data[run_idx][:, :]
    train_labels = data.train_labels[run_idx][:, 0:2]

    test_data = data.test_data[run_idx][:, :]
    test_labels = data.test_labels[run_idx][:, 0:2]

    return train_data, train_labels, test_data, test_labels


# Calculate how many windows are in total dataset.
def tot_windows(data, window_size, stride):
    tot_windows = 0
    for run_idx in range(0, len(data)):
        sample = data[run_idx]
        for sample_idx in range(0, len(sample) - window_size + 1, stride):
            tot_windows += 1
    return tot_windows


# Turn dataset into windows based on stride and window size (def. values 30 and 1)
def turn_into_windows(data, labels, window_size=30, stride=30):
    # X sensors x Y deflections per sensor = X * Y inputs
    n_inputs = len(data[0])
    # x- and y-coordinates = 2 outputs
    n_outputs = len(labels[0])

    n_win = tot_windows(data, window_size, stride)
    x = np.zeros((n_win, window_size, n_inputs))
    y = np.zeros((n_win, n_outputs))
    tot_idx = 0
    for sample_idx in range(0, len(data) - window_size + 1, stride):
        x[tot_idx, :] = np.reshape(data[sample_idx:sample_idx + window_size],
                                   (1, window_size, n_inputs))
        y[tot_idx, :] = np.reshape(
            labels[sample_idx + window_size - 1:sample_idx + window_size],
            (1, n_outputs))
        tot_idx += 1
    return x, y


def euclidean_error_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1))


def build_lstm(n_input_nodes, n_hidden_nodes):
    # We use a sequential model..
    model = Sequential()
    # with an LSTM layer
    # TODO: Fix the input shape in some way: https://wandb.ai/ayush-thakur/dl-question-bank/reports/Keras-Layer-Input-Explanation-With-Code-Samples--VmlldzoyMDIzMDU
    model.add(
        LSTM(n_hidden_nodes, input_shape=n_hidden_nodes, activation='tanh'))
    # and a dropout layer (which only does something if dropout > 0).
    model.add(Dropout(0.05))
    # Finally we have a fully connected layer with 2 to 3 nodes - the x and y positions,
    # and optionally the stimulus diameter (automatically extracted from the dataset).
    model.add(Dense(2, activation='linear'))
    # Compile the model with euclidean error and adagrad
    optimizer = optimizers.Adagrad(lr=0.05, epsilon=None, clipnorm=1.)
    model.compile(loss=euclidean_error_loss, optimizer=optimizer)
    return model


def main(n_nodes, window_size, stride, model_type="ELM"):

    # ===========================================
    # Instantiate os-elm
    # ===========================================

    # load dataset
    x_train, y_train, x_test, y_test = load_data_csv("../data/simulation_data/combined.npy", 0.8)
    n_sensor_readings = len(x_train[0])

    # divide data into windows
    x_train, y_train = turn_into_windows(x_train, y_train, window_size, stride)
    # test set always has stride of 1.
    x_test, y_test = turn_into_windows(x_test, y_test, window_size, stride)

    print("x_train, y_train ", len(x_train), len(y_train))
    print("x_test, y_test ", len(x_test), len(y_test))

    # size of flattened input: sensor readings times the number of
    # samples in each window.
    # n_input_nodes = window_size * n_sensor_readings
    n_input_nodes = n_sensor_readings
    n_hidden_nodes = n_nodes
    n_output_nodes = 2  # x and y position

    # flatten arrays
    x_train = x_train.reshape(-1, n_input_nodes)
    x_test = x_test.reshape(-1, n_input_nodes)

    # cast to floats
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # TODO: Use batching instead of this
    # Divide dataset into initial and sequential parts.
    # We define the smalles initial dataset size as 1.5 times
    # the total number of hidden nodes.
    border = int(1.5 * n_hidden_nodes)
    x_train_init = x_train[:border]
    y_train_init = y_train[:border]

    x_train_seq = x_train[border:]
    y_train_seq = y_train[border:]

    model = None
    if model_type == "ELM":
        model = OS_ELM(
            n_input_nodes=n_input_nodes,
            n_hidden_nodes=n_hidden_nodes,
            n_output_nodes=n_output_nodes,
            loss='mean_squared_error',
        )
    elif model_type == "LSTM":
        model = build_lstm(n_input_nodes, n_hidden_nodes)

    # TODO: Write proper tests for this as well
    assert (
        model != None
    ), "Incorrect model type was given, or model failed to init. Model is None"
    # ===========================================
    # Training
    # ===========================================
    if model_type == "ELM":
        # the initial training phase
        pbar = tqdm.tqdm(total=len(x_train), desc='initial training phase')
        model.init_train(x_train_init, y_train_init)
        pbar.update(n=len(x_train_init))

        # the sequential training phase
        pbar.set_description('sequential training phase')
        batch_size = 64
        for i in range(0, len(x_train_seq), batch_size):
            x_batch = x_train_seq[i:i + batch_size]
            t_batch = y_train_seq[i:i + batch_size]
            model.seq_train(x_batch, t_batch)
            pbar.update(n=len(x_batch))
            pbar.close()
    elif model_type == "LSTM":
        model.fit(x_train, y_train)

    y = model.predict(x_test)
    t = y_test

    # print random 10 results, compared to labels.
    for i in np.random.permutation(len(y))[:10]:
        print('========== sample index %d ==========' % i)
        print('estimated answer: (%.2f, %.2f)' % (y[i][0], y[i][1]))
        print('true answer: (%.2f, %.2f)' % (t[i][0], t[i][1]))
        print('error: %.2f' % mse(y[i], t[i]))

    dists = np.zeros(len(x_test))
    for i in range(len(x_test)):
        dists[i] = mse(y[i], t[i])

    # print errors +/- std.dev.
    print('mean error on all test samples: %.2f' % np.mean(dists))
    print('standard deviation on test    : %.2f' % np.std(dists))

    # save results
    raw_t = dt.datetime.now()
    init_t = raw_t.strftime("%Y_%m_%d_%X")

    dirname = "results/" + init_t + "_" + str(
        n_hidden_nodes) + "_" + "test_dataset" + "_" + str(
            window_size) + "_" + str(stride)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    res = [np.mean(dists), np.std(dists)]
    np.savetxt(dirname + "/" + "errors.out", dists)
    np.savetxt(dirname + "/" + "labels.out", t)
    np.savetxt(dirname + "/" + "pred.out", y)
    np.savetxt(dirname + "/" + "res.out", res)


if __name__ == '__main__':

    # TODO: Make this a proper argparse
    _l = len(sys.argv) == 4

    n_nodes = int(sys.argv[1]) if _l else 1991
    window_size = int(sys.argv[2]) if _l else 10
    stride = int(sys.argv[3]) if _l else 1

    if not _l:
        print("[[WARNING]] INCORRECT NUMBER OF ARGUMENTS SPECIFIED.")
        print(
            "[[WARNING]] PLEASE SPECIFY 3 ARGUMENTS (nodes, window size, stride)."
        )
        print("[[WARNING]] USING DEFAULT ARGUMENTS.")

    print("Starting run with:\n", "nodes:\t\t", n_nodes, "\nwindow size:\t",
          window_size, "\nstride:\t\t", stride)
    main(n_nodes, window_size, stride)
