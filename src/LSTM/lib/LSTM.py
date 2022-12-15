# Required libraries
import datetime as dt
import os
import sys

import matplotlib as mpl
import numpy as np
from scipy.spatial.distance import euclidean
from tensorflow.keras import backend as K
from tensorflow.keras import losses, optimizers
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

"""
LSTM_network class
- must be initiated with data and settings objects.
- has 3 public members:
   + train() trains the network
   + test() tests the network and outputs network performance
   + save_results() saves all results found during testing, along
   +    with the training (and - if applicable - validation) loss.
"""


class LSTM_network:
    """
    LSTM_network constructor
    - saves data, settings, start time and settings as data
      members of the LSTM class, and initializes the network
    """
    def __init__(self, data, settings):
        self.data = data
        self.settings = settings
        self.activation = settings.ac_fun
        self.raw_time = dt.datetime.now()
        self.init_time = self.raw_time.strftime("%Y_%m_%d_%X")
        self.model = self.__init_network()
        self.batch_size = 32

    """
    LSTM_network::init_network()
    - 'private' function
    - sets the network architecture based on the current settings
    - returns the resulting model.
    """

    def __init_network(self):
        # Read from settings and data (for readability)
        n_inputs = self.data.n_inputs
        n_outputs = self.data.n_outputs
        n_runs = self.data.n_datapoints
        n_nodes = self.settings.n_nodes
        dropout = self.settings.dropout_ratio
        alpha = self.settings.alpha
        decay = self.settings.decay
        win_size = self.settings.window_size
        stride = self.settings.stride

        # We use a sequential model..
        model = Sequential()
        # with an LSTM layer
        model.add(
            LSTM(n_nodes,
                 input_shape=(win_size, n_inputs),
                 activation=self.activation))
        # and a dropout layer (which only does something if dropout > 0).
        model.add(Dropout(dropout))
        # Finally we have a fully connected layer with 2 to 3 nodes - the x and y positions and an angle
        model.add(Dense(n_outputs, activation='linear'))
        # Compile the model with euclidean error and adam
        optimizer = optimizers.Adam(learning_rate=alpha,
                                       epsilon=None,
                                       decay=decay,
                                       clipnorm=1.)
        # model.compile(loss=losses.MeanSquaredError(), optimizer=optimizer)
        model.compile(loss=self.__euclidean_error_loss, optimizer=optimizer)
        # Return the resulting model
        return model


    def __euclidean_error_loss(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1))

    """
    LSTM_network::train()
    - public function
    - trains the network based on settings and data.
    - creates the self.hist data member which contains the training history
    """

    def train(self):
        print("Training network...")
        # Read from settings and data (for readability)
        epochs = self.settings.n_epochs
        window_size = self.settings.window_size
        stride = self.settings.stride
        train_dat = self.data.train_data
        train_lab = self.data.train_labels
        val_dat = self.data.val_data
        val_labels = self.data.val_labels
        
        train_steps = int(self.__num_batches(train_dat))
        val_steps = int(self.__num_batches(val_dat))

        early_stopping_callback = EarlyStopping(patience=3, restore_best_weights=True)

        # Train the model
        self.hist = self.model.fit( \
                                    self.__generator(train_dat, train_lab, window_size, stride), \
                                    steps_per_epoch = train_steps, \
                                    epochs = epochs, \
                                    verbose = 1, \
                                    callbacks=[early_stopping_callback], \
                                    validation_data = self.__generator(val_dat, val_labels, window_size, stride), \
                                    validation_steps = val_steps)

        print("Completed training network.")

    """
    LSTM_network::test()
    - public function
    - tests the network on test data. The network must be trained.
    - outputs an error message if the network has not been trained yet.
    - outputs numerical info and plots of model accuracy
    - automatically plots network prediction vs actual source locations.
    - test() always tests with a stride of 1.
    """
    def test(self, data, labels, dirname=None):
        # TODO: Check that network is trained
        print("Testing network...")
        # TODO: Make this function more efficient

        if dirname is None:
            # Create directory for test results
            dirname = "../results/" + self.init_time + "_" + str(self.settings.n_nodes) + "_" + str(self.settings.n_epochs) \
                + "_" + str(self.settings.window_size) + "_" + str(self.settings.stride) + "_" + str(self.settings.alpha) \
                + "_" + str(self.settings.decay) + "_" + str(self.settings.data_split) \
                + "_" + str(self.settings.dropout_ratio)

        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        # Read from settings and data (for readability)
        win_size = self.settings.window_size

        self.errors = np.zeros((0, 1))
        self.labels = np.zeros((0, self.data.n_outputs))
        self.pred = np.zeros((0, self.data.n_outputs))

        # Test all windows in the test set
        for lab_idx in tqdm(range(0, len(labels))):
            y_pred = np.zeros((0, self.data.n_outputs))
            y_true = np.zeros((0, self.data.n_outputs))
            for idx in range(0, len(data[lab_idx]) - win_size + 1):
                dat = data[lab_idx][idx:idx + win_size]
                if np.shape(dat) != (win_size, self.data.n_inputs):
                    print("ERROR: invalid size: ", np.shape(dat))
                else:
                    dat = np.reshape(dat, (1, win_size, self.data.n_inputs))

                    test_result = self.model.predict(dat, verbose=0)
                    true_label = labels[lab_idx][idx + win_size - 1:idx +
                                                 win_size][0]

                    y_pred = np.vstack((y_pred, test_result))
                    y_true = np.vstack((y_true, true_label))

                    error = euclidean(test_result, true_label)
                    self.errors = np.vstack((self.errors, error))
                    self.labels = np.vstack((self.labels, true_label))
                    self.pred = np.vstack((self.pred, test_result))

            # Automatically make figure every 10th window
            if lab_idx % 10 == 0:
                plt.plot(y_pred)
                plt.plot(y_true)
                plt.legend(['x_pred', 'y_pred', 'theta_pred', 'x_true', 'y_true', 'theta_true'],
                           loc='upper left')
                # plt.ylim(-35, 35)
                plt.savefig(dirname + '/lab_vs_out_' + str(lab_idx) + '.png')
                plt.clf()
                plt.cla()
                plt.close()

                np.savetxt(dirname + "/" + "pred_ " + str(lab_idx) + ".out",
                           y_pred)
                np.savetxt(dirname + "/" + "true_" + str(lab_idx) + ".out",
                           y_true)

        # print errors
        print("\n", np.mean(self.errors), "+/-", np.std(self.errors))

    """
    LSTM_network()::generator(data, labels)
    - private function
    - generator function. Receives a np.array of matrices, extracts one
      sequence each time the generator is called, reshapes the sequence such
      that it can be fed to the network and yields it. Used by fit_generator
      and evaluate_generator in train() and test() respectively.
    """

    def __generator(self, data, labels, window_size, stride):
        while True:
            for i in range(0, len(data)):
                sample = data[i]
                sample_lab = labels[i]
                batch_idx = 0
                x = np.zeros((0, window_size, self.data.n_inputs))
                y = np.zeros((0, self.data.n_outputs))
                for idx in range(0, len(sample) - window_size + 1, stride):
                    while batch_idx > (self.batch_size - 1):
                        yield (x, y)
                        x = np.zeros((0, window_size, self.data.n_inputs))
                        y = np.zeros((0, self.data.n_outputs))
                        batch_idx = 0
                    batch_idx += 1
                    x = np.vstack(
                        (x,
                         np.reshape(sample[idx:idx + window_size],
                                    (1, window_size, self.data.n_inputs))))
                    y = np.vstack((y,
                                   np.reshape(
                                       sample_lab[idx + window_size - 1:idx +
                                                  window_size],
                                       (1, self.data.n_outputs))))
                yield (x, y)

    """
    LSTM_network::__num_batches(data)
    - private function
    - calculates total number of batches in the total dataset (per epoch)
    """

    def __num_batches(self, data):
        win_size = self.settings.window_size
        stride = self.settings.stride
        batch_size = self.batch_size
        tot_batches = 0
        for run in data:
            seq_idx = 0
            batch_idx = 0
            while seq_idx + win_size <= len(run):
                seq_idx += stride
                if batch_idx % batch_size == 0:
                    tot_batches += 1
                batch_idx += 1
        return tot_batches

    """
    LSTM_network::save_model()
    - public function
    - used to save plots and text files of network performance. plots are train and validation loss over training time,
      and plain txt files contain settings and loss values over time.
    """

    def save_model(self, model_location):
        print("\nSaving model...")
        isExist = os.path.exists(model_location)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(model_location)

        self.model.save(model_location)

        # TODO: Load hist at some point
        np.save(model_location + "/loss_history.npy", self.hist)

    """
    LSTM_network::save_results()
    - public function
    - used to save plots and text files of network performance. plots are train and validation loss over training time,
      and plain txt files contain settings and loss values over time.
    """

    def save_results(self, dirname=None):
        print("\nSaving results...")

        if dirname is None:
            # determine directory name based on time of program start and settings
            dirname = "../results/" + self.init_time + "_" + str(self.settings.n_nodes) + "_" + str(self.settings.n_epochs) \
                + "_" + str(self.settings.window_size) + "_" + str(self.settings.stride) + "_" + str(self.settings.alpha) \
                + "_" + str(self.settings.decay) + "_" + str(self.settings.data_split) \
                + "_" + str(self.settings.dropout_ratio)

        # only create the directory if it does not yet exist
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        # calculate how long running the program (training + testing) took
        total_duration = (dt.datetime.now() - self.raw_time)

        # create and save plot of model training loss/validation loss over epochs
        # plt.plot(self.hist.history['loss'])
        # plt.plot(self.hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(dirname + "/" + 'loss.png')

        # create and save the used settings/parameters
        file = open(dirname + "/" + "parameters.txt", "w")
        file.write("Number of Nodes: " + str(self.settings.n_nodes) + '\n')
        file.write("Max epochs: " + str(self.settings.n_epochs) + '\n')
        file.write("Window Size: " + str(self.settings.window_size) + '\n')
        file.write("Stride: " + str(self.settings.stride) + '\n')
        file.write("Learning Rate: " + str(self.settings.alpha) + '\n')
        file.write("Number of Sensors: " + str(self.data.n_inputs) + '\n')
        file.write("Training sequences: " + str(self.data.n_datapoints) + '\n')
        file.write("Decay: " + str(self.settings.decay) + '\n')
        file.write("Dropout Ratio: " + str(self.settings.dropout_ratio) + '\n')
        file.write("Total duration: " + str(total_duration) + '\n')
        file.write("Test error mean: " + str(np.mean(self.errors)) + '\n')
        file.write("Test error std:  " + str(np.std(self.errors)) + '\n')
        file.write("File location: " + self.settings.train_location + '\n')
        file.close()

        # np.savetxt(dirname + "/" + "train_loss.out", self.hist.history['loss'])
        # np.savetxt(dirname + "/" + "val_loss.out",
        #            self.hist.history['val_loss'])
        np.savetxt(dirname + "/" + "errors.out", self.errors)
        np.savetxt(dirname + "/" + "labels.out", self.labels)
        np.savetxt(dirname + "/" + "pred.out", self.pred)

        print("Results saved.")
