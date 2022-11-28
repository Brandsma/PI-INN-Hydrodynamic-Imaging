if __name__=="__main__":
    import sys
    sys.path.append("..")

import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

from lib import LSTM, params
from lib.peregrine_util import get_scratch_dir

from lib.logger import setup_logger
from matplotlib import pyplot as plt
import tensorflow as tf

log = setup_logger(__name__)


# TODO: remove read_inputs and replace it with CLI arguments
def read_inputs():
    n_nodes = 128
    n_epochs = 8
    window_size = 16
    stride = 2
    alpha = 0.05
    decay = 1e-9
    shuffle_data = True
    data_split = 0.8
    dropout = 0
    train_loc = "../data/simulation_data/combined.npy"
    ac_fun = "relu"
    return n_nodes, n_epochs, window_size, stride, \
        alpha, decay, shuffle_data, data_split, dropout, train_loc, ac_fun


if __name__ == "__main__":

    (n_nodes, n_epochs, window_size, stride, alpha, decay, \
     shuffle_data, data_split, dropout_ratio, train_location, ac_fun) =  \
        read_inputs()

    # Load settings
    settings = params.Settings(window_size, stride, n_nodes, \
                               alpha, decay, n_epochs, shuffle_data, data_split, dropout_ratio, \
               train_location, ac_fun)
    # Load data
    data = params.Data(settings, train_location)

    # data.normalize()

    # Recreate data into new form
    x_train = np.concatenate((data.train_data, data.train_labels), axis=2).reshape((-1, data.train_data.shape[2] + data.train_labels.shape[2]))
    y_train = np.repeat(data.train_volumes.reshape((-1,1)), 1024, axis=1).reshape((-1, 1))
    #train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    x_val = np.concatenate((data.val_data, data.val_labels), axis=2).reshape((-1, data.val_data.shape[2] + data.val_labels.shape[2]))
    y_val = np.repeat(data.val_volumes.reshape((-1,1)), 1024, axis=1).reshape((-1, 1))

    x_test = np.concatenate((data.test_data, data.test_labels), axis=2).reshape((-1, data.test_data.shape[2] + data.test_labels.shape[2]))
    y_test = np.repeat(data.test_volumes.reshape((-1,1)), 1024, axis=1).reshape((-1, 1))
    
    print(f"{x_train.shape=} {y_train.shape=}")

    batch_size = 16

    # Initiate the LSTM network using data and settings
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(data.train_data.shape[2] + data.train_labels.shape[2])))
    model.add(tf.keras.layers.Dense(32, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(32, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(32, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(1))
    
    print(model.output_shape)
    model.summary()
    
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.LogCosh(), metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.losses.LogCosh()])

    # Train the network
    model.fit(x_train, y_train, batch_size=batch_size, epochs=4, validation_data=(x_val, y_val))
    
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("test loss, test acc:", results)


    # test_results = model.predict(x_test)
    # print(test_results.shape)
    # print(y_test.shape)

    # plt.plot(test_results, label="a_pred")
    # plt.plot(y_test, label="y_label")
    # plt.legend(loc='upper left')
    # plt.savefig('./volume_nn.png')
    # plt.clf()
    # plt.cla()
    # plt.close()
    
    volume_error = {}
    for run_idx in tqdm(range(data.test_data.shape[0])):
        a = data.test_volumes[run_idx]
        a = data.test_volumes[run_idx]
        if a not in volume_error:
            volume_error[a] = []

        x_test = np.concatenate((data.test_data[run_idx], data.test_labels[run_idx]), axis=1).reshape(
            (-1, data.test_data.shape[2] + data.test_labels.shape[2]))
        
        test_result = model.predict(x_test, verbose=0)
        volume = np.mean(test_result)
        print(a, volume, np.median(test_result))

        volume_error[a].append(abs(volume - a))

    volumes = []
    real_volumes = []
    for key in volume_error:
        volumes.extend([x + key for x in volume_error[key]])
        for _ in volume_error[key]:
            real_volumes.append(key)

    print(volumes, real_volumes)

    plt.plot(volumes, "bo", label="Predicted Volume")
    plt.plot(real_volumes, "r.", label="Real Volume")

    for idx in range(len(volumes)):
        line_x_values = [idx, idx]
        line_y_values = [volumes[idx], real_volumes[idx]]
        plt.plot(line_x_values, line_y_values, "k-")
    plt.ylim((0, 70))
    plt.xlabel("run")
    plt.ylabel("Volume (mm)")
    MSE = np.square(np.subtract(real_volumes, volumes)).mean()
    plt.text(0, 60, f"MSE: {MSE:.2f} mm")
    plt.title(f"Estimated vs Real volume per run")
    plt.legend(loc = 'lower right')
    plt.show()

    for key in volume_error:
        print(f"{key}: {np.mean(volume_error[key])} ({np.std(volume_error[key])})")

    # Save the network for later use
    # trained_models_folder = "../data/trained_models/"
    # Path(trained_models_folder).mkdir(parents=True, exist_ok=True)
    # log.info(f"Saving trained model to {trained_models_folder}{settings.name}...")
    # network.save_model(f"{trained_models_folder}{settings.name}")
