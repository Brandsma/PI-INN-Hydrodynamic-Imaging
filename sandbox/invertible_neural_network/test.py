from tensorflow import keras

import itertools

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.spatial.distance import euclidean

from flow import *
from utils import *

def test(model, data, labels, z_dim, dirname=None, n_outputs=3, n_inputs=128):
    print("Testing network...")
    
    if dirname is None:
        # Create directory for test results
        dirname = "./results/"
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    errors = np.zeros((0, 1))
    # labels = np.zeros((0, n_outputs))
    # pred = np.zeros((0, n_outputs))

    # Test all windows in the test set
    for lab_idx in tqdm(range(0, len(labels))):
        y_pred = np.zeros((0, n_outputs))
        y_true = np.zeros((0, n_outputs))
        for idx in range(0, len(data[lab_idx])):
            dat = data[lab_idx][idx]
            if np.shape(dat) != (n_inputs - z_dim,):
                print("ERROR: invalid size: ", np.shape(dat))
            else:

                # test_result = model.predict(dat, verbose=0)

                z = np.random.multivariate_normal([0.] * z_dim, np.eye(z_dim), 1)
                z = np.reshape(z, (z.shape[1],))
                y = np.concatenate([z, dat], axis=-1).astype('float32')
                # dat = np.reshape(y, (1, win_size, n_inputs))
                
                test_result = model.inverse(y).numpy()[:, 0:3]
                true_label = labels[lab_idx][idx][0:3]

                y_pred = np.vstack((y_pred, test_result))
                y_true = np.vstack((y_true, true_label))
                
                error = euclidean(np.reshape(test_result, (test_result.shape[1],)), true_label)
                errors = np.vstack((errors, error))
                # labels = np.vstack((labels, true_label))
                # pred = np.vstack((pred, test_result))

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
    print("\n", np.mean(errors), "+/-", np.std(errors))



def main(a, w):

    ## SETUP DATA ##
    data = np.load(f'../../data/simulation_data/a{a}_normw{w}_data.npy')
    X_raw = np.load(f'../../data/simulation_data/a{a}_normw{w}_data_labels.npy')

    # labels = ['red','red','red','red','blue','blue','green','purple']

    x_dim = X_raw.shape[2]
    y_dim = data.shape[2]
    z_dim = 500
    tot_dim = y_dim + z_dim
    pad_dim = tot_dim - x_dim
    n_data = data.shape[0] * data.shape[1]
    n_couple_layer = 3
    n_hid_layer = 3
    n_hid_dim = 512

    # print("dims", x_dim, y_dim, z_dim)

    n_batch = 128
    n_epoch = 10
    n_display = 1

    ###
    # Make data

    # X_raw = np.zeros((data.shape[0], data.shape[1], x_dim), dtype='float32')
    # for y in range(data.shape[0]):
    #     for x in range(data.shape[1]):
    #         X_raw[y, x, :] = np.array([y + 1, x + 1])

    # TODO: Duplicate the data to have some more training data for now?

    ###
    # Preprocess
    X = X_raw.reshape((-1, x_dim))
    #X = StandardScaler().fit_transform(X)

    y = data.reshape((-1, data.shape[2]))

    ###
    # Pad data
    pad_x = np.zeros((X.shape[0], pad_dim))
    x_data = np.concatenate([X, pad_x], axis=-1).astype('float32')
    # TODO: This z should be a gaussian (I think based on the paper), which it is right now.
    # But do check if this is correct in the future
    z = np.random.multivariate_normal([0.] * z_dim, np.eye(z_dim), y.shape[0])
    y_data = np.concatenate([z, y], axis=-1).astype('float32')

    # Make dataset generator
    x_data = tf.data.Dataset.from_tensor_slices(x_data)
    y_data = tf.data.Dataset.from_tensor_slices(y_data)
    dataset = (tf.data.Dataset.zip(
        (x_data, y_data)).shuffle(buffer_size=X.shape[0]).batch(
            n_batch, drop_remainder=True).repeat())


    model = NVP(tot_dim, n_couple_layer, n_hid_layer, n_hid_dim, name='NVP')
    x = tfk.Input((tot_dim, ))
    model(x)
    # model.summary()

    model.compile()


    loaded_model = keras.models.load_model('./models')
    model.set_weights(loaded_model.get_weights())

    # z = np.random.multivariate_normal([0.] * z_dim, np.eye(z_dim), y.shape[0])
    # y = np.concatenate([z, y], axis=-1).astype('float32')
    # x_pred = model.inverse(y).numpy()
    # print(y.shape)
    # print(x_pred.shape)
    # exit()

    # print(pad_dim)
    # for idx in range(200):
    #     print(x_pred[idx][0:3], " - ", X[idx])

    # TODO: Test it better
    #print(np.linalg.norm(x_pred[:][0:3] - X))

    test(model, data, X_raw, z_dim, dirname=f"./results/a{a}_w{w}/", n_inputs=tot_dim, n_outputs=x_dim)

if __name__=="__main__":
    a_set = [10,20,30,40,50]
    w_set = [10,20,30,40,50]
    count = 0
    for a in a_set:
        for w in w_set:
            count += 1
            print(f"Running test {count}/{len(a_set)*len(w_set)}")
            main(a,w)
