import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from flow import *
from utils import *

from trainer import Trainer

def solution(x):
    return np.sin(np.pi * x)

def run_inn(given_data, given_labels, n_couple_layer = 2, n_hid_layer = 12, n_hid_dim = 128, n_batch = 8, z_dim = 65):

    # plt.plot(given_data, label="given_data")
    # plt.plot(given_labels, label="given_labels")
    # plt.legend()
    # plt.show()
    # exit()

    # print(given_data.shape)
    # print(given_labels.shape)

    # NOTE: Remove this
    if z_dim % 2 == 0:
        print("z_dim should be uneven, but currently is even with", z_dim)

    x_dim = given_data.shape[1]
    y_dim = given_labels.shape[1]
    # z_dim = 65
    tot_dim = y_dim + z_dim
    pad_dim = tot_dim - x_dim
    n_data = given_data.shape[0]
    # n_couple_layer = 4
    # n_hid_layer = 12
    # n_hid_dim = 128

    # n_batch = 8
    n_epoch = 4
    # n_display = 1

    ###
    # Make given_data

    X = given_data
    y = given_labels
    # X = given_labels
    # y = given_data

    ###
    # Preprocess
    # print(X_raw.shape)
    # X = X_raw.reshape((-1, x_dim))
    # print(X.shape)
    # exit()
    # X = StandardScaler().fit_transform(X)

    ###
    # Pad given_data
    pad_x = np.zeros((X.shape[0], pad_dim))
    pad_x = np.random.multivariate_normal([0.] * pad_dim, np.eye(pad_dim), X.shape[0])
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

    ## INITIALIZE MODEL ##
    model = NVP(tot_dim, n_couple_layer, n_hid_layer, n_hid_dim, name='NVP')
    x = tfk.Input((tot_dim, ))
    model(x)
    # model.summary()



    trainer = Trainer(model, x_dim, y_dim, z_dim, tot_dim, n_couple_layer,
                    n_hid_layer, n_hid_dim)
    trainer.compile(optimizer='Adam')

    LossFactor = UpdateLossFactor(n_epoch)
    # logger = NBatchLogger(n_display, n_epoch)
    hist = trainer.fit(dataset,
                    batch_size=n_batch,
                    epochs=n_epoch,
                    steps_per_epoch=n_data // n_batch,
                    callbacks=[LossFactor],
                       verbose=0)

    ## CHECK RESULTS ##

    # fig, ax = plt.subplots(1, facecolor='white', figsize=(8, 5))
    # ax.plot(hist.history['total_loss'], 'k.-', label='total_loss')
    # ax.plot(hist.history['forward_loss'], 'b.-', label='forward_loss')
    # ax.plot(hist.history['latent_loss'], 'g.-', label='latent_loss')
    # ax.plot(hist.history['rev_loss'], 'r.-', label='inverse_loss')
    # plt.legend()

    z = np.random.multivariate_normal([1.] * z_dim, np.eye(z_dim), y.shape[0])
    y_data = np.concatenate([z, y], axis=-1).astype('float32')
    x_pred = model.inverse(y_data).numpy()
    x_data = np.concatenate([X, pad_x], axis=-1).astype('float32')
    y_pred = model(x_data).numpy()

    # print(" -- Saving model")
    # model.save("./models/")
    # model.save_weights("./trained_model_weights.h5")

    print("Showing figures...")

    plt.figure()
    plt.title(f"Backward Process - x = arcsin(y)/$\\pi$ - (z: {z_dim})")
    plt.plot(x_pred[:, 0], label="predicted")
    plt.plot(x_data[:, 0], label="label")
    plt.xticks(np.linspace(0,2048, num=8), map(round, np.linspace(-1,1,num=8), [2] * 8))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    # plt.savefig(f"./results/zdim/backward/backward-couple{n_couple_layer}-layer{n_hid_layer}-dim{n_hid_dim}")
    # plt.savefig(f"./results/zdim2/backward/backward-z{z_dim}")
    # plt.close()

    plt.figure()
    plt.title(f"Forward Process - y = sin($\\pi$ x) - (z={z_dim})")
    plt.plot(y_pred[:, -1], label="predicted")
    plt.plot(y_data[:, -1], label="label")
    plt.xticks(np.linspace(0,2048, num=8), map(round, np.linspace(-1,1,num=8), [2] * 8))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    # plt.savefig(f"./results/zdim/forward/forward-couple{n_couple_layer}-layer{n_hid_layer}-dim{n_hid_dim}-z{z_dim}")
    # plt.savefig(f"./results/zdim2/forward/forward-z{z_dim}")
    # plt.close()

    # plt.figure()
    # plt.title("Backward Process Distribution")
    # plt.plot(x_pred[:, 1:-1])
    plt.show()

    print(" -- DONE -- ")


def gridsearch_nn_architecture():
    # layer_list = [2,4]
    # coupling_list = [2]
    # hidden_dim_list = [16]

    ## SETUP DATA ##
    data = np.linspace(-1, 1, num=2048).reshape((-1, 1))
    # labels = np.linspace(1, -1, num=2048).reshape((-1, 1))
    labels = solution(data).reshape((-1,1))

    data_noise = np.random.normal(0, .005, data.shape)
    labels_noise = np.random.normal(0, .005, labels.shape)

    data = data + data_noise
    labels = labels + labels_noise

    ## Gridsearch
    layer_list = [2,4,6]
    coupling_list = [2,4,6]
    hidden_dim_list = [16, 32, 64, 128]
    counter = 0
    total_iters = len(layer_list) * len(coupling_list) * len(hidden_dim_list)
    for layer_num in layer_list:
        for coupling_num in coupling_list:
            for hidden_dim_num in hidden_dim_list:
                print(f"\n\n------------------\n ---  Running # {counter+1}/{total_iters}... (coupling {coupling_num} layer {layer_num} dim {hidden_dim_num}) \n------------------\n\n")
                run_inn(data, labels, coupling_num, layer_num, hidden_dim_num)
                counter += 1

def gridsearch_z_dim():
    ## SETUP DATA ##
    data = np.linspace(-1, 1, num=2048).reshape((-1, 1))
    # labels = np.linspace(1, -1, num=2048).reshape((-1, 1))
    labels = solution(data).reshape((-1,1))

    data_noise = np.random.normal(0, .005, data.shape)
    labels_noise = np.random.normal(0, .005, labels.shape)

    data = data + data_noise
    labels = labels + labels_noise

    ## Gridsearch
    z_list = list(range(1,256,8))
    counter = 0
    total_iters = len(z_list)
    for z in z_list:
        print(f"\n\n------------------\n ---  Running # {counter+1}/{total_iters}... (z {z}) \n------------------\n\n")
        run_inn(data, labels, 4, 4, 128, z_dim=z)
        counter += 1

def simple_run():
    ## SETUP DATA ##
    data = np.linspace(-1, 1, num=2048).reshape((-1, 1))
    # labels = np.linspace(1, -1, num=2048).reshape((-1, 1))
    labels = solution(data).reshape((-1,1))

    data_noise = np.random.normal(0, .005, data.shape)
    labels_noise = np.random.normal(0, .005, labels.shape)

    data = data + data_noise
    labels = labels + labels_noise

    run_inn(data, labels, 4, 4, 128, z_dim=33)

if __name__ == '__main__':
    simple_run()
