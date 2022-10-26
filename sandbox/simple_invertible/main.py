import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from flow import *
from utils import *

from trainer import Trainer

def solution(x):
    return np.sin(np.pi * x)

def main():
    ## SETUP DATA ##
    data = np.linspace(-1, 1, num=1024).reshape((-1, 1))
    labels = np.linspace(1, -1, num=1024).reshape((-1, 1))
    # labels = solution(data).reshape((-1,1))

    data_noise = np.random.normal(0, .005, data.shape)
    labels_noise = np.random.normal(0, .005, labels.shape)

    data = data + data_noise
    labels = labels + labels_noise

    # plt.plot(data, label="data")
    # plt.plot(labels, label="labels")
    # plt.legend()
    # plt.show()
    # exit()

    print(data.shape)
    print(labels.shape)

    x_dim = data.shape[1]
    y_dim = labels.shape[1]
    z_dim = 33
    tot_dim = y_dim + z_dim
    pad_dim = tot_dim - x_dim
    n_data = data.shape[0]
    n_couple_layer = 6
    n_hid_layer = 6
    n_hid_dim = 128

    n_batch = 8
    n_epoch = 4
    n_display = 1

    ###
    # Make data

    # X = data
    # y = labels
    X = labels
    y = data

    ###
    # Preprocess
    # print(X_raw.shape)
    # X = X_raw.reshape((-1, x_dim))
    # print(X.shape)
    # exit()
    # X = StandardScaler().fit_transform(X)

    ###
    # Pad data
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
    model.summary()



    trainer = Trainer(model, x_dim, y_dim, z_dim, tot_dim, n_couple_layer,
                    n_hid_layer, n_hid_dim)
    trainer.compile(optimizer='Adam')

    LossFactor = UpdateLossFactor(n_epoch)
    logger = NBatchLogger(n_display, n_epoch)
    hist = trainer.fit(dataset,
                    batch_size=n_batch,
                    epochs=n_epoch,
                    steps_per_epoch=n_data // n_batch,
                    callbacks=[logger, LossFactor],
                    verbose=1)

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

    # print(x_pred)
    # print(x_pred.shape)
    # print(x_data.shape)

    # print(y_pred.shape)
    # print(y_data.shape)
    # print(labels.shape)

    print("Showing figures...")

    plt.figure()
    plt.title("Backward Process")
    plt.plot(x_pred[:, 0], label="predicted")
    plt.plot(x_data[:, 0], label="label")
    plt.legend()

    plt.figure()
    plt.title("Forward Process")
    plt.plot(y_pred[:, -1], label="predicted")
    plt.plot(y_data[:, -1], label="label")
    plt.legend()

    plt.figure()
    plt.title("Backward Process Distribution")
    plt.plot(x_pred[:, 1:-1])
    plt.show()

    print(" -- DONE -- ")

if __name__ == '__main__':
    main()
