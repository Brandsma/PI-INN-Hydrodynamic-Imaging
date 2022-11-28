import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from .data import get_data, DataType

from .inn import create_model, INNConfig

from . import sine
from . import hydro

from .flow import *
from .utils import *
from .trainer import Trainer

def run_inn(given_data,
            given_labels,
            config: INNConfig,
            n_batch = 8,
            n_epoch = 32,
            datatype=DataType.Hydro):

    # plt.plot(given_data, label="given_data")
    # plt.plot(given_labels, label="given_labels")
    # plt.legend()
    # plt.show()
    # exit()

    # print(given_data.shape)
    # print(given_labels.shape)

    n_couple_layer = config.n_couple_layer
    n_hid_layer = config.n_hid_layer
    n_hid_dim = config.n_hid_dim
    x_dim = config.x_dim
    y_dim = config.y_dim
    z_dim = config.z_dim

    tot_dim = y_dim + z_dim
    pad_dim = tot_dim - x_dim
    n_data = given_data.shape[0]

    ###
    # Make given_data

    X = given_data
    y = given_labels

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
    pad_x = np.random.multivariate_normal([0.] * pad_dim, np.eye(pad_dim),
                                          X.shape[0])
    x_data = np.concatenate([X, pad_x], axis=-1).astype('float32')
    # TODO: This z should be a gaussian (I think based on the paper), which it is right now.
    # But do check if this is correct in the future
    z = np.random.multivariate_normal([0.] * z_dim, np.eye(z_dim), y.shape[0])
    y_data = np.concatenate([y, z], axis=-1).astype('float32')

    # Make dataset generator
    x_data = tf.data.Dataset.from_tensor_slices(x_data)
    y_data = tf.data.Dataset.from_tensor_slices(y_data)
    dataset = (tf.data.Dataset.zip(
        (x_data, y_data)).shuffle(buffer_size=X.shape[0]).batch(
            n_batch, drop_remainder=True).repeat())

    ## INITIALIZE MODEL ##
    model = create_model(tot_dim, n_couple_layer, n_hid_layer, n_hid_dim)
    # model = NVP(tot_dim, n_couple_layer, n_hid_layer, n_hid_dim, name='NVP')
    # x = tfk.Input((tot_dim, ))
    # model(x)
    # model.summary()

    # Determine PDE loss func
    pde_loss_func = config.pde_loss_func
    # if datatype == DataType.Sine:
    #     pde_loss_func = sine.interior_loss
    # elif datatype == DataType.Hydro:
    #     pde_loss_func = hydro.interior_loss

    trainer = Trainer(model, x_dim, y_dim, z_dim, tot_dim, n_couple_layer,
                      n_hid_layer, n_hid_dim, pde_loss_func=pde_loss_func, pde_applied_forward=True)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=0.5))

    LossFactor = UpdateLossFactor(n_epoch)
    # logger = NBatchLogger(n_display, n_epoch)
    hist = trainer.fit(dataset,
                       batch_size=n_batch,
                       epochs=n_epoch,
                       steps_per_epoch=n_data // n_batch,
                       callbacks=[LossFactor],
                       verbose=1)

    ## CHECK RESULTS ##

    fig, ax = plt.subplots(1, facecolor='white', figsize=(8, 5))
    ax.plot(hist.history['total_loss'], 'k.-', label='total loss')
    ax.plot(hist.history['forward_loss'], 'b.-', label='forward loss')
    ax.plot(hist.history['latent_loss'], 'g.-', label='latent loss')
    ax.plot(hist.history['rev_loss'], 'r.-', label='inverse loss')
    if pde_loss_func != None:
        ax.plot(hist.history['pde_loss'], 'y.-', label='pde loss')
    plt.legend()
    plt.savefig("../../data/trained_models/INN/latest/trained_model_train_hist")

    # z = np.random.multivariate_normal([1.] * z_dim, np.eye(z_dim), y.shape[0])
    # y_data = np.concatenate([y, z], axis=-1).astype('float32')
    # x_pred = model.inverse(y_data).numpy()
    # x_data = np.concatenate([X, pad_x], axis=-1).astype('float32')
    # y_pred = model(x_data).numpy()

    print(" -- Saving model")
    # model.save("./models/")
    config.to_file("../../data/trained_models/INN/latest/INNConfig.pkl")
    model.save_weights("../../data/trained_models/INN/latest/trained_model_weights.tf")

    # if datatype == DataType.Sine:
    #     sine.plot_results(x_data, x_pred, y_data, y_pred, title="Sine")
    # elif datatype == DataType.Hydro:
    #     hydro.plot_results(x_data, x_pred, y_data, y_pred, x_dim, y_dim, title="Hydro")
    #     # pde_loss_func = hydro.interior_loss


    # plt.figure()
    # plt.title("Backward Process Distribution")
    # plt.plot(x_pred[:, 1:-1])
    # plt.show()

    print(" -- DONE -- ")


# def gridsearch_nn_architecture(dt):
#     data, labels = get_data(dt)

#     ## Gridsearch
#     layer_list = [2, 4, 6]
#     coupling_list = [2, 4, 6]
#     hidden_dim_list = [16, 32, 64, 128]
#     counter = 0
#     total_iters = len(layer_list) * len(coupling_list) * len(hidden_dim_list)
#     for layer_num in layer_list:
#         for coupling_num in coupling_list:
#             for hidden_dim_num in hidden_dim_list:
#                 print(
#                     f"\n\n------------------\n ---  Running # {counter+1}/{total_iters}... (coupling {coupling_num} layer {layer_num} dim {hidden_dim_num}) \n------------------\n\n"
#                 )
#                 run_inn(data,
#                         labels,
#                         coupling_num,
#                         layer_num,
#                         hidden_dim_num,
#                         z_dim=33)
#                 counter += 1


# def gridsearch_z_dim(dt):
#     data, labels = get_data(dt)

#     ## Gridsearch
#     z_list = list(range(1, 256, 8))
#     counter = 0
#     total_iters = len(z_list)
#     for z in z_list:
#         print(
#             f"\n\n------------------\n ---  Running # {counter+1}/{total_iters}... (z {z}) \n------------------\n\n"
#         )
#         run_inn(data, labels, 4, 4, 128, z_dim=z)
#         counter += 1


# def simple_run_repeated(dt):
#     data, labels = get_data(dt)

#     for run_idx in range(5):
#         run_inn(data, labels, 4, 4, 128, z_dim=32, run_idx=run_idx)

def simple_run(dt, use_pde=False):
    data, labels, test_d, test_l = get_data(dt, subset="offset", shuffle_data=False)

    print(data.shape)
    print(labels.shape)

    data = np.concatenate([data, test_d], axis=0).astype('float32')
    labels = np.concatenate([labels, test_l], axis=0).astype('float32')

    print(f"{data.shape=}")
    print(f"{labels.shape=}")

    pde_loss_func = None
    if use_pde:
        pde_loss_func = hydro.interior_loss

    config = INNConfig(4, 4, 128, data.shape[1], labels.shape[1], 32 + (1 if ((32 + labels.shape[1]) % 2 == 1) else 0), pde_loss_func)
    print("Running with config:", config)

    run_inn(data, labels, config, n_batch=8, n_epoch=16, datatype=dt)


if __name__ == '__main__':
    # gridsearch_z_dim()
    # gridsearch_nn_architecture()
    simple_run(DataType.Hydro, use_pde=True)
    # simple_run_repeated()
