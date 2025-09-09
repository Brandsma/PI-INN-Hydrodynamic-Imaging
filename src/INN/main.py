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


import os


def run_inn(
    given_data: np.ndarray,
    given_labels: np.ndarray,
    config: INNConfig,
    n_batch: int = 8,
    n_epoch: int = 32,
    datatype: DataType = DataType.Hydro,
    subset: str = "offset",
    num_sensors: int = 8,
            noise_experiment: bool = False,
            model_dir: str = "trained_models",
):
    """
    Runs the training and evaluation of the INN/PINN model.

    Args:
        given_data: The input data for the model.
        given_labels: The labels for the model.
        config: The configuration object for the INN.
        n_batch: The batch size for training.
        n_epoch: The number of epochs for training.
        datatype: The type of data being used (e.g., Hydro or Sine).
        subset: The subset of the data to use.
        num_sensors: The number of sensors used in the data.
        noise_experiment: A flag to indicate if this is a noise experiment.
    """
    n_couple_layer = config.n_couple_layer
    n_hid_layer = config.n_hid_layer
    n_hid_dim = config.n_hid_dim
    x_dim = config.x_dim
    y_dim = config.y_dim
    z_dim = config.z_dim

    tot_dim = y_dim + z_dim
    pad_dim = tot_dim - x_dim
    n_data = given_data.shape[0]

    X = given_data
    y = given_labels

    data_length = X.shape[0] // n_batch

    # Pad the input data with random noise to match the total dimension.
    pad_x = np.random.multivariate_normal([0.0] * pad_dim, np.eye(pad_dim), X.shape[0])
    x_data = np.concatenate([X, pad_x], axis=-1).astype("float32")

    # The latent space `z` is sampled from a standard normal (Gaussian)
    # distribution. This is a standard practice in INNs to ensure a simple and
    # tractable latent distribution, which the model learns to map to the
    # complex data distribution.
    z = np.random.multivariate_normal([0.0] * z_dim, np.eye(z_dim), y.shape[0])
    y_data = np.concatenate([y, z], axis=-1).astype("float32")

    # Make dataset generator
    x_data = tf.data.Dataset.from_tensor_slices(x_data)
    y_data = tf.data.Dataset.from_tensor_slices(y_data)
    dataset = (
        tf.data.Dataset.zip((x_data, y_data))
        .shuffle(buffer_size=X.shape[0])
        .batch(n_batch, drop_remainder=True)
        .repeat()
    )

    val_dataset = dataset.take(int(data_length * 0.15))
    train_dataset = dataset.skip(int(data_length * 0.15))

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

    trainer = Trainer(
        model=model,
        x_dim=x_dim,
        y_dim=y_dim,
        z_dim=z_dim,
        pde_loss_func=pde_loss_func,
        pde_applied_forward=True,
    )
    trainer.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=0.5))

    LossFactor = UpdateLossFactor(n_epoch)
    EarlyStop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=6,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    # logger = NBatchLogger(n_display, n_epoch)
    _ = trainer.fit(
        train_dataset,
        batch_size=n_batch,
        epochs=n_epoch,
        steps_per_epoch=n_data // n_batch,
        callbacks=[LossFactor, EarlyStop],
        verbose=1,
        validation_data=val_dataset,
    )

    ## CHECK RESULTS ##

    # fig, ax = plt.subplots(1, facecolor='white', figsize=(8, 5))
    # ax.plot(hist.history['total_loss'], 'k.-', label='total loss')
    # ax.plot(hist.history['forward_loss'], 'b.-', label='forward loss')
    # ax.plot(hist.history['latent_loss'], 'g.-', label='latent loss')
    # ax.plot(hist.history['rev_loss'], 'r.-', label='inverse loss')
    # if pde_loss_func != None:
    #     ax.plot(hist.history['pde_loss'], 'y.-', label='pde loss')
    # plt.legend()
    # if pde_loss_func == None:
    #     plt.savefig("../data/trained_models/INN/latest/trained_model_train_hist")
    # else:
    #     plt.savefig("../data/trained_models/INNPINN/latest/trained_model_train_hist")

    # z = np.random.multivariate_normal([1.] * z_dim, np.eye(z_dim), y.shape[0])
    # y_data = np.concatenate([y, z], axis=-1).astype('float32')
    # x_pred = model.inverse(y_data).numpy()
    # x_data = np.concatenate([X, pad_x], axis=-1).astype('float32')
    # y_pred = model(x_data).numpy()

    # print(" -- Saving model")
    # model.save("./models/")

    # Make the folder if it does not exist
    model_type = "INNPINN" if pde_loss_func else "INN"
    noise_str = "noise/" if noise_experiment else ""
    save_model_path = f"{model_dir}/{noise_str}{model_type}/{subset}_sensors{num_sensors}"

    os.makedirs(save_model_path, exist_ok=True)

    # Save the model
    config.to_file(f"{save_model_path}/INNConfig.pkl")
    model.save_weights(f"{save_model_path}/trained_model.weights.h5")

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


def simple_run(
    dt,
    subset="all",
    num_sensors=64,
    use_pde=False,
    noise_experiment=False,
    config: INNConfig = None,
):
    data, labels, _, _ = get_data(
        dt,
        subset=subset,
        num_sensors=num_sensors,
        shuffle_data=True,
        use_pde=use_pde,
        noise_experiment=noise_experiment,
    )

    # data = np.concatenate([data, test_d], axis=0).astype('float32')
    # labels = np.concatenate([labels, test_l], axis=0).astype('float32')

    # print(f"{data.shape=}")
    # print(f"{labels.shape=}")

    if config == None:
        config = INNConfig(4, 4, 128, 0, 0, 32, None)

    config.x_dim = data.shape[1]
    config.y_dim = labels.shape[1]
    config.z_dim += 1 if ((32 + labels.shape[1]) % 2 == 1) else 0
    if use_pde:
        pde_loss_func = hydro.interior_loss
        config.pde_loss_func = pde_loss_func
    print("Running with config:", config)

    run_inn(
        data,
        labels,
        config,
        n_batch=8,
        n_epoch=16,
        datatype=dt,
        subset=subset,
        num_sensors=num_sensors,
        noise_experiment=noise_experiment,
    )


if __name__ == "__main__":
    # gridsearch_z_dim()
    # gridsearch_nn_architecture()
    simple_run(DataType.Hydro, use_pde=False)
    # simple_run_repeated()
