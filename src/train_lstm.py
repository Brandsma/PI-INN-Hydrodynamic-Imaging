from lib import params

from LSTM.train_lstm import train_lstm as external_train_lstm


def read_inputs(noise_experiment=False):
    n_nodes = 256
    n_epochs = 16
    window_size = 16
    stride = 2
    alpha = 0.05
    decay = 1e-9
    shuffle_data = True
    data_split = 0.8
    dropout = 0
    # train_loc = get_scratch_dir() + "/data/combined.npy"
    if noise_experiment:
        train_loc = "../data/simulation_data/noise/combined_groups/combined.npy"
    else:
        train_loc = "../data/simulation_data/combined_groups/combined.npy"
    ac_fun = "tanh"
    return (
        n_nodes,
        n_epochs,
        window_size,
        stride,
        alpha,
        decay,
        shuffle_data,
        data_split,
        dropout,
        train_loc,
        ac_fun,
    )


def train_lstm(
    settings=None,
    data_folder="../data/simulation_data/tiny/combined.npy",
    noise_experiment=False,
):
    data = params.Data(settings, data_folder)
    data.normalize()

    if noise_experiment:
        trained_models_folder = "../data/trained_models/noise/LSTM/"
    else:
        trained_models_folder = "../data/trained_models/LSTM/"

    external_train_lstm(trained_models_folder, data, settings)


if __name__ == "__main__":
    noise_experiment = True

    (
        n_nodes,
        n_epochs,
        window_size,
        stride,
        alpha,
        decay,
        shuffle_data,
        data_split,
        dropout_ratio,
        train_location,
        ac_fun,
    ) = read_inputs(noise_experiment)
    settings = params.Settings(
        window_size,
        stride,
        n_nodes,
        alpha,
        decay,
        n_epochs,
        shuffle_data,
        data_split,
        dropout_ratio,
        train_location,
        ac_fun,
    )
    if noise_experiment:
        num_sensor_variants = [8]
    else:
        # num_sensor_variants = [1, 3, 8, 64]
        num_sensor_variants = [8]
    for num_sensor in num_sensor_variants:
        print(f"Training LSTM with {num_sensor} sensors")
        print(f"Train location: {train_location}")

        settings.num_sensors = num_sensor
        train_lstm(settings, train_location, noise_experiment)

    print("Done")
