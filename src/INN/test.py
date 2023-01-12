import statistics
from INN.data import get_data, DataType
from tqdm import tqdm

import numpy as np
from INN.inn import create_model, INNConfig
import INN.hydro as hydro
import INN.sine as sine
from sklearn.metrics import mean_squared_error

def run_test_on_model(subset="all", num_sensors=64, test_pinn=False):
    base_folder= "../data/trained_models/INN"
    if test_pinn:
        base_folder= "../data/trained_models/INNPINN"

    results_folder = f"../results/{'PINN' if test_pinn else 'INN'}/{subset}"
    print(f"using {base_folder} now...")

    # Config
    config: INNConfig = INNConfig.from_file(f"{base_folder}/latest/INNConfig.pkl")
    dt = DataType.Hydro

    x_dim = config.x_dim
    y_dim = config.y_dim
    z_dim = config.z_dim
    tot_dim = y_dim + config.z_dim


    # Get Model
    model = create_model(tot_dim, config.n_couple_layer, config.n_hid_layer, config.n_hid_dim)
    latest_model_path = f"{base_folder}/latest/trained_model_weights.tf"
    model.load_weights(latest_model_path)

    # forward_mse_errors = []
    # backward_mse_errors = []

    # Get dataset
    _, _, data, labels = get_data(dt, subset=subset, num_sensors=num_sensors, shuffle_data=False)

    # data = np.concatenate([train_data, data], axis=0).astype('float32')
    # labels = np.concatenate([train_labels, labels], axis=0).astype('float32')

    x_data, x_pred, y_data, y_pred = test_model(model, data, labels, x_dim, y_dim, z_dim)

    if dt == DataType.Sine:
        sine.plot_results(x_data, x_pred, y_data, y_pred, title="Sine")
    elif dt == DataType.Hydro:
        hydro.plot_results_from_array(x_data, x_pred, subset, num_sensors, title=f"Sensors: {num_sensors}", savefig=True, savepath=results_folder)
        # hydro.plot_results(x_data, x_pred, y_data, y_pred, x_dim, y_dim, run_idx, title=f"Hydro | {run_idx}", savefig=True)

    np.save(results_folder+f"/x_data_{num_sensors}.npy", x_data)
    np.save(results_folder+f"/y_data_{num_sensors}.npy", y_data)
    np.save(results_folder+f"/x_pred_{num_sensors}.npy", x_pred)
    np.save(results_folder+f"/y_pred_{num_sensors}.npy", y_pred)


    #     forward_mse_errors.append(mean_squared_error(y_data[:, :y_dim], y_pred[:, :y_dim]))
    #     backward_mse_errors.append(mean_squared_error(x_data[:, :x_dim], x_pred[:, :x_dim]))

    # print("Forward --")
    # print(sum(forward_mse_errors) / len(forward_mse_errors), statistics.stdev(forward_mse_errors))
    # print("Backward --")
    # print(backward_mse_errors)
    # print(sum(backward_mse_errors) / len(backward_mse_errors), statistics.stdev(backward_mse_errors))

    print(" -- Done -- ")


def test_model(model, data, labels, x_dim, y_dim, z_dim):
    z_dim = z_dim
    tot_dim = y_dim + z_dim
    pad_dim = tot_dim - x_dim

    # Test model
    pad_x = np.zeros((data.shape[0], pad_dim))
    pad_x = np.random.multivariate_normal([0.] * pad_dim, np.eye(pad_dim),
                                          data.shape[0])

    z = np.random.multivariate_normal([1.] * z_dim, np.eye(z_dim), labels.shape[0])
    y_data = np.concatenate([labels, z], axis=-1).astype('float32')
    x_pred = model.inverse(y_data).numpy()
    x_data = np.concatenate([data, pad_x], axis=-1).astype('float32')
    y_pred = model(x_data).numpy()

    return x_data, x_pred, y_data, y_pred



if __name__ == '__main__':
    run_test_on_model()
