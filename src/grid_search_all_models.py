from numpy.core.fromnumeric import mean
import csv
from LSTMTester import LSTMTester
from INNTester import INNTester, generate_inn_options
from INNPINNTester import INNPINNTester
from tqdm import tqdm
import pyfiglet
from time import perf_counter

from util import cartesian_coord

from train_inn import train_inn
from INN.inn import INNConfig

from train_lstm import train_lstm
from lib import params


def run_tests() -> None:
    """
    Run tests for all the testers and all the subsets.

    :return: None
    """
    testers = [INNTester(), INNPINNTester(), LSTMTester()]

    a_list = [10, 20, 30, 40, 50]
    w_list = [10, 20, 30, 40, 50]
    a_w_meshgrid = cartesian_coord(a_list, w_list)
    subsets = ["offset", "inverse_offset",
               "parallel", "mult_path", "far_off_parallel"]

    for tester in testers:
        print(f"Currently running with tester '{tester.__class__.__name__}'")
        for subset in subsets:
            for a, w in a_w_meshgrid:
                tester.set_input_data(a, w, subset)

                tester.get_data()

                tester.save_result_data()

def write_pinn_config(config_score_dict):
    with open("./pinn_score.csv", 'w') as f:
        w = csv.writer(f)
        w.writerow(["n_couple_layer", "n_hid_layer", "n_hid_dim", "z_dim",
                   "Weighted_score", "MSE_forward", "MSE_backward", "Speed_error", "Volume_error"])

        for config in config_score_dict:
            scores = config_score_dict[config]
            line = [config.n_couple_layer, config.n_hid_layer, config.n_hid_dim, config.z_dim, scores["weighted_score"],
                    scores["MSE_forward"], scores["MSE_backward"], scores["Volume_error"], scores["Speed_error"]]
            w.writerow(line)

def write_inn_config(config_score_dict):
    with open("./inn_score.csv", 'w') as f:
        w = csv.writer(f)
        w.writerow(["n_couple_layer", "n_hid_layer", "n_hid_dim", "z_dim",
                   "Weighted_score", "MSE_forward", "MSE_backward", "Speed_error", "Volume_error"])

        for config in config_score_dict:
            scores = config_score_dict[config]
            line = [config.n_couple_layer, config.n_hid_layer, config.n_hid_dim, config.z_dim, scores["weighted_score"],
                    scores["MSE_forward"], scores["MSE_backward"], scores["Volume_error"], scores["Speed_error"]]
            w.writerow(line)

def write_lstm_config(config_score_dict):
    with open("./lstm_score.csv", 'w') as f:
        w = csv.writer(f)
        w.writerow(["n_nodes", "ac_fun",
                   "Weighted_score", "Localization_error", "Speed_error", "Volume_error"])

        for config in config_score_dict:
            scores = config_score_dict[config]
            line = [config.n_nodes, config.ac_fun, scores["weighted_score"],
                    scores["Localization_error"], scores["Volume_error"], scores["Speed_error"]]
            w.writerow(line)

def grid_search():
    print("Trying INN...")

    # inn_options = generate_inn_options()

    # config_score_dict = {}
    # # Try out various hyperparameter combinations
    # for (n_couple_layer, n_hid_layer, n_hid_dim, z_dim) in tqdm(inn_options):
    #     # for (n_couple_layer, n_hid_layer, n_hid_dim, z_dim) in tqdm([(1,1,2,16), (1,1,2,32)]):
    #     config = INNConfig(n_couple_layer, n_hid_layer,
    #                        n_hid_dim, 0, 0, z_dim, None)
    #     train_inn(config, use_pde=False)
    #     tester = INNTester()

    #     mse_forward = []
    #     mse_backward = []
    #     volume_error = []
    #     speed_error = []
    #     weighted_score = []

    #     # Loop over 5 unique volume/speed subsets
    #     for speed, volume in [(10, 10), (10, 50), (50, 10), (30, 30), (50, 50)]:
    #         tester.set_input_data(volume, speed, "offset")
    #         tester.get_data()
    #         result_data = tester.result_data

    #         new_score = mean(result_data[:, 2]) * 0.1 + mean(result_data[:, 3]) * 0.5 + abs(mean(
    #             volume - result_data[:, 4])) * 0.2 + abs(mean(speed - result_data[:, 5])) * 0.2
    #         weighted_score.append(new_score)
    #         mse_forward.append(mean(result_data[:, 2]))
    #         mse_backward.append(mean(result_data[:, 3]))
    #         volume_error.append(abs(mean(volume - result_data[:, 4])))
    #         speed_error.append(abs(mean(speed - result_data[:, 5])))

    #     config_score_dict[config] = {
    #         "weighted_score": mean(weighted_score),
    #         "MSE_forward": mean(mse_forward),
    #         "MSE_backward": mean(mse_backward),
    #         "Volume_error": mean(volume_error),
    #         "Speed_error": mean(speed_error),
    #     }

    # write_inn_config(config_score_dict)

    print("Trying INNPINN...")
    config_score_dict = {}
    # Try out various hyperparameter combinations
    # for (n_couple_layer, n_hid_layer, n_hid_dim, z_dim) in tqdm(inn_options):
    # # for (n_couple_layer, n_hid_layer, n_hid_dim, z_dim) in tqdm([(1,1,2,16), (1,1,2,32)]):
    #     config = INNConfig(n_couple_layer, n_hid_layer,
    #                        n_hid_dim, 0, 0, z_dim, None)
    #     train_inn(config, use_pde=True)
    #     tester = INNTester()

    #     mse_forward = []
    #     mse_backward = []
    #     volume_error = []
    #     speed_error = []
    #     weighted_score = []

    #     # Loop over 5 unique volume/speed subsets
    #     for speed, volume in [(10, 10), (10, 50), (50, 10), (30, 30), (50, 50)]:
    #         tester.set_input_data(volume, speed, "offset")
    #         tester.get_data()
    #         result_data = tester.result_data

    #         new_score = mean(result_data[:, 2]) * 0.1 + mean(result_data[:, 3]) * 0.5 + abs(mean(
    #             volume - result_data[:, 4])) * 0.2 + abs(mean(speed - result_data[:, 5])) * 0.2
    #         weighted_score.append(new_score)
    #         mse_forward.append(mean(result_data[:, 2]))
    #         mse_backward.append(mean(result_data[:, 3]))
    #         volume_error.append(abs(mean(volume - result_data[:, 4])))
    #         speed_error.append(abs(mean(speed - result_data[:, 5])))

    #     config_score_dict[config] = {
    #         "weighted_score": mean(weighted_score),
    #         "MSE_forward": mean(mse_forward),
    #         "MSE_backward": mean(mse_backward),
    #         "Volume_error": mean(volume_error),
    #         "Speed_error": mean(speed_error),
    #     }

    # write_pinn_config(config_score_dict)


    print("Trying LSTM...")

    config_score_dict = {}
    # Try out various hyperparameter combinations
    # for (n_nodes, ac_fun) in tqdm(LSTM_options):
    for (n_nodes, ac_fun) in tqdm([(2, "relu"), (3, "tanh")]):
        settings = params.Settings(16, 2, n_nodes, 0.05, 1e-9, 8, True, 0.8, 0.2, "../data/simulation_data/tiny/combined.npy", ac_fun)
        train_lstm(settings)

        tester = LSTMTester()

        mse_localization = []
        volume_error = []
        speed_error = []
        weighted_score = []

        # Loop over 5 unique volume/speed subsets
        print("Testing LSTM...")
        for speed, volume in [(10, 10), (10, 50), (50, 10), (30, 30), (50, 50)]:
            tester.set_input_data(volume, speed, "offset")
            tester.get_data()
            result_data = tester.result_data

            new_score = mean(result_data[:, 1]) * 0.6 + abs(mean(volume - result_data[:, 2])) * 0.2 + abs(mean(speed - result_data[:,3])) * 0.2

            weighted_score.append(new_score)
            mse_localization.append(mean(result_data[:, 1]))
            volume_error.append(abs(mean(volume - result_data[:, 2])))
            speed_error.append(abs(mean(speed - result_data[:, 3])))

        config_score_dict[settings] = {
            "weighted_score": mean(weighted_score),
            "Localization_error": mean(mse_localization),
            "Volume_error": mean(volume_error),
            "Speed_error": mean(speed_error),
        }

    write_lstm_config(config_score_dict)


if __name__ == "__main__":
    ascii_banner = pyfiglet.figlet_format("Welcome To The Testing Suite")
    print(ascii_banner)
    grid_search()
