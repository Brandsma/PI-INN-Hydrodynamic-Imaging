from numpy.core.fromnumeric import mean
import csv
from LSTMTester import LSTMTester
from INNTester import INNTester, generate_inn_options
from INNPINNTester import INNPINNTester
from tqdm import tqdm
import pyfiglet

from util import cartesian_coord

from train_inn import train_inn
from INN.inn import INNConfig

from train_lstm import train_lstm
from lib import params

def run_tests():
    testers = [INNTester(), INNPINNTester(), LSTMTester()]


    a_list = [10, 20, 30, 40, 50]
    w_list = [10, 20, 30, 40, 50]
    a_w_meshgrid = cartesian_coord(a_list, w_list)
    subsets = ["offset", "inverse_offset", "parallel", "mult_path", "far_off_parallel"]

    for tester in testers:
        print(f"Currently running with tester '{tester.__class__.__name__}'")
        for subset in subsets:
            for a, w in a_w_meshgrid:
                tester.set_input_data(a, w, subset)

                tester.get_data()

                tester.save_result_data()

def write_inn_config(config_score_dict):
    with open("./inn_score.csv", 'w') as f:
        w = csv.writer(f)
        w.writerow(["n_couple_layer", "n_hid_layer", "n_hid_dim", "z_dim", "Weighted_score", "MSE_forward", "MSE_backward", "Speed_error", "Volume_error"])

        for config in config_score_dict:
            scores = config_score_dict[config]
            line = [config.n_couple_layer, config.n_hid_layer, config.n_hid_dim, config.z_dim, scores["weighted_score"], scores["MSE_forward"], scores["MSE_backward"], scores["Volume_error"], scores["Speed_error"]]
            w.writerow(line)

def grid_search():
    print("Trying INN...")

    inn_options = generate_inn_options()

    config_score_dict = {}
    # Try out various hyperparameter combinations
    for (n_couple_layer, n_hid_layer, n_hid_dim, z_dim) in tqdm(inn_options):
    # for (n_couple_layer, n_hid_layer, n_hid_dim, z_dim) in tqdm([(1,1,2,16), (1,1,2,32)]):
        config = INNConfig(n_couple_layer, n_hid_layer, n_hid_dim, 0, 0, z_dim, None)
        train_inn(config, use_pde=False)
        tester = INNTester()

        mse_forward = []
        mse_backward = []
        volume_error = []
        speed_error = []
        weighted_score = []

        # Loop over 5 unique volume/speed subsets
        for speed, volume in [(10,10), (10,50), (50,10), (30,30), (50,50)]:
            tester.set_input_data(speed,volume,"offset")
            tester.get_data()
            result_data = tester.result_data

            new_score = mean(result_data[:, 2]) * 0.1 + mean(result_data[:, 3]) * 0.5 + abs(volume - mean(result_data[:, 4])) * 0.2 + abs(speed - mean(result_data[:, 5])) * 0.2
            weighted_score.append(new_score)
            mse_forward.append(mean(result_data[:, 2]))
            mse_backward.append(mean(result_data[:, 3]))
            volume_error.append(abs(volume - mean(result_data[:, 4])))
            speed_error.append(abs(speed - mean(result_data[:, 5])))


        config_score_dict[config] = {
            "weighted_score": mean(weighted_score),
            "MSE_forward": mean(mse_forward),
            "MSE_backward": mean(mse_backward),
            "Volume_error": mean(volume_error),
            "Speed_error": mean(speed_error),
        }

    write_inn_config(config_score_dict)


    print("Trying INNPINN...")
    config_score_dict = {}
    # Try out various hyperparameter combinations
    for (n_couple_layer, n_hid_layer, n_hid_dim, z_dim) in tqdm(inn_options):
    # for (n_couple_layer, n_hid_layer, n_hid_dim, z_dim) in tqdm([(1,1,2,16), (1,1,2,32)]):
        config = INNConfig(n_couple_layer, n_hid_layer, n_hid_dim, 0, 0, z_dim, None)
        train_inn(config, use_pde=False)
        tester = INNTester()

        mse_forward = []
        mse_backward = []
        volume_error = []
        speed_error = []
        weighted_score = []

        # Loop over 5 unique volume/speed subsets
        for speed, volume in [(10,10), (10,50), (50,10), (30,30), (50,50)]:
            tester.set_input_data(speed,volume,"offset")
            tester.get_data()
            result_data = tester.result_data

            new_score = mean(result_data[:, 2]) * 0.1 + mean(result_data[:, 3]) * 0.5 + abs(volume - mean(result_data[:, 4])) * 0.2 + abs(speed - mean(result_data[:, 5])) * 0.2
            weighted_score.append(new_score)
            mse_forward.append(mean(result_data[:, 2]))
            mse_backward.append(mean(result_data[:, 3]))
            volume_error.append(abs(volume - mean(result_data[:, 4])))
            speed_error.append(abs(speed - mean(result_data[:, 5])))


        config_score_dict[config] = {
            "weighted_score": mean(weighted_score),
            "MSE_forward": mean(mse_forward),
            "MSE_backward": mean(mse_backward),
            "Volume_error": mean(volume_error),
            "Speed_error": mean(speed_error),
        }

    write_inn_config(config_score_dict)

    # best_config = None
    # best_score = None
    # for (n_couple_layer, n_hid_layer, n_hid_dim, z_dim) in tqdm(inn_options):
    #     config = INNConfig(n_couple_layer, n_hid_layer, n_hid_dim, 0, 0, z_dim, None)
    #     train_inn(config, use_pde=True)
    #     tester = INNTester()
    #     tester.set_input_data(30,30,"offset")
    #     tester.get_data()
    #     result_data = tester.result_data

    #     new_score = result_data[2] * 0.2 + result_data[3] * 0.8 + abs(30 - result_data[3]) * 0.4 + abs(30 - result_data[4]) * 0.4

    #     if best_score is None:
    #         best_score = new_score
    #         best_config = config
    #     elif new_score < best_score:
    #         best_score = new_score
    #         best_config = config

    # print("Best config for PINN is:")
    # print("config:", best_config)

    # print("Trying LSTM...")
    # n_nodes_options = [32, 64, 128, 256]
    # ac_fun_options = ["tanh", "sigmoid"]

    # LSTM_options = cartesian_coord(n_nodes_options, ac_fun_options)

    # best_score = None
    # best_config: params.Settings = params.Settings()
    # for (n_nodes, ac_fun) in tqdm(LSTM_options):
    #     settings = params.Settings(16, 2, n_nodes, 0.05, 1e-9, 8, True, 0.8, 0.2, "../data/simulation_data/tiny/combined.npy", ac_fun)
    #     train_lstm(settings)

    #     tester = LSTMTester()
    #     tester.set_input_data(30,30,"offset")
    #     tester.get_data()
    #     result_data = tester.result_data

    #     new_score = result_data[1] * 0.8 + abs(30 - result_data[2]) * 0.4 + abs(30 - result_data[3]) * 0.4

    #     if best_score is None:
    #         best_score = new_score
    #         best_config = settings
    #     elif new_score < best_score:
    #         best_score = new_score
    #         best_config = settings

    # print("Best config for LSTM is:")
    # best_config.__printSettings()

if __name__ == "__main__":
    ascii_banner = pyfiglet.figlet_format("Welcome To The Testing Suite")
    print(ascii_banner)
    grid_search()
