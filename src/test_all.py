from LSTMTester import LSTMTester
from INNTester import INNTester
from INNPINNTester import INNPINNTester
from tqdm import tqdm

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

def grid_search():
    print("Trying INN...")
    n_couple_layer_options = [2,4,8]
    n_hid_layers_options = [2,4,8]
    n_hid_dim_options = [16,32,64,128, 256]
    z_dim_options = [2,16,32,64]

    inn_options = cartesian_coord(n_couple_layer_options, n_hid_layers_options, n_hid_dim_options, z_dim_options)

    best_config = None
    best_score = None
    for (n_couple_layer, n_hid_layer, n_hid_dim, z_dim) in tqdm(inn_options):
        config = INNConfig(n_couple_layer, n_hid_layer, n_hid_dim, 0, 0, z_dim, None)
        train_inn(config, use_pde=False)
        tester = INNTester()
        tester.set_input_data(30,30,"offset")
        tester.get_data()
        result_data = tester.result_data

        new_score = result_data[2] * 0.2 + result_data[3] * 0.8 + abs(30 - result_data[3]) * 0.4 + abs(30 - result_data[4]) * 0.4

        if best_score is None:
            best_score = new_score
            best_config = config
        elif new_score < best_score:
            best_score = new_score
            best_config = config

    print("Best config for INN is:")
    print("config:", best_config)

    print("Trying INNPINN...")
    best_config = None
    best_score = None
    for (n_couple_layer, n_hid_layer, n_hid_dim, z_dim) in tqdm(inn_options):
        config = INNConfig(n_couple_layer, n_hid_layer, n_hid_dim, 0, 0, z_dim, None)
        train_inn(config, use_pde=True)
        tester = INNTester()
        tester.set_input_data(30,30,"offset")
        tester.get_data()
        result_data = tester.result_data

        new_score = result_data[2] * 0.2 + result_data[3] * 0.8 + abs(30 - result_data[3]) * 0.4 + abs(30 - result_data[4]) * 0.4

        if best_score is None:
            best_score = new_score
            best_config = config
        elif new_score < best_score:
            best_score = new_score
            best_config = config

    print("Best config for PINN is:")
    print("config:", best_config)

    print("Trying LSTM...")
    n_nodes_options = [32, 64, 128, 256]
    ac_fun_options = ["tanh", "sigmoid"]

    LSTM_options = cartesian_coord(n_nodes_options, ac_fun_options)

    best_score = None
    best_config = None
    for (n_nodes, ac_fun) in tqdm(LSTM_options):
        settings = params.Settings(16, 2, n_nodes, 0.05, 1e-9, 8, True, 0.8, 0.2, "../data/simulation_data/tiny/combined.npy", ac_fun)
        train_lstm(settings)

        tester = LSTMTester()
        tester.set_input_data(30,30,"offset")
        tester.get_data()
        result_data = tester.result_data

        new_score = result_data[1] * 0.8 + abs(30 - result_data[2]) * 0.4 + abs(30 - result_data[3]) * 0.4

        if best_score is None:
            best_score = new_score
            best_config = settings
        elif new_score < best_score:
            best_score = new_score
            best_config = settings

    print("Best config for LSTM is:")
    best_config.__printSettings()

if __name__ == "__main__":
    grid_search()
