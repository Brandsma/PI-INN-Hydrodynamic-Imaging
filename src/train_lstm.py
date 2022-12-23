from lib import params

from LSTM.train_lstm import train_lstm as external_train_lstm

def read_inputs():
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
    train_loc = "../data/simulation_data/combined.npy"
    ac_fun = "tanh"
    return n_nodes, n_epochs, window_size, stride, \
        alpha, decay, shuffle_data, data_split, dropout, train_loc, ac_fun

def train_lstm(settings=None, data_folder="../data/simulation_data/tiny/combined.npy"):
    data = params.Data(settings, data_folder)
    data.normalize()

    trained_models_folder = "../data/trained_models/LSTM/"

    external_train_lstm(trained_models_folder, data, settings)


if __name__=="__main__":
    (n_nodes, n_epochs, window_size, stride, alpha, decay, \
     shuffle_data, data_split, dropout_ratio, train_location, ac_fun) =  \
        read_inputs()
    settings = params.Settings(window_size, stride, n_nodes, \
                               alpha, decay, n_epochs, shuffle_data, data_split, dropout_ratio, \
               train_location, ac_fun)
    train_lstm(settings, train_location)
