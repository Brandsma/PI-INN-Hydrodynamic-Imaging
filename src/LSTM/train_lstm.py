if __name__=="__main__":
    import sys
    sys.path.append("..")

import os
from pathlib import Path

from lib import LSTM, params
from lib.peregrine_util import get_scratch_dir

from lib.logger import setup_logger

log = setup_logger(__name__)


# TODO: remove read_inputs and replace it with CLI arguments
def read_inputs():
    n_nodes = 128
    n_epochs = 16
    window_size = 16
    stride = 2
    alpha = 0.05
    decay = 1e-9
    shuffle_data = True
    data_split = 0.8
    dropout = 0.2
    # train_loc = get_scratch_dir() + "/data/combined.npy"
    train_loc = "../../data/simulation_data/combined.npy"
    ac_fun = "tanh"
    return n_nodes, n_epochs, window_size, stride, \
        alpha, decay, shuffle_data, data_split, dropout, train_loc, ac_fun


if __name__ == "__main__":

    (n_nodes, n_epochs, window_size, stride, alpha, decay, \
     shuffle_data, data_split, dropout_ratio, train_location, ac_fun) =  \
        read_inputs()

    # Load settings
    settings = params.Settings(window_size, stride, n_nodes, \
                               alpha, decay, n_epochs, shuffle_data, data_split, dropout_ratio, \
               train_location, ac_fun)
    # Load data
    data = params.Data(settings, train_location)

    data.normalize()

    # Initiate the LSTM network using data and settings
    network = LSTM.LSTM_network(data, settings)
    network.model.summary()
    # Train the network
    network.train()

    # Save the network for later use
    trained_models_folder = "../../data/trained_models/LSTM/"
    Path(trained_models_folder).mkdir(parents=True, exist_ok=True)
    log.info(f"Saving trained model to {trained_models_folder}{settings.name}...")
    network.save_model(f"{trained_models_folder}{settings.name}")

    # # Test the network
    # network.test(data.test_data, data.test_labels)
    # # Save results
    # network.save_results()
