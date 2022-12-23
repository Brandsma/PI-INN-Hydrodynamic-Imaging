from INN.test import run_test_on_model
import tensorflow as tf

from lib import params, LSTM

def test_lstm_model(new_model):
    train_loc = f"../data/simulation_data/combined.npy"
    model_location = "window_size:16&stride:2&n_nodes:256&alpha:0.05&decay:1e-09&n_epochs:16&shuffle_data:True&data_split:0.8&dropout_ratio:0.0&ac_fun:tanh"
    train_location = train_loc

    # Load settings
    settings = params.Settings.from_model_location(model_location,
                                                   train_location)
    # Load data
    data = params.Data(settings, train_location)

    data.normalize()

    network = LSTM.LSTM_network(data, settings)
    network.model = new_model

    network.test(data.test_data,
                 data.test_labels,
                 dirname=f"../results/")

    print(network.pred.shape)
    print(network.labels.shape)

def main():
    model = tf.keras.models.load_model(
        "../data/trained_models/LSTM/window_size:16&stride:2&n_nodes:256&alpha:0.05&decay:1e-09&n_epochs:16&shuffle_data:True&data_split:0.8&dropout_ratio:0.0&ac_fun:tanh"
    )
    test_lstm_model(model)

if __name__ == '__main__':
    main()
