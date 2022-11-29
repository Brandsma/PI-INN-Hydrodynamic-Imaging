from lib import params

from LSTM.train_lstm import train_lstm as external_train_lstm

def train_lstm(settings=None):
    data = params.Data(settings, "../data/simulation_data/tiny/combined.npy")
    data.normalize()

    trained_models_folder = "../data/trained_models/LSTM/"

    external_train_lstm(trained_models_folder, data, settings)


if __name__=="__main__":
    train_lstm()
