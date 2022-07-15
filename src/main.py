import numpy as np
from lib import params
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize

from elm import ELM


def main(data):
    # matData = loadmat("barrel_near_traces.mat")
    # motions = matData['sensorMotion'][0]

    # print(data.train_data.shape)
    # print(data.train_labels.shape)
    # print(data.test_data.shape)
    # print(data.test_labels.shape)
    print(data.train_data.shape)
    end_range = 200

    train_data = normalize(
        np.reshape(data.train_data[0:end_range, :, :],
                   (-1, data.train_data.shape[2])))
    train_labels = np.reshape(data.train_labels[0:end_range, :, 0], (-1, ))

    # test_data = normalize(data.test_data[0, :, :])
    # test_labels = data.test_labels[0, :, :]

    # train_data = normalize(motions[0][0])
    # targetVector = [0, 1, 0, 0, 0, 0, 0]
    # target = [targetVector for _ in train_data]
    # target = [2 for _ in train_data]

    print(train_data.shape)
    print(train_labels.shape)

    le = LabelEncoder()
    le.fit(train_labels)

    print(f"num classes: {len(list(le.classes_))}")

    X_train, X_test, y_train, y_test = train_test_split(
        train_data, le.transform(train_labels), test_size=0.3)

    elm = ELM(hid_num=4000).fit(X_train, y_train)

    print("ELM Accuracy %0.3f " % elm.score(X_test, y_test))


# TODO: remove read_inputs and replace it with CLI arguments
def read_inputs():
    n_nodes = 128
    n_epochs = 30
    window_size = 16
    stride = 2
    alpha = 0.05
    decay = 1e-9
    shuffle_data = True
    data_split = 0.8
    dropout = 0
    # train_loc = get_scratch_dir() + "/data/combined.npy"
    train_loc = "../data/simulation_data/combined.npy"
    ac_fun = "relu"
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

    main(data)
