import numpy as np
from pyoselm import OSELMClassifier, OSELMRegressor
from sklearn.datasets import load_digits, make_regression
from sklearn.model_selection import train_test_split

data = np.load("../../data/a1_normw1_theta0.npy")
print(data.shape)

# number of sensors * xy deflection
data = np.transpose(data, (1, 2, 3, 0))
data = np.reshape(
    data, (data.shape[0], data.shape[1], data.shape[2] * data.shape[3]))

x_data = data[4, :, :]
print("X", x_data.shape)


def create_labels(x_data, y):
    labels = []
    for idx in range(x_data.shape[0]):
        labels.append([idx + 1, y])
    return np.array(labels)


labels = create_labels(x_data, 4)
print("y", labels.shape)


# Calculate how many windows are in total dataset.
def tot_windows(data, window_size, stride):
    tot_windows = 0
    for sample_idx in range(0, len(data) - window_size + 1, stride):
        tot_windows += 1
    return tot_windows


# Turn dataset into windows based on stride and window size (def. values 30 and 1)
def turn_into_windows(data, labels, window_size=30, stride=1):
    print("Turning into windows")
    print(data.shape, labels.shape)
    # 64 sensors x 2 deflections per sensor = 14 inputs
    n_inputs = 128
    # x- and y-coordinates = 2 outputs
    n_outputs = 2

    n_win = tot_windows(data, window_size, stride)
    x = np.zeros((n_win, window_size, n_inputs))
    t = np.zeros((n_win, n_outputs))
    tot_idx = 0
    for sample_idx in range(0, len(data) - window_size + 1, stride):
        x[tot_idx, :] = np.reshape(data[sample_idx:sample_idx + window_size],
                                   (1, window_size, n_inputs))
        t[tot_idx, :] = np.reshape(
            labels[sample_idx + window_size - 1:sample_idx + window_size],
            (1, n_outputs))
        tot_idx += 1
    return x, t


win_x, win_y = turn_into_windows(x_data, labels)
# flatten arrays
win_x = win_x.reshape(win_x.shape[0], -1)

# Model
oselmr = OSELMRegressor(n_hidden=20,
                        activation_func='sigmoid',
                        random_state=123)
# Data
X_train, X_test, y_train, y_test = train_test_split(win_x,
                                                    win_y,
                                                    test_size=0.2,
                                                    random_state=123)
n_batch = X_train.shape[0] // 100

# Fit model with chunks of data
for i in range(100):
    X_batch = X_train[i * n_batch:(i + 1) * n_batch]
    y_batch = y_train[i * n_batch:(i + 1) * n_batch]
    print(X_batch.shape)
    oselmr.fit(X_batch, y_batch)
    print("Train score for batch %i: %s" %
          (i + 1, str(oselmr.score(X_batch, y_batch))))

# Results
print("Train score of total: %s" % str(oselmr.score(X_train, y_train)))
print("Test score of total: %s" % str(oselmr.score(X_test, y_test)))
print("")
