import numpy as np
from matplotlib import pyplot as plt
from pyoselm import OSELMClassifier, OSELMRegressor
from sklearn.datasets import load_digits, make_regression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

data = np.load("../../data/a1_normw1_theta0.npy")
labels = np.load("../../data/a1_normw1_theta0_labels.npy")

print(labels.shape)

# number of sensors * xy deflection
data = np.transpose(data, (1, 2, 3, 0))
labels = np.transpose(labels, (1, 2, 3, 0))
data = np.reshape(
    data, (data.shape[0] * data.shape[1], data.shape[2] * data.shape[3]))
labels = np.reshape(
    labels,
    (labels.shape[0] * labels.shape[1], labels.shape[2], labels.shape[3]))
labels = labels[:, :, 0]

x_data = data
y_labels = labels
print("X", x_data.shape)
print("y", y_labels.shape)
print(x_data[0])

# Add noise
noise = np.random.normal(0, .000001, x_data.shape)
x_data = x_data + noise

# from matplotlib.animation import FuncAnimation

# fig, ax = plt.subplots()
# xdata, ydata = [], []
# ln, = plt.plot(x_data[0, :], 'b')

# def init():
#     ax.set_xlim(0, 2048)
#     ax.set_ylim(-1, 1)
#     plt.axvline(x=1024)
#     return ln,

# def update(frame_number):
#     print(frame_number, y_labels[frame_number])
#     ln.set_ydata(x_data[frame_number, :])
#     return ln,

# ani = FuncAnimation(fig, update, x_data.shape[0], init_func=init, blit=True)
# plt.show()


# Calculate how many windows are in total dataset.
def tot_windows(data, window_size, stride):
    tot_windows = 0
    for sample_idx in range(0, len(data) - window_size + 1, stride):
        tot_windows += 1
    return tot_windows


# Turn dataset into windows based on stride and window size (def. values 30 and 1)
def turn_into_windows(data, labels, window_size=15, stride=1):
    print("Turning into windows")
    print(data.shape, labels.shape)
    # 1024 sensors x 2 deflections per sensor = 14 inputs
    n_inputs = 2048
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


# win_x, win_y = turn_into_windows(x_data, y_labels)
# # # flatten arrays
# win_x = win_x.reshape(win_x.shape[0], -1)

# Multiple 'runs'
win_x = np.repeat(x_data, 2, axis=0)
win_y = np.repeat(y_labels, 2, axis=0)

print("windowed X ", win_x.shape)
print("windowed y ", win_y.shape)
# Model
# oselmr = OSELMRegressor(n_hidden=40,
#                         activation_func='sigmoid',
#                         random_state=123)
# Data
X_train, X_test, y_train, y_test = train_test_split(win_x,
                                                    win_y,
                                                    test_size=0.2,
                                                    random_state=42)
n_batch = 32
print("n_batch size", n_batch)

# ELM
# # Fit model with chunks of data
# for i in tqdm(range(1000)):
#     X_batch = X_train[i * n_batch:(i + 1) * n_batch]
#     y_batch = y_train[i * n_batch:(i + 1) * n_batch]
#     oselmr.fit(X_batch, y_batch)
#     # print("Train score for batch %i: %s" %
#     #       (i + 1, str(oselmr.score(X_batch, y_batch))))

# # Results
# print("Train score of total: %s" % str(oselmr.score(X_train, y_train)))
# print("Test score of total: %s" % str(oselmr.score(X_test, y_test)))
# print("")

# LSTM
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim=2048, output_dim=64))

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128))

# Add a Dense layer with 2 units.
model.add(layers.Dense(2))

model.summary()
model.compile(
    loss=keras.losses.MeanAbsoluteError(),
    optimizer="sgd",
    metrics=["accuracy", "mean_absolute_error"],
)

model.fit(X_train,
          y_train,
          validation_data=(X_test, y_test),
          batch_size=n_batch,
          epochs=50)

y_pred = model.predict(X_test)
print(len(X_test))
print(X_test[0], y_pred[0], y_test[0])
print(X_test[50], y_pred[50], y_test[50])
print(X_test[100], y_pred[100], y_test[100])

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("Score:", score)
