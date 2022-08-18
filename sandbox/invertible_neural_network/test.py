from tensorflow import keras

import itertools

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import cm
from sklearn.preprocessing import StandardScaler

from flow import *
from utils import *

## SETUP DATA ##
data = np.load('../../data/simulation_data/a20_normw20_data.npy')
X_raw = np.load('../../data/simulation_data/a20_normw20_data_labels.npy')

# labels = ['red','red','red','red','blue','blue','green','purple']

x_dim = X_raw.shape[2]
y_dim = data.shape[2]
z_dim = 100
tot_dim = y_dim + z_dim
pad_dim = tot_dim - x_dim
n_data = data.shape[0] * data.shape[1]
n_couple_layer = 3
n_hid_layer = 3
n_hid_dim = 64

n_batch = 200
n_epoch = 1000
n_display = 100

###
# Make data

# X_raw = np.zeros((data.shape[0], data.shape[1], x_dim), dtype='float32')
# for y in range(data.shape[0]):
#     for x in range(data.shape[1]):
#         X_raw[y, x, :] = np.array([y + 1, x + 1])

# TODO: Duplicate the data to have some more training data for now?

###
# Preprocess
X = X_raw.reshape((-1, x_dim))
#X = StandardScaler().fit_transform(X)

y = data.reshape((-1, data.shape[2]))

###
# Pad data
pad_x = np.zeros((X.shape[0], pad_dim))
x_data = np.concatenate([X, pad_x], axis=-1).astype('float32')
# TODO: This z should be a gaussian (I think based on the paper), which it is right now.
# But do check if this is correct in the future
z = np.random.multivariate_normal([0.] * z_dim, np.eye(z_dim), y.shape[0])
y_data = np.concatenate([z, y], axis=-1).astype('float32')

# Make dataset generator
x_data = tf.data.Dataset.from_tensor_slices(x_data)
y_data = tf.data.Dataset.from_tensor_slices(y_data)
dataset = (tf.data.Dataset.zip(
    (x_data, y_data)).shuffle(buffer_size=X.shape[0]).batch(
        n_batch, drop_remainder=True).repeat())


model = NVP(tot_dim, n_couple_layer, n_hid_layer, n_hid_dim, name='NVP')
x = tfk.Input((tot_dim, ))
model(x)
model.summary()

model.compile()


loaded_model = keras.models.load_model('./models')
model.set_weights(loaded_model.get_weights())

z = np.random.multivariate_normal([0.] * z_dim, np.eye(z_dim), y.shape[0])
y = np.concatenate([z, y], axis=-1).astype('float32')
x_pred = model.inverse(y).numpy()

print(pad_dim)
for idx in range(200):
    print(x_pred[idx][0:3], " - ", X[idx])
    
print(np.linalg.norm(x_pred[:][0:3] - X))

