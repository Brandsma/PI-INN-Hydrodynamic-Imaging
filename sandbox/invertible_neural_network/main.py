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
labels = np.load('../../data/simulation_data/a20_normw20_data_labels.npy')

# data = np.transpose(data, (1, 2, 3, 0))
# data = np.reshape(
    # data, (data.shape[0], data.shape[1], data.shape[2] * data.shape[3]))


# labels = []
# labels = ['red','red','red','red','blue','blue','green','purple']

x_dim = labels.shape[2]
y_dim = data.shape[2]
z_dim = 100
tot_dim = y_dim + z_dim
pad_dim = tot_dim - x_dim
n_data = data.shape[0] * data.shape[1]
n_couple_layer = 3
n_hid_layer = 3
n_hid_dim = 512

n_batch = 200
n_epoch = 100
n_display = 100

###
# Make data

X_raw = labels

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

## INITIALIZE MODEL ##
model = NVP(tot_dim, n_couple_layer, n_hid_layer, n_hid_dim, name='NVP')
x = tfk.Input((tot_dim, ))
model(x)
model.summary()


class Trainer(tfk.Model):
    def __init__(self,
                 model,
                 x_dim,
                 y_dim,
                 z_dim,
                 tot_dim,
                 n_couple_layer,
                 n_hid_layer,
                 n_hid_dim,
                 shuffle_type='reverse'):
        super(Trainer, self).__init__()
        self.model = model
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.tot_dim = tot_dim
        self.x_pad_dim = tot_dim - x_dim
        self.y_pad_dim = tot_dim - (y_dim + z_dim)
        self.n_couple_layer = n_couple_layer
        self.n_hid_layer = n_hid_layer
        self.n_hid_dim = n_hid_dim
        self.shuffle_type = shuffle_type

        self.w1 = 5.
        self.w2 = 1.
        self.w3 = 10.
        self.loss_factor = 1.
        self.loss_fit = MSE
        self.loss_latent = MMD_multiscale

    def train_step(self, data):
        x_data, y_data = data
        x = x_data[:, :self.x_dim]
        y = y_data[:, -self.y_dim:]
        z = y_data[:, :self.z_dim]
        y_short = tf.concat([z, y], axis=-1)

        # Forward loss
        with tf.GradientTape() as tape:
            y_out = self.model(x_data)
            pred_loss = self.w1 * self.loss_fit(
                y_data[:, self.z_dim:],
                y_out[:, self.z_dim:])  # [zeros, y] <=> [zeros, yhat]
            output_block_grad = tf.concat(
                [y_out[:, :self.z_dim], y_out[:, -self.y_dim:]],
                axis=-1)  # take out [z, y] only (not zeros)
            latent_loss = self.w2 * self.loss_latent(
                y_short, output_block_grad)  # [z, y] <=> [zhat, yhat]
            forward_loss = pred_loss + latent_loss
        grads_forward = tape.gradient(forward_loss,
                                      self.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads_forward, self.model.trainable_weights))

        # Backward loss
        with tf.GradientTape() as tape:
            x_rev = self.model.inverse(y_data)
            rev_loss = self.w3 * self.loss_factor * self.loss_fit(
                x_rev, x_data)
        grads_backward = tape.gradient(rev_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads_backward, self.model.trainable_weights))

        total_loss = forward_loss + latent_loss + rev_loss
        return {
            'total_loss': total_loss,
            'forward_loss': forward_loss,
            'latent_loss': latent_loss,
            'rev_loss': rev_loss
        }

    def test_step(self, data):
        x_data, y_data = data
        return NotImplementedError


trainer = Trainer(model, x_dim, y_dim, z_dim, tot_dim, n_couple_layer,
                  n_hid_layer, n_hid_dim)
trainer.compile(optimizer='Adam')

LossFactor = UpdateLossFactor(n_epoch)
logger = NBatchLogger(n_display, n_epoch)
hist = trainer.fit(dataset,
                   batch_size=n_batch,
                   epochs=n_epoch,
                   steps_per_epoch=n_data // n_batch,
                   callbacks=[logger, LossFactor],
                   verbose=0)

## CHECK RESULTS ##

fig, ax = plt.subplots(1, facecolor='white', figsize=(8, 5))
ax.plot(hist.history['total_loss'], 'k.-', label='total_loss')
ax.plot(hist.history['forward_loss'], 'b.-', label='forward_loss')
ax.plot(hist.history['latent_loss'], 'g.-', label='latent_loss')
ax.plot(hist.history['rev_loss'], 'r.-', label='inverse_loss')
plt.legend()
plt.show()

z = np.random.multivariate_normal([1.] * z_dim, np.eye(z_dim), y.shape[0])
y = np.concatenate([z, y], axis=-1).astype('float32')
x_pred = model.inverse(y).numpy()

model.save("./models/")
# model.save_weights("./trained_model_weights.h5")

print(x_pred)
print(x_pred.shape)
