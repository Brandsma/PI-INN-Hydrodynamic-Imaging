import tensorflow as tf

from flow import *
from utils import *

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

        self.w1 = 1.
        self.w2 = 1.
        self.w3 = 1.
        self.loss_factor = 1.
        self.loss_fit = MSE
        self.loss_latent = MMD_multiscale

    def train_step(self, data):
        x_data, y_data = data
        # x = x_data[:, :self.x_dim]
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

        # Physics informed loss
        # with tf.GradientTape()

        total_loss = forward_loss + latent_loss + rev_loss
        return {
            'total_loss': total_loss,
            'forward_loss': forward_loss,
            'latent_loss': latent_loss,
            'rev_loss': rev_loss
        }

    def test_step(self, data):
        x_data, y_data = data
        print(x_data.shape, y_data.shape)
        return NotImplementedError()
