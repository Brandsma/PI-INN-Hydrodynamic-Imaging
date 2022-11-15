import tensorflow as tf

from flow import *
from utils import *

from matplotlib import pyplot as plt


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
                 pde_loss_func=None,
                 pde_applied_forward=True,
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
        self.pde_loss_func = pde_loss_func
        self.pde_applied_forward = pde_applied_forward

    def train_step(self, data):
        x_data, y_data = data
        # x = x_data[:, :self.x_dim]
        y = y_data[:, :self.y_dim]
        z = y_data[:, -self.z_dim:]
        y_short = tf.concat([y, z], axis=-1)
        pde_loss = None

        # Forward loss
        with tf.GradientTape() as tape:
            y_out = self.model(x_data)
            # TODO: invert z and y dim
            pred_loss = self.w1 * self.loss_fit(
                y_data[:, :self.z_dim],
                y_out[:, :self.z_dim])  # [zeros, y] <=> [zeros, yhat]
            output_block_grad = tf.concat(
                [y_out[:, -self.z_dim:], y_out[:, :self.y_dim]],
                axis=-1)  # take out [z, y] only (not zeros)
            latent_loss = self.w2 * self.loss_latent(
                y_short, output_block_grad)  # [z, y] <=> [zhat, yhat]

            forward_loss = pred_loss + latent_loss
            if not (self.pde_loss_func is None or not self.pde_applied_forward):
                pde_loss = self.pde_loss_func(self.model, x_data, self.x_dim, self.y_dim)
                forward_loss += pde_loss
        grads_forward = tape.gradient(forward_loss,
                                      self.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads_forward, self.model.trainable_weights))

        # Backward loss
        with tf.GradientTape() as tape:
            x_rev = self.model.inverse(y_data)
            # PDE loss
            # pde_loss = interior_loss(self.model, x_data, self.x_dim, self.y_dim)
            rev_loss = self.w3 * self.loss_factor * self.loss_fit(
                x_rev, x_data)# + pde_loss

            if not (self.pde_loss_func is None or self.pde_applied_forward):
                pde_loss = self.pde_loss_func(self.model, x_data, self.x_dim, self.y_dim)
                rev_loss += pde_loss
        grads_backward = tape.gradient(rev_loss, self.model.trainable_weights)
        # print(grads_backward)
        self.optimizer.apply_gradients(
            zip(grads_backward, self.model.trainable_weights))


        total_loss = forward_loss + latent_loss + rev_loss
        if self.pde_loss_func is None:
            return {
                'total_loss': total_loss,
                'forward_loss': forward_loss,
                'latent_loss': latent_loss,
                'rev_loss': rev_loss,
            }
        else:
            total_loss += pde_loss
            return {
                'total_loss': total_loss,
                'forward_loss': forward_loss,
                'latent_loss': latent_loss,
                'rev_loss': rev_loss,
                'pde_loss': pde_loss,
            }

    def test_step(self, data):
        x_data, y_data = data
        print(x_data.shape, y_data.shape)
        return NotImplementedError()

