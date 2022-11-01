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
        self.loss_pde = MSE

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
        # print(grads_backward)
        self.optimizer.apply_gradients(
            zip(grads_backward, self.model.trainable_weights))

        # TODO: Separate this input beforehand
        x_input = x_data[:, 0]
        y_input = x_data[:, 1]
        # theta_input = x_data[:, 2]
        # Physics informed loss
        # TODO: Possibly add this in the forward loss calculation
        # NOTE: also see https://github.com/deepmorzaria/Physics-Informed-Neural-Network-PINNs---TF-2.0/blob/master/PINNs_2.ipynb
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_input)
            tape.watch(y_input)
            vx_y_out = self.model(x_input)[0]
            vy_y_out = self.model(y_input)[1]

        # TODO: Don't do entire output with respect to partial input (entire output is both V_x and V_y)

        vx_only_out = vx_y_out[:, -self.y_dim:][:, 0::2]
        vy_only_out = vy_y_out[:, -self.y_dim:][:, 1::2]

        v_x = tape.gradient(vx_only_out, x_input)
        v_y = tape.gradient(vy_only_out, y_input)

        # print(f"{v_x=} {v_y=}")

        pde_loss_output = pde(v_x, v_y, x_input, y_input)
        pde_loss = self.w1 * self.loss_pde(pde_loss_output,
                                           tf.zeros(pde_loss_output.shape))

        print(pde_loss_output)
        print(pde_loss)
        exit()

        grads_pde = tape.gradient(pde_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients((zip(grads_pde,
                                            self.model.trainable_weights)))

        del tape

        total_loss = forward_loss + latent_loss + rev_loss + pde_loss
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


def pde(dvx_x, dvy_y, x, y, W=25):
    # TODO: find the W from somewhere
    # dvx_x = dde.grad.jacobian(y, x, i=0, j=0)
    # dvy_y = dde.grad.jacobian(y, x, i=1, j=1)

    # x1, y1 = x[:, 0:1], x[:, 1:2]
    # # vx1, vy1 = y[:, 0:1], y[:, 1:2]
    # # TODO: This should use the normalized (i.e. sensor-relative positions)
    return dvx_x + dvy_y + ((3 * (W * x + W * y)) / ((x**2 + y**2)**(5 / 2)))
