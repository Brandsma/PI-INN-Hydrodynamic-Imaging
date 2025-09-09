import tensorflow as tf
import numpy as np
from typing import Callable, Dict

from .flow import *
from .utils import *

from matplotlib import pyplot as plt


class Trainer(tf.keras.Model):
    """
    A custom Keras model trainer for the INN/PINN.

    This trainer handles the custom loss function and training loop for the
    Invertible Neural Network. The loss is composed of three main parts:
    1.  Forward Loss: Reconstruction loss for the forward pass (y -> y_hat).
    2.  Latent Loss: MMD loss to ensure the latent space follows a Gaussian distribution.
    3.  Reverse Loss: Reconstruction loss for the reverse pass (x -> x_hat).
    4.  PDE Loss (optional): Physics-informed loss based on the PDE residual.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        x_dim: int,
        y_dim: int,
        z_dim: int,
        pde_loss_func: Callable = None,
        pde_applied_forward: bool = True,
        **kwargs
    ):
        super(Trainer, self).__init__(**kwargs)
        self.model = model
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        self.loss_fit = MSE
        self.loss_latent = MMD_multiscale
        self.pde_loss_func = pde_loss_func
        self.pde_applied_forward = pde_applied_forward

        # Loss weights
        self.w_forward = 1.0
        self.w_latent = 1.0
        self.w_reverse = 1.0
        self.w_pde = 1.0

    def train_step(self, data: tf.Tensor) -> Dict[str, tf.Tensor]:
        x_data, y_data = data

        with tf.GradientTape() as tape:
            # Forward pass
            y_out = self.model(x_data)
            y_true = y_data[:, : self.y_dim]
            y_pred = y_out[:, : self.y_dim]
            forward_loss = self.w_forward * self.loss_fit(y_true, y_pred)

            # Latent space loss
            z_true = y_data[:, -self.z_dim :]
            z_pred = y_out[:, -self.z_dim :]
            latent_loss = self.w_latent * self.loss_latent(z_true, z_pred)

            # Reverse pass
            x_rev = self.model.inverse(y_data)
            reverse_loss = self.w_reverse * self.loss_fit(x_rev, x_data)

            # PDE loss
            pde_loss = 0.0
            if self.pde_loss_func:
                pde_loss = self.w_pde * self.pde_loss_func(
                    self.model, x_data, self.x_dim, self.y_dim
                )

            total_loss = forward_loss + latent_loss + reverse_loss + pde_loss

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        metrics = {
            "total_loss": total_loss,
            "forward_loss": forward_loss,
            "latent_loss": latent_loss,
            "rev_loss": reverse_loss,
        }
        if self.pde_loss_func:
            metrics["pde_loss"] = pde_loss

        return metrics

    def test_step(self, data: tf.Tensor) -> Dict[str, tf.Tensor]:
        x_data, y_data = data

        # Forward pass
        y_out = self.model(x_data)
        y_true = y_data[:, : self.y_dim]
        y_pred = y_out[:, : self.y_dim]
        forward_loss = self.w_forward * self.loss_fit(y_true, y_pred)

        # Latent space loss
        z_true = y_data[:, -self.z_dim :]
        z_pred = y_out[:, -self.z_dim :]
        latent_loss = self.w_latent * self.loss_latent(z_true, z_pred)

        # Reverse pass
        x_rev = self.model.inverse(y_data)
        reverse_loss = self.w_reverse * self.loss_fit(x_rev, x_data)

        # PDE loss
        pde_loss = 0.0
        if self.pde_loss_func:
            pde_loss = self.w_pde * self.pde_loss_func(
                self.model, x_data, self.x_dim, self.y_dim
            )

        total_loss = forward_loss + latent_loss + reverse_loss + pde_loss

        return {"loss": total_loss}
