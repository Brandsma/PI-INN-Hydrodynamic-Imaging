from typing import Callable
import pickle
from dataclasses import dataclass

import tensorflow as tf
from .flow import NVP


@dataclass
class INNConfig:
    """
    Configuration for the Invertible Neural Network.

    Attributes:
        n_couple_layer: The number of coupling layers in the NVP.
        n_hid_layer: The number of hidden layers in the subnetworks.
        n_hid_dim: The number of hidden units in the subnetworks.
        x_dim: The dimension of the input data.
        y_dim: The dimension of the output data.
        z_dim: The dimension of the latent space.
        pde_loss_func: The PDE loss function (optional).
    """

    n_couple_layer: int
    n_hid_layer: int
    n_hid_dim: int
    x_dim: int
    y_dim: int
    z_dim: int
    pde_loss_func: Callable

    def to_file(self, filename: str = "./models/INNConfig.pkl"):
        """Saves the configuration to a file."""
        with open(filename, "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def from_file(filename: str = "./models/INNConfig.pkl") -> "INNConfig":
        """Loads the configuration from a file."""
        with open(filename, "rb") as inp:
            return pickle.load(inp)

    def __hash__(self) -> int:
        return hash(repr(self))


def create_model(
    tot_dim: int, n_couple_layer: int, n_hid_layer: int, n_hid_dim: int
) -> tf.keras.Model:
    """
    Creates the NVP model.

    Args:
        tot_dim: The total dimension of the model's input.
        n_couple_layer: The number of coupling layers.
        n_hid_layer: The number of hidden layers in the subnetworks.
        n_hid_dim: The number of hidden units in the subnetworks.

    Returns:
        The created NVP model.
    """
    model = NVP(tot_dim, n_couple_layer, n_hid_layer, n_hid_dim, name="NVP")
    x = tf.keras.Input((tot_dim,))
    model(x)
    return model
