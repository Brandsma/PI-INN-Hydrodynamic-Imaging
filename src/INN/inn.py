from INN.flow import NVP
from typing import Callable

import tensorflow.keras as tfk
from dataclasses import dataclass
import pickle

@dataclass
class INNConfig:
    n_couple_layer: int
    n_hid_layer: int
    n_hid_dim: int
    x_dim: int
    y_dim: int
    z_dim: int
    pde_loss_func: Callable

    def to_file(self, filename="./models/INNConfig.pkl"):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def from_file(filename="./models/INNConfig.pkl"):
        with open(filename, 'rb') as inp:
            return pickle.load(inp)

    def __hash__(self):
        return hash(repr(self))


def create_model(
        tot_dim, n_couple_layer, n_hid_layer, n_hid_dim
):
    model = NVP(tot_dim, n_couple_layer, n_hid_layer, n_hid_dim, name='NVP')
    x = tfk.Input((tot_dim, ))
    model(x)
    # model.summary()
    return model
