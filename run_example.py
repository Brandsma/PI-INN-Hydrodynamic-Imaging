import argparse
import sys

sys.path.append("src")

from src.INN.data import get_data, DataType
from src.INN.main import run_inn
from src.INN.inn import INNConfig
from src.INN import hydro, sine


def run_inn_example():
    """
    Runs a minimal example of the Invertible Neural Network (INN) on synthetic sine wave data.
    """
    print("--- Running INN Example ---")
    data, labels, _, _ = get_data(DataType.Sine, shuffle_data=True)

    config = INNConfig(
        n_couple_layer=4,
        n_hid_layer=4,
        n_hid_dim=128,
        x_dim=data.shape[1],
        y_dim=labels.shape[1],
        z_dim=31,  # z_dim needs to make y_dim + z_dim even
        pde_loss_func=None,
    )

    run_inn(
        given_data=data,
        given_labels=labels,
        config=config,
        n_batch=8,
        n_epoch=4,  # Use a small number of epochs for a quick example
        datatype=DataType.Sine,
        model_dir="./trained_models_example",
    )
    print("--- INN Example Complete ---")


def run_pinn_example():
    """
    Runs a minimal example of the Physics-Informed Invertible Neural Network (PINN)
    on synthetic sine wave data.
    """
    print("\n--- Running PINN Example ---")
    data, labels, _, _ = get_data(DataType.Sine, shuffle_data=True)

    config = INNConfig(
        n_couple_layer=4,
        n_hid_layer=4,
        n_hid_dim=128,
        x_dim=data.shape[1],
        y_dim=labels.shape[1],
        z_dim=31,  # z_dim needs to make y_dim + z_dim even
        pde_loss_func=sine.interior_loss,
    )

    run_inn(
        given_data=data,
        given_labels=labels,
        config=config,
        n_batch=8,
        n_epoch=4,  # Use a small number of epochs for a quick example
        datatype=DataType.Sine,
        model_dir="./trained_models_example",
    )
    print("--- PINN Example Complete ---")


if __name__ == "__main__":
    run_inn_example()
    run_pinn_example()
