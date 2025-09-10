import argparse
import sys
from pathlib import Path

from ..lib import LSTM, params
from ..lib.logger import setup_logger

log = setup_logger(__name__)


def train_lstm(
    data: params.Data,
    settings: params.Settings,
    trained_models_folder: str,
):
    """
    Initializes, trains, and saves an LSTM network.

    Args:
        data: The data object containing train and test sets.
        settings: The settings object for the LSTM network.
        trained_models_folder: The directory to save the trained model.
    """
    # Initiate the LSTM network using data and settings
    network = LSTM.LSTM_network(data, settings)
    network.model.summary()
    # Train the network
    network.train()

    # Save the network for later use
    Path(trained_models_folder).mkdir(parents=True, exist_ok=True)
    log.info(f"Saving trained model to {trained_models_folder}/{settings.name}...")
    network.save_model(f"{trained_models_folder}/{settings.name}")


def main():
    """
    Main function to parse command-line arguments and run the LSTM training.
    """
    parser = argparse.ArgumentParser(description="Train an LSTM model.")
    parser.add_argument(
        "--train_loc",
        type=str,
        default="../../data/simulation_data/combined.npy",
        help="Path to the training data file.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="../../data/trained_models/LSTM/",
        help="Directory to save the trained model.",
    )
    parser.add_argument("--n_nodes", type=int, default=128, help="Number of nodes in the LSTM layer.")
    parser.add_argument("--n_epochs", type=int, default=16, help="Number of training epochs.")
    parser.add_argument("--window_size", type=int, default=16, help="Window size for the LSTM.")
    parser.add_argument("--stride", type=int, default=2, help="Stride for the window.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout ratio.")
    args = parser.parse_args()

    # Load settings
    settings = params.Settings(
        window_size=args.window_size,
        stride=args.stride,
        n_nodes=args.n_nodes,
        alpha=0.05,  # Learning rate
        decay=1e-9,
        n_epochs=args.n_epochs,
        shuffle_data=True,
        data_split=0.8,
        dropout_ratio=args.dropout,
        train_location=args.train_loc,
        ac_fun="relu",
    )

    # Load data
    data = params.Data(settings, args.train_loc)
    data.normalize()

    # Train the model
    train_lstm(data, settings, args.model_dir)


if __name__ == "__main__":
    main()
