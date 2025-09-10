from enum import Enum
from typing import Tuple

import numpy as np

from . import hydro, sine


class DataType(Enum):
    """An enumeration for the different types of data."""

    Sine = (1,)
    Hydro = (2,)


def get_data(
    dt: DataType,
    subset: str = "all",
    num_sensors: int = 64,
    shuffle_data: bool = True,
    run: int = -1,
    use_pde: bool = False,
    noise_experiment: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads and returns the specified dataset.

    Args:
    ----
        dt: The type of data to load (Sine or Hydro).
        subset: The subset of the data to use.
        num_sensors: The number of sensors used in the data.
        shuffle_data: Whether to shuffle the data.
        run: The run number to use.
        use_pde: Whether to use the PDE-specific data setup.
        noise_experiment: A flag to indicate if this is a noise experiment.

    Returns:
    -------
        A tuple containing the training data, training labels, test data, and test labels.

    """
    if dt == DataType.Sine:
        return sine.setup_data(shuffle_data)
    elif dt == DataType.Hydro:
        return hydro.setup_data(
            subset,
            shuffle_data,
            num_sensors,
            run,
            use_pde,
            noise_experiment=noise_experiment,
        )
    else:
        print("Warning: Incorrect data type found, returning null")

    return None, None, None, None
