
import numpy as np
from typing import Tuple
from enum import Enum

import sine
import hydro

class DataType(Enum):
    Sine = 1,
    Hydro = 2,

def get_data(dt: DataType, subset="all", shuffle_data=True, run=-1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if dt == DataType.Sine:
        return sine.setup_data(shuffle_data)
    elif dt == DataType.Hydro:
        return hydro.setup_data(subset, shuffle_data, run)
    else:
        print("Warning: Incorrect data type found, returning null")

    return None, None, None, None
