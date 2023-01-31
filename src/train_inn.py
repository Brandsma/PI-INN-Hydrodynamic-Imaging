from INN.test import run_test_on_model
from INN.inn import INNConfig

from INN.main import simple_run
from INN.data import DataType

def train_inn_with_config(config, use_pde=False):
    simple_run(DataType.Hydro,
              subset="offset",
              num_sensors=64,
              use_pde=False,
              config=config)

def train_inn():
    config = INNConfig(4, 4, 64, 0, 0, 32, None)

    # subsets = [
    #     "offset", "offset_inverse", "mult_path", "parallel", "far_off_parallel"
    # ]
    subsets = ["sine"]
    sensor_options = [1, 3, 8, 64]

    for subset in subsets:
        for num_sensors in sensor_options:
            simple_run(DataType.Hydro,
                      subset=subset,
                      num_sensors=num_sensors,
                      use_pde=False,
                      config=config)


if __name__ == '__main__':
    train_inn()
