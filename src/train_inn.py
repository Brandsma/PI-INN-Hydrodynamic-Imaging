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

    noise_experiment = False

    if noise_experiment:
        subsets = [
            "low_noise_parallel", "medium_noise_parallel", "high_noise_parallel",
            "low_noise_saw", "medium_noise_saw", "high_noise_saw",
        ]
    else:
        subsets = [
            "offset", "offset_inverse", "mult_path", "parallel", "far_off_parallel", "sine"
        ]

    # subsets = ["sine"]
    sensor_options = [8]

    for subset in subsets:
        for num_sensors in sensor_options:
            simple_run(DataType.Hydro,
                       subset=subset,
                       num_sensors=num_sensors,
                       use_pde=False,
                       noise_experiment=noise_experiment,
                       config=config)


if __name__ == '__main__':
    train_inn()
