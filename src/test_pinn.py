from INN.test import run_test_on_model
from INN.inn import INNConfig

from INN.main import simple_run
from INN.data import DataType


def test_pinn():
    config = INNConfig(4, 4, 64, 0, 0, 32, None)

    noise_experiment = True
    if noise_experiment:
        subsets = [
            "low_noise_parallel", "high_noise_parallel",
            "low_noise_saw", "high_noise_saw",
        ]
    else:
        subsets = ["offset", "offset_inverse", "mult_path", "parallel", "far_off_parallel", "sine"]
    sensor_options = [8]

    for subset in subsets:
        for num_sensors in sensor_options:
            # simple_run(DataType.Hydro,
            #           subset=subset,
            #           num_sensors=num_sensors,
            #           use_pde=False,
            #           config=config)
            run_test_on_model(subset=subset, num_sensors=num_sensors, noise_experiment=noise_experiment, test_pinn=True)

if __name__ == '__main__':
    test_pinn()
