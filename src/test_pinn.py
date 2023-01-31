from INN.test import run_test_on_model
from INN.inn import INNConfig

from INN.main import simple_run
from INN.data import DataType


def test_pinn():
    config = INNConfig(4, 4, 64, 0, 0, 32, None)

    # subsets = ["offset", "offset_inverse", "mult_path", "parallel", "far_off_parallel"]
    subsets = ["sine"]
    sensor_options = [1,3,8,64]

    for subset in subsets:
        for num_sensors in sensor_options:
            # simple_run(DataType.Hydro, subset=subset, num_sensors=num_sensors, use_pde=True, config=config)
            run_test_on_model(subset=subset, num_sensors=num_sensors, test_pinn=True)

if __name__ == '__main__':
    test_pinn()
