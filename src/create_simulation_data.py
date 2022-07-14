import argparse
import math
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lib.logger import LOGGING_LEVELS, set_global_logging_level, setup_logger
from lib.peregrine_util import ensure_scratch_dir, is_running_on_peregrine

log = setup_logger(__name__)

## Velocity Profiles ##


def wavelet_e(p):
    return (1 - 2 * p**2) / ((1 + p**2)**(5 / 2))


def wavelet_o(p):
    return (-3 * p) / ((1 + p**2)**(5 / 2))


def wavelet_n(p):
    return (2 - p**2) / ((1 + p**2)**(5 / 2))


def v_x(s, x, y, theta, a, norm_w):
    p = (s - x) / y
    C = (norm_w * a**3) / (2 * y**3)
    return C * (wavelet_o(p) * math.sin(theta) -
                wavelet_e(p) * math.cos(theta))


def v_y(s, x, y, theta, a, norm_w):
    p = (s - x) / y
    C = (norm_w * a**3) / (2 * y**3)
    return C * (wavelet_n(p) * math.sin(theta) -
                wavelet_o(p) * math.cos(theta))


## Simulation ##


def simulate(theta=0,
             a=10,
             norm_w=10,
             sensor_range=(-200, 200),
             number_of_sensors=64,
             x_range=(-500, 500),
             y_range=(0, 500),
             number_of_x_steps=1024,
             number_of_y_steps=1,
             simulation_area_offset=75,
             number_of_runs=32,
             add_noise=True,
             noise_power=1.5e-5,
             forward_and_backward_runs=False,
             folder_path="../../data/simulation/"):

    input_sensors = list(
        np.linspace(sensor_range[0], sensor_range[1], num=number_of_sensors))
    x_input = list(np.linspace(x_range[0], x_range[1], num=number_of_x_steps))
    y_input = list(
        np.linspace(y_range[0] + simulation_area_offset,
                    y_range[1] + simulation_area_offset,
                    num=number_of_y_steps))
    time_step = abs(x_input[1] - x_input[0]) / norm_w
    start_time = 0

    all_data = []
    all_labels = []
    all_timestamp = []
    for _ in tqdm(range(number_of_runs)):
        if forward_and_backward_runs:
            x_input = list(reversed(x_input))
        time = start_time
        data = []
        labels = []
        timestamp = []
        for y_idx, y in enumerate(y_input):
            data.append([])
            labels.append([])
            timestamp.append([])
            for x_idx, x in enumerate(x_input):
                data[y_idx].append([])
                labels[y_idx].append([])
                timestamp[y_idx].append([])
                for input_sensor in input_sensors:
                    # NOTE: the x and y coordinates are different than the array coordinates
                    data[y_idx][x_idx].append(
                        v_x(input_sensor, x, y + 1, theta, a, norm_w))
                    data[y_idx][x_idx].append(
                        v_y(input_sensor, x, y + 1, theta, a, norm_w))

                labels[y_idx][x_idx].append(x)
                labels[y_idx][x_idx].append(y + 1)
                timestamp[y_idx][x_idx].append(time)
                time += time_step

                if add_noise:
                    data[y_idx][x_idx] += np.random.normal(
                        0, noise_power, len(data[y_idx][x_idx]))
        all_data.append(data)
        all_labels.append(labels)
        all_timestamp.append(timestamp)

    data_path = folder_path / f"a{a}_normw{norm_w}_theta{theta}.npy"
    labels_path = folder_path / f"a{a}_normw{norm_w}_theta{theta}_labels.npy"
    timestamp_path = folder_path / f"a{a}_normw{norm_w}_theta{theta}_timestamp.npy"

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    all_timestamp = np.array(all_timestamp)

    all_data = np.reshape(all_data, (all_data.shape[0], all_data.shape[1] *
                                     all_data.shape[2], all_data.shape[3]))
    all_labels = np.reshape(all_labels,
                            (all_labels.shape[0], all_labels.shape[1] *
                             all_labels.shape[2], all_labels.shape[3]))
    all_timestamp = np.reshape(
        all_timestamp, (all_timestamp.shape[0], all_timestamp.shape[1] *
                        all_timestamp.shape[2], all_timestamp.shape[3]))

    log.debug(all_data.shape)
    log.debug(all_labels.shape)
    log.debug(all_timestamp.shape)
    np.save(data_path, all_data)
    np.save(labels_path, all_labels)
    np.save(timestamp_path, all_timestamp)


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Simulation program that 'moves' a sphere through water and measures the movement of the water at discrete locations (i.e. at the sensor locations). The water movement is described using the velocity profiles, which are derivatives of the Velocity Potential function."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default="../data/simulation_data/",
        help="Folder where the results of the simulation should be stored.")
    parser.add_argument(
        "--without-noise",
        action='store_true',
        help=
        "Normally the simulation adds Gaussian white noise, this can be turned off with this flag."
    )
    parser.add_argument(
        "--noise-power",
        type=int,
        default=1.5e-5,
        help=
        "The standard deviation of the Gaussian White noise that is added to the simulation. Default is 1.5e-5 based on existing sensors (see bot et al.)."
    )
    parser.add_argument("--forward-and-backward-runs",
                        action='store_true',
                        help="Half of the runs will be reversed runs.")
    parser.add_argument(
        "--number-of-runs",
        type=int,
        default=32,
        help="The number of separate runs the simulation should perform")
    parser.add_argument(
        "--simulation-area-offset",
        type=int,
        default=75,
        help=
        "Offset of the area in which the sphere moves as compared to the sensor array (on the y-axis)"
    )
    parser.add_argument("--size",
                        type=int,
                        default=10,
                        help="The size of the sphere (in mm)")
    parser.add_argument("--speed",
                        type=int,
                        default=10,
                        help="The speed of the sphere (in mm/s)")
    parser.add_argument(
        "--theta",
        type=int,
        default=0,
        help=
        "Azimuth angle under which the sphere moves (relative to sensor array)"
    )

    parser.add_argument(
        "--number_of_sensors",
        type=int,
        default=64,
        help=
        "Number of discrete locations that are measured. Equidistantly spaced in the sensor range."
    )

    parser.add_argument(
        "--number_of_x_steps",
        type=int,
        default=1024,
        help=
        "Number of separate positions for the sphere along the x-axis equidistantly spaced in the x range.",
    )
    parser.add_argument(
        "--number_of_y_steps",
        type=int,
        default=1,
        help=
        "Number of separate positions for the sphere along the y-axis equidistantly spaced in the y range.",
    )

    parser.add_argument(
        "--sensor-range",
        help=
        "The range (in mm) along which the sensors are placed (equidistantly). default is (-200, 200). example usage: \"--sensor-range -200 200\"",
        type=int,
        nargs=2,
        default=[-200, 200])
    parser.add_argument(
        "--x-range",
        help=
        "The range (in mm) along which the sphere is moved for the specified axis. default is (-500, 500). example usage: \"--x-range -500 500\"",
        type=int,
        nargs=2,
        default=[-500, 500])
    parser.add_argument(
        "--y-range",
        help=
        "The range (in mm) along which the sphere is moved for the specified axis. Note that this is still offset by the simulation offset. default is (0, 500). example usage: \"--y-range 0 500\"",
        type=int,
        nargs=2,
        default=[0, 500])
    parser.add_argument("--logging-level",
                        help=("Provide logging level. "
                              "Example --log debug', default='info'"),
                        choices=LOGGING_LEVELS.keys(),
                        default="info",
                        type=str.lower)

    return parser.parse_args()


def main():
    # Get arguments from command line
    args = parse_args()
    set_global_logging_level(LOGGING_LEVELS[args.logging_level])

    log.debug(f"Running program with following parameters: {args}")

    # Ensure
    if is_running_on_peregrine():
        folder_path = ensure_scratch_dir(subfolder_path="/data/")
    else:
        log.debug("Running locally (getting output folder from arguments)...")
        folder_path = args.output_dir
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    # Replace any old data files
    # TODO: Maybe add a --force flag to remove old files, instead of just removing them
    old_data_files = os.listdir(folder_path)
    for old_data_file in old_data_files:
        if old_data_file.endswith(".npy"):
            os.remove(os.path.join(folder_path, old_data_file))

    simulate(theta=args.theta,
             a=args.size,
             norm_w=args.speed,
             sensor_range=args.sensor_range,
             number_of_sensors=args.number_of_sensors,
             x_range=args.x_range,
             y_range=args.y_range,
             number_of_x_steps=args.number_of_x_steps,
             number_of_y_steps=args.number_of_y_steps,
             simulation_area_offset=args.simulation_area_offset,
             number_of_runs=args.number_of_runs,
             add_noise=not args.without_noise,
             noise_power=args.noise_power,
             forward_and_backward_runs=args.forward_and_backward_runs,
             folder_path=folder_path)

    log.debug("-- DONE --")


if __name__ == "__main__":
    main()
