import os
import sys

if __name__ == "__main__":
    sys.path.insert(1, os.path.join(sys.path[0], ".."))

import argparse
from pathlib import Path

from lib.logger import LOGGING_LEVELS, set_global_logging_level, setup_logger
from lib.peregrine_util import ensure_scratch_dir, is_running_on_peregrine
from lib.util import coords

from simulation import simulate

log = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine the simulation data into one npy object. Useful when having various permutations of sphere sizes and speeds."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default="../../data/simulation_data/",
        help="Folder where the results of the simulation should be stored.",
    )
    parser.add_argument(
        "--without-noise",
        action="store_true",
        help="Normally the simulation adds Gaussian white noise, this can be turned off with this flag.",
    )
    parser.add_argument(
        "--noise-power",
        type=float,
        default=1.5e-5,
        help="The standard deviation of the Gaussian White noise that is added to the simulation. Default is 1.5e-5 based on existing sensors (see bot et al.).",
    )
    parser.add_argument(
        "--forward-and-backward-runs",
        action="store_true",
        help="Half of the runs will be reversed runs.",
    )
    parser.add_argument(
        "--number-of-runs",
        type=int,
        default=32,
        help="The number of separate runs the simulation should perform",
    )
    parser.add_argument(
        "--simulation-area-offset",
        type=int,
        default=75,
        help="Offset of the area in which the sphere moves as compared to the sensor array (on the y-axis)",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs="+",
        default=[10],
        help="The size(s) of the sphere (in mm)",
    )
    parser.add_argument(
        "--speed",
        type=int,
        nargs="+",
        default=[10],
        help="The speed(s) of the sphere (in mm/s)",
    )
    parser.add_argument(
        "--theta",
        type=int,
        default=0,
        help="Azimuth angle under which the sphere moves (relative to sensor array)",
    )

    parser.add_argument(
        "--number_of_sensors",
        type=int,
        default=64,
        help="Number of discrete locations that are measured. Equidistantly spaced in the sensor range.",
    )

    parser.add_argument(
        "--number_of_steps",
        type=int,
        default=1024,
        help="Number of total separate positions for the sphere along the path. Equidistantly spaced for each subpath (between two points) with the length of the subpath being equal to 'NUM_STEPS // (len(PATH) - 1)'.",
    )

    parser.add_argument(
        "--sensor-range",
        help='The range (in mm) along which the sensors are placed (equidistantly). default is (-200, 200). example usage: "--sensor-range -200 200"',
        type=int,
        nargs=2,
        default=[-200, 200],
    )
    parser.add_argument(
        "--path",
        help="The points along which the path of the sphere will run. example usage: '--path -500,0 500,0'",
        type=float,
        nargs="+",
        default=[-500, 0, 500, 0],
    )
    parser.add_argument(
        "--logging-level",
        help=("Provide logging level. " "Example --log debug', default='info'"),
        choices=LOGGING_LEVELS.keys(),
        default="info",
        type=str.lower,
    )

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

    # # Replace any old data files
    # # TODO: Maybe add a --force flag to remove old files, instead of just removing them
    # old_data_files = os.listdir(folder_path)
    # for old_data_file in old_data_files:
    #     if old_data_file.endswith(".npy"):
    #         os.remove(os.path.join(folder_path, old_data_file))

    # Setup the sphere path (points)
    if len(args.path) < 4 or len(args.path) % 2 != 0:
        raise Exception(
            "Incorrect format of sphere point path was given. Should be at least 4 values and even number"
        )
    points = []
    for idx in range(len(args.path) // 2):
        points.append([args.path[idx * 2], args.path[idx * 2 + 1]])

    count = 1
    for w in args.speed:
        for a in args.size:
            log.info(
                f"Running simulation {count}/{len(args.speed) * len(args.size)} with (a = {a}, w = {w})..."
            )
            simulate(
                theta=args.theta,
                a=a,
                norm_w=w,
                sensor_range=args.sensor_range,
                number_of_sensors=args.number_of_sensors,
                points=points,
                number_of_steps=args.number_of_steps,
                simulation_area_offset=args.simulation_area_offset,
                number_of_runs=args.number_of_runs,
                add_noise=not args.without_noise,
                noise_power=args.noise_power,
                forward_and_backward_runs=args.forward_and_backward_runs,
                folder_path=folder_path,
            )
            count += 1

    log.debug("-- DONE --")


if __name__ == "__main__":
    main()
