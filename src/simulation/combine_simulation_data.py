import os
import sys

if __name__ == "__main__":
    sys.path.insert(1, os.path.join(sys.path[0], '..'))

import argparse
import glob
import os
from pathlib import Path

import numpy as np
from lib.logger import LOGGING_LEVELS, set_global_logging_level, setup_logger
from lib.peregrine_util import ensure_scratch_dir, is_running_on_peregrine

log = setup_logger(__name__)


def find_files(folder_path: str):
    base_names = []
    for filename in glob.glob(f"{folder_path}/*_theta0.npy"):
        base_names.append(filename)

    return base_names


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Simulation program that 'moves' a sphere through water and measures the movement of the water at discrete locations (i.e. at the sensor locations). The water movement is described using the velocity profiles, which are derivatives of the Velocity Potential function."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default="../../data/simulation_data/",
        help="Folder where the results of the simulation were stored.")

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

    if is_running_on_peregrine():
        folder_path = ensure_scratch_dir(subfolder_path="/data/")
    else:
        log.debug("Running locally (getting output folder from arguments)...")
        folder_path = args.input_dir
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    # Find the names of the files to be combined
    base_names = find_files(folder_path)

    # Start making the combined data from the first found data
    current_filename = os.path.splitext(base_names.pop(0))
    all_data = np.load(current_filename[0] + current_filename[1])
    all_labels = np.load(f"{current_filename[0]}_labels{current_filename[-1]}")
    all_timestamp = np.load(
        f"{current_filename[0]}_timestamp{current_filename[-1]}")

    # For each data found, add it to the total runs
    for name in base_names:
        base_name = os.path.splitext(name)
        labels = np.load(f"{base_name[0]}_labels{base_name[-1]}")
        data = np.load(name)
        timestamp = np.load(f"{base_name[0]}_timestamp{base_name[-1]}")

        all_labels = np.append(all_labels, labels, axis=0)
        all_data = np.append(all_data, data, axis=0)
        all_timestamp = np.append(all_timestamp, timestamp, axis=0)

    log.debug(all_data.shape)
    log.debug(all_labels.shape)
    log.debug(all_timestamp.shape)

    # Save it to disk
    result_filename = "combined"
    result_ext = "npy"
    log.info(
        f"Saving to {result_filename}_(..., labels, timestamp).{result_ext}")

    np.save(f"{folder_path}{result_filename}.{result_ext}", all_data)
    np.save(f"{folder_path}{result_filename}_labels.{result_ext}", all_labels)
    np.save(f"{folder_path}{result_filename}_timestamp.{result_ext}",
            all_timestamp)


if __name__ == '__main__':
    main()
