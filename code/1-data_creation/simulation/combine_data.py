import os
from pathlib import Path

import numpy as np

def find_files(folder_path: str):
    # a_set = [10, 20, 30, 40, 50]
    # w_set = [10, 20, 30, 40, 50]
    a_set = [10, 20]
    w_set = [10]

    base_names = []

    for a in a_set:
        for w in w_set:
            base_names.append(f"a{a}_normw{w}_theta0.npy")

    for idx in range(len(base_names)):
        base_names[idx] = folder_path + base_names[idx]

    return base_names

def get_scratch_dir():
    # Find the folder where the data should be found and should be saved
    data_folder_key = "SCRATCHDIR"
    SCRATCHDIR = os.getenv(data_folder_key)
    if SCRATCHDIR is None:
        print(f"{data_folder_key} environment variable does not exist")
        exit(1)
    if SCRATCHDIR[-1] == "/":
        SCRATCHDIR = SCRATCHDIR[:-1]

    return SCRATCHDIR

def main():
    SCRATCHDIR = get_scratch_dir()
    folder_path = f"{SCRATCHDIR}/data/"
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

    print(all_data.shape)
    print(all_labels.shape)
    print(all_timestamp.shape)

    # Save it to disk
    result_filename = "combined"
    result_ext = "npy"
    print(f"Saving to {result_filename}_(data, labels, timestamp).{result_ext}")

    np.save(f"{folder_path}{result_filename}.{result_ext}", all_data)
    np.save(f"{folder_path}{result_filename}_labels.{result_ext}", all_labels)
    np.save(f"{folder_path}{result_filename}_timestamp.{result_ext}", all_timestamp)


if __name__ == '__main__':
    main()
