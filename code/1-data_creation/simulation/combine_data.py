import os

import numpy as np


def main():
    folder_path = "../../../data/"
    base_names = [
        "a10_normw10_theta0.npy",
        "a10_normw20_theta0.npy",
        "a10_normw30_theta0.npy",
        "a10_normw40_theta0.npy",
        "a10_normw50_theta0.npy",
    ]
    for idx in range(len(base_names)):
        base_names[idx] = folder_path + base_names[idx]

    current_filename = os.path.splitext(base_names.pop(0))
    all_data = np.load(current_filename[0] + current_filename[1])
    all_labels = np.load(f"{current_filename[0]}_labels{current_filename[-1]}")
    all_timestamp = np.load(
        f"{current_filename[0]}_timestamp{current_filename[-1]}")

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

    print("Saving to a10_theta0.npy")

    np.save("../../../data/a10_theta0.npy", all_data)
    np.save("../../../data/a10_theta0_labels.npy", all_labels)
    np.save("../../../data/a10_theta0_timestamp.npy", all_timestamp)


if __name__ == '__main__':
    main()
