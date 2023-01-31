import numpy as np
import os

import matplotlib.pyplot as plt


def main(dt="offset"):
    print("Loading data...")
    file_location = f"../../../data/simulation_data/{dt}/combined.npy"

    base_name = os.path.splitext(file_location)
    labels = np.load(f"{base_name[0]}_labels{base_name[-1]}")
    data = np.load(file_location)
    timestamp = np.load(f"{base_name[0]}_timestamp{base_name[-1]}")
    volumes = np.load(f"{base_name[0]}_volumes{base_name[-1]}")

    plt.plot(labels[0, :, 0], labels[0, :, 1], label=f"{dt} Path")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dt = "sine"
    main(dt)
