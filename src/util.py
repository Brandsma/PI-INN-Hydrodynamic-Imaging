import os

import numpy as np

def cartesian_coord(*arrays):
    grid = np.meshgrid(*arrays)
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    return points

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
