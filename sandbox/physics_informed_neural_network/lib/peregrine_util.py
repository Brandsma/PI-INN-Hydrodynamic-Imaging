import os
from pathlib import Path

from lib.logger import setup_logger

log = setup_logger(__name__)

SCRATCHDIR_ENV_KEY = "SCRATCHDIR"


def is_running_on_peregrine():
    """Checks if the program is currently running on Peregrine. Returns True if it is running on Peregrine.

    This relies on a hacky check whether or not the $SCRATCHDIR environment variable is present.
    TODO: Find a better way to check this.
    """
    is_running_on_peregrine = (SCRATCHDIR_ENV_KEY in os.environ)

    if is_running_on_peregrine:
        log.debug("Currently running on Peregrine...")

    return is_running_on_peregrine


def get_scratch_dir(subfolder_path: str = ""):
    """Returns the scratch directory in peregrine, with an optional added subfolder path

    for example, calling this function without parameters would return the scratch directory for a specific job ('scratch/jobs/$JOB_ID') without trailing slash.

    Calling this function with a subfolder path will return the scratch directory for a specific job follow by the subfolder path ('scratch/jobs/$JOB_ID/subfolder_path') without trailing slash.

    :param subfolder_path: path that is appended to the scratch directory
    :returns: scratch directory folder path (optionally including the subfolder path)

    """
    # Find the folder where the data should be found and should be saved
    SCRATCHDIR = os.getenv(SCRATCHDIR_ENV_KEY)
    if SCRATCHDIR is None:
        print(f"{SCRATCHDIR_ENV_KEY} environment variable does not exist")
        exit(1)

    if SCRATCHDIR[-1] == "/":
        SCRATCHDIR = SCRATCHDIR[:-1]

    if subfolder_path != "":
        SCRATCHDIR += subfolder_path

    if SCRATCHDIR[-1] == "/":
        SCRATCHDIR = SCRATCHDIR[:-1]

    return SCRATCHDIR


def ensure_scratch_dir(subfolder_path: str = ""):
    """The same as 'get_scratch_dir' function, but also ensures that the folder exists. If it does not exist, it creates the folder

    :param subfolder_path: path that is appended to the scratch directory
    :returns: scratch directory folder path (optionally including the subfolder path)

    """
    SCRATCHDIR = get_scratch_dir(subfolder_path)
    Path(SCRATCHDIR).mkdir(parents=True, exist_ok=True)
    return SCRATCHDIR
