# %% Imports

from os.path import splitext, join
from src.config import landmarks_dir


# %% Split file name


def filename(vid_name):
    file_name, _ = splitext(vid_name)  # remove extension
    return file_name


# %% return .npy location of a `vid_name.mov`


def npy_location(vid_name):
    npy_file_name = filename(vid_name) + ".npy"
    npy_path = join(landmarks_dir, npy_file_name)

    return npy_path

# %% Save