# %% Imports
# this file intends to augment data
# save the whole data into X.npy, and Y.npy

from math import floor

import numpy as np
from numpy import load, arange, where
from numpy.typing import NDArray

from src.config import num_frames_per_new
from src.paths.path_manager import npy_location


# %% Create index array of frames to be seperated


# this one is based on Anit's idea on augmenting by
# taking one 30 frames from across the frame, and make one
# new frame-group do this as many times as the total frames
# the function below returns just the array of indexes for
# modularity and simplicity and SoC.
def sparsify_frames(num_frames: int, num_frames_per_new: int) -> NDArray:
    # dimensions of the new array
    rows = floor(num_frames / num_frames_per_new)
    cols = num_frames_per_new

    # total number of frames in the augmented videos
    total_frames = rows * cols
    indices = arange(total_frames)

    _reshaped_indx = indices.reshape((cols, rows)).T

    nan_val = -1  # index for illegal frames
    augm_indx = where(_reshaped_indx >= num_frames, nan_val, _reshaped_indx)

    return np.array(augm_indx)


# Example usage
result = sparsify_frames(21, 4)
print(result)


# %% Sliding augmentation
def slidify(num_frames, num_frames_per_new):
    sliding = 10
    gap = 3

    window_len = gap * (num_frames_per_new - 1) + 1

    augm_indx = []
    i = 0
    while i < (num_frames - window_len):
        augm_indx.append(np.arange(start=i, stop=(i + window_len), step=gap))
        i += sliding
    return np.array(augm_indx)


print(slidify(1000, 30))

# %% Augment a single video


def augment_frames(vid_name: str):
    # extract the location of the vid_name's extracted features npy
    npy_file = npy_location(vid_name)

    all_frames = load(npy_file)
    # print(f'Loaded {npy_file}')
    num_frames, _, _ = all_frames.shape

    # print(num_frames)

    # indices_matrix = sparsify_frames(
    #     num_frames=num_frames, num_frames_per_new=num_frames_per_new
    # )
    indices_matrix = slidify(num_frames=num_frames, num_frames_per_new=num_frames_per_new)

    augmented_frames = all_frames[indices_matrix]

    # print(augmented_frames.shape)

    return augmented_frames


# Example usage
# result = augment_frames("alli1.mov")


# %% Entry Point

if __name__ == "__main__":
    # print(seperate_frames(5))
    pass
