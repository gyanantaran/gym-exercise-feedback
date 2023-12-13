# %% Imports
# this file intends to augment data
# save the whole data into X.npy, and Y.npy

from math import floor
from os.path import join
from numpy import load, arange, where, array, concatenate
from numpy.typing import NDArray

from config import vid_names, landmarks_dir, train_test_dir, num_frames_per_new
from path_manager import npy_location


# %% Create index array of frames to be seperated


# this one is based on Anit's idea on augmenting by
# taking one 30 frames from accross the frame, and make one
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

    return augm_indx


# Example usage
# result = one_frame_per_frame_group(21, 4)
# print(result)


# %% Augment a single video


def augment_frames(vid_name: str):
    # extract the location of the vid_name's extracted features npy
    npy_file = npy_location(vid_name)

    all_frames = load(npy_file)
    num_frames, _, _ = all_frames.shape

    print(num_frames)

    indices_matrix = sparsify_frames(
        num_frames=num_frames, num_frames_per_new=num_frames_per_new
    )

    augmented_frames = all_frames[indices_matrix]

    print(augmented_frames.shape)

    return augmented_frames


# Example usage
# result = augment_frames("alli1.mov")


# %% Save X.npy


def save_X() -> None:
    file_name = "X.npy"
    dir_loc = train_test_dir
    save_loc = join(dir_loc, file_name)

    num_vids = len(vid_names)
    X = array([])
    for vid_name in vid_names:
        augmeneted_frames = augment_frames(vid_name=vid_name)
        concatenate(X, augmeneted_frames)

    return


# %% Entry Point

if __name__ == "__main__":
    # print(seperate_frames(5))
    pass
