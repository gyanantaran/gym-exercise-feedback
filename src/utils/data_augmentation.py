# %% Imports
# this file intends to augment data
# save the whole data into X.npy, and Y.npy

import numpy as np
import os
from config import vid_names

# %% Config
main_fps = 30
num_new_videos = 2
# target_fps = main_fps // num_new_groups


# %% Seperate Frames
def seperate_frames(num_frames, num_new_groups=num_new_videos):
    num_frames_per_group = -(
        -num_frames // num_new_videos
    )  # this is basically ceil operation

    augm_frame_ids = np.full((num_new_videos, num_frames_per_group), np.NaN)

    for i in range(0, num_new_videos):
        for j in range(0, num_frames_per_group):
            frame_id = i + j * num_new_videos
            if frame_id < num_frames:
                augm_frame_ids[i, j] = frame_id

    return augm_frame_ids


# %% Get Augmented Frames
def get_augm_vids(frames):
    num_frames, num_landmarks, num_dimensions = frames.shape

    augm_frame_ids = seperate_frames(num_frames)
    M, N = augm_frame_ids.shape

    augm_videos = np.zeros((M, N, num_landmarks, num_dimensions))

    for i in range(0, M):
        for j in range(0, N):
            frame = frames[augm_frame_ids[i, j], :, :]
            augm_videos[i, j, :, :] = frame


# %% Finally Save Data
def save_X():
    X = []
    for vid_name in vid_names:
        vid, _ = os.path.splitext(vid_name)
        file_name = vid + ".npy"

        frames = np.load()


if __name__ == "__main__":
    print(seperate_frames(5))
