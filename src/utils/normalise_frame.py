#!/opt/homebrew/bin/python3.11
import numpy as np


def normalise_frame(frames):
    print(frames.shape())

    num_frame = frames.shape[0]

    for frame_id in range(num_frame):
        frame = frames[frame_id, :, :]

        print(frame.shape)
        break


if __name__ == "__main__":
    from config import landmarks_dir
