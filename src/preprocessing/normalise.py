# %% Imports
import numpy as np


# %% Normalise a frame, TODO
def normalise_frames(frames):

    # print(frames.shape)
    num_frame = frames.shape[0]

    normalised_frames = []
    for frame_id in range(num_frame):
        frame = frames[frame_id, :]

        means = np.mean(frame, axis=0)
        stds = np.std(frame, axis=0)

        mean_shifted = frame - means
        std_scaled = mean_shifted / stds

        normalised_frames.append(std_scaled)

    return np.array(normalised_frames)


if __name__ == "__main__":
    pass
