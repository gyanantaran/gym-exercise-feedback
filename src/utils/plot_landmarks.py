from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_landmark(frame):
    for landmark in range(0, 33):
        landmark_test = frame[landmark][:2]
        # plt.ylim(0, 1)
        # plt.xlim(0.4, 0.6)
        plt.plot(landmark_test[0], landmark_test[1], "o")


def plot_landmarks(frames, win_title):
    fig = plt.figure(f"Frames for {win_title}")

    def update(frame_index):
        plt.clf()
        plot_landmark(frames[frame_index, :, :])
        plt.title(f"Frame Index {frame_index}")

    num_frames = frames.shape[0]
    animation = FuncAnimation(fig, update, frames=num_frames, repeat=False)
    plt.show()


if __name__ == "__main__":
    from numpy import load
    from numpy.random import choice, randint
    from os.path import splitext, join

    from config import vid_names, landmarks_dir
    from normalise_frame import normalise_frame

    chosen_vid = splitext(choice(vid_names))[0]
    file_name = f"{chosen_vid}.npy"
    file_path = join(landmarks_dir, file_name)
    frames = load(file_path)

    frame_index = randint(0, frames.shape[0])
    frame = frames[frame_index, :, :]

    normalised_frame = normalise_frame(frame)

    plot_landmarks(frames, chosen_vid)
