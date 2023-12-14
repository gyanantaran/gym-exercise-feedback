from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_landmark(cur_frame):
    for landmark in range(0, 33):
        landmark_test = cur_frame[landmark][:2]
        plt.ylim(-2, 2)
        plt.xlim(-2, 2)
        plt.plot(landmark_test[0], landmark_test[1], "o")


def plot_landmarks(all_frames, win_title):
    fig = plt.figure(f"Frames for {win_title}")

    def update(frame_index):
        plt.clf()
        plot_landmark(all_frames[frame_index, :, :])
        plt.title(f"Frame Index {frame_index}")

    num_frames = all_frames.shape[0]
    animation = FuncAnimation(fig, update, frames=num_frames, repeat=False)
    plt.show()


if __name__ == "__main__":
    from numpy import load
    from numpy.random import choice, randint
    from os.path import splitext, join

    from src.config import vid_names, landmarks_dir
    from src.preprocessing.normalise import normalise_frames

    chosen_vid = splitext(choice(vid_names))[0]
    file_name = f"{chosen_vid}.npy"
    file_path = join(landmarks_dir, file_name)
    frames = load(file_path)

    frame_index = randint(0, frames.shape[0])
    frame = frames[frame_index, :, :]

    normalised_frames = normalise_frames(frames)

    # plot_landmarks(frames, chosen_vid)
    plot_landmarks(normalised_frames, chosen_vid)
