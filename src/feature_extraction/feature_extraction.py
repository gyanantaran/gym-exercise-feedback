# author: @vishalpaudel
# %% Imports


from os.path import join
from cv2 import (
    VideoCapture,
    cvtColor,
    COLOR_BGR2RGB,
    waitKey,
    destroyAllWindows,
    CAP_PROP_FPS,
)
from mediapipe import solutions
from numpy import save, array

from time import time

from src.paths.path_manager import npy_location

from src.paths.paths import vid_dir, vid_names

# %% Initialize MediaPipe Pose module

pose_module = solutions.pose.Pose()


# %% Extract landmarks from frame


def _extract_landmarks(frame, pose_module):
    frame_rgb = cvtColor(frame, COLOR_BGR2RGB)
    results = pose_module.process(frame_rgb)
    landmarks = []

    if results.pose_landmarks is not None:
        landmarks = [
            (landmark.x, landmark.y, landmark.z)
            for landmark in results.pose_landmarks.landmark
        ]

    return landmarks


# %% Process the whole video


def _process_video(video_path, pose_module):
    mode = "not-camera"

    video_capture = VideoCapture(video_path)

    # We assume FPS to be 30
    if round(video_capture.get(CAP_PROP_FPS)) != 30:
        print(f"FPS was not 30 for {video_path}")

    # Check if the video file is opened successfully
    if not video_capture.isOpened():
        print(f"Error opening video file {video_path}")

    landmarks_data = []

    if mode == "camera":
        while video_capture.isOpened():
            ret, frame = video_capture.read()

            if not ret:
                break

            landmarks = _extract_landmarks(frame, pose_module)
            landmarks_data.append(landmarks)

            if waitKey(1) & 0xFF == ord("q"):
                break

        video_capture.release()
        destroyAllWindows()

    else:
        while True:
            ret, frame = video_capture.read()

            if not ret:
                break

            landmarks = _extract_landmarks(frame, pose_module)
            landmarks_data.append(landmarks)

        video_capture.release()

    return array(landmarks_data)


# %% Save extracted features(the landmarks) into src.utils.config.landmarks_dir
def save_extracted_features():
    global pose_module

    print("\nStarted saving frame-landmarks!")
    _start_time = time()

    vids = vid_names[vid_names.index("shashank3.mov") :]

    for vid_name in vids:
        _local_time = time()

        test_video = join(vid_dir, vid_name)
        landmarks_array = _process_video(test_video, pose_module)
        save_npy_path = npy_location(vid_name)

        # plot_landmark(landmarks_array[0])
        save(save_npy_path, landmarks_array)  # save landmarks data NumPy file

        _end_time = time()
        _elapsed_time = _end_time - _local_time
        print("saved:", vid_name, f"{_elapsed_time} sec", sep="\t")

    _end_time = time()
    _elapsed_time = _end_time - _start_time

    print(f"Completed! Took {_elapsed_time} sec")
