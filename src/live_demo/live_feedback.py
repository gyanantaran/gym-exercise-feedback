# author: @vishalpaudel
# %% Imports


from os.path import join

import cv2
from cv2 import (
    VideoCapture,
    cvtColor,
    COLOR_BGR2RGB,
    waitKey,
    destroyAllWindows,
    CAP_PROP_FPS,
)
from mediapipe import solutions
from numpy import save, array, arange

from time import time

from src.paths.path_manager import npy_location

from src.paths.paths import vid_dir, vid_names

from src.config import num_frames_per_new

# %% Initialize MediaPipe Pose module

default_pose_module = solutions.pose

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

    return results, landmarks


# %% Evaluate the frames on the model
best_model = load_model("best_model.h5")  # Load the best model saved by ModelCheckpoint
print("Model loaded!")

def _evaluate_model(X_test):

    # X_test = np.expand_dims(X_test, axis=0)
    X_test_reshaped = X_test.reshape((X_test.shape[0], 30, 33 * 3))
    print('reshaped shape', X_test_reshaped.shape)

    # Assuming X_test_reshaped is your test data
    predictions = best_model.predict(X_test_reshaped)
    # print(predictions)

    # Extract the last prediction for each sequence
    final_predictions = predictions[:, -1, 0]

    # Print the shape of final_predictions
    prediction = final_predictions > threshold
    print(prediction)
    print(type(prediction))
    print("Shape of final_predictions:", prediction.shape)

    return None


# %% Process the whole video

def _process_video(video_path, pose_module):
    pose = pose_module.Pose()
    mode = "camera"

    video_capture = VideoCapture(video_path)

    # We assume FPS to be 30
    if round(video_capture.get(CAP_PROP_FPS)) != 30:
        print(f"FPS was not 30 for {video_path}")

    # Check if the video file is opened successfully
    if not video_capture.isOpened():
        print(f"Error opening video file {video_path}")

    landmarks_data = []
    current_features = []
    final_features = []

    sliding = 10
    gap = 3
    window_len = gap * (num_frames_per_new - 1) + 1

    slide_index = 0

    while video_capture.isOpened():
        # one frame read
        ret, frame = video_capture.read()

        if not ret:
            break

        results, landmarks = _extract_landmarks(frame, pose)
        if len(landmarks) > 0:
            if len(landmarks_data) < window_len:
                landmarks_data.append(landmarks)
            else:
                if slide_index == 0:
                    # once in every slide
                    indexes = arange(start=0, stop=(0 + window_len), step=gap)
                    current_features = array(landmarks_data)[indexes]
                    print(current_features.shape)

                    final_features.append(current_features)

                slide_index = (slide_index + 1) % sliding

                # changing the horizon
                landmarks_data.pop(0)
                landmarks_data.append(landmarks)

        print("Another frame down")

        if results.pose_landmarks:
            solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, pose_module.POSE_CONNECTIONS)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Original', frame)

        if waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    pose.close()
    destroyAllWindows()

    print(array(final_features).shape)



_process_video(0, default_pose_module)


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
