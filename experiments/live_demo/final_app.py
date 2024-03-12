import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from numpy import arange, array
from cv2 import cvtColor, COLOR_BGR2RGB
from mediapipe import solutions
from tensorflow.keras.models import load_model

pose_module = solutions.pose
threshold = 0.887

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

best_model = load_model("best_model.h5")  # Load the best model saved by ModelCheckpoint

def _evaluate_model(X_test):
    X_test_reshaped = X_test.reshape((X_test.shape[0], 30, 33 * 3))
    predictions = best_model.predict(X_test_reshaped)
    final_predictions = predictions[:, -1, 0]
    prediction = final_predictions > threshold
    st.write('prediction:', prediction.shape)
    st.write(prediction)

    true_count, false_count = sum(prediction), len(prediction) - sum(prediction)

    if true_count > false_count:
        st.write("Most common value: True")
    elif true_count < false_count:
        st.write("Most common value: False")
    else:
        st.write("Equal number of True and False values")

    return None

def main():
    st.title("Live Video Recorder with Pose Estimation")

    if "recording" not in st.session_state:
        st.session_state.recording = False
        st.session_state.final_features = []

    start_button = st.button("Start Recording")
    stop_button = st.button("Stop Recording")
    video_placeholder = st.empty()
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture(0)
    landmarks_data = []
    current_features = []
    sliding = 25
    gap = 2
    window_len = gap * (30 - 1) + 1
    slide_index = 0
    text = "Hello"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    thickness = 2
    position = (50, 200)

    while not stop_button:
        if start_button:
            st.session_state.recording = True
            start_button = False

        if st.session_state.recording:
            ret, frame = cap.read()
            results, landmarks = _extract_landmarks(frame, pose)

            if len(landmarks) > 0:
                if len(landmarks_data) < window_len:
                    landmarks_data.append(landmarks)
                else:
                    if slide_index == 0:
                        indexes = arange(start=0, stop=(0 + window_len), step=gap)
                        current_features = array(landmarks_data)[indexes]
                        st.session_state.final_features.append(current_features)
                    slide_index = (slide_index + 1) % sliding
                    landmarks_data.pop(0)
                    landmarks_data.append(landmarks)

            if results.pose_landmarks:
                landmarks = [(landmark.x, landmark.y) for landmark in results.pose_landmarks.landmark]
                for x, y in landmarks:
                    h, w, _ = frame.shape
                    x, y = int(x * w), int(y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, pose_module.POSE_CONNECTIONS)
                cv2.putText(frame, text, position, font, font_scale, font_color, thickness)

            video_placeholder.image(frame, channels="BGR", use_column_width=True)

    st.write("Set of frames appended final: ", len(st.session_state.final_features))
    text = _evaluate_model(np.array(st.session_state.final_features))
    st.write('text:', text)

if __name__ == "__main__":
    main()
