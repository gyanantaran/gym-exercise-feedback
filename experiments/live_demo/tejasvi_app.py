import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

def main():
    st.title("Live Video Recorder with Pose Estimation")

    # Create a button to start and stop the video recording
    start_button = st.button("Start Recording")
    stop_button = st.button("Stop Recording")

    # Create a placeholder for the video feed
    video_placeholder = st.empty()

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # OpenCV video capture
    cap = cv2.VideoCapture(0)
    recording = False
    last_frame = None
    landmarks_list = []

    while start_button:
        # Check if the "Start Recording" button is pressed
        if start_button:
            recording = True
            start_button = False

        while recording and not stop_button:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform pose estimation on the frame
            results = pose.process(rgb_frame)

            # Display the pose landmarks on the frame
            if results.pose_landmarks:
                landmarks = [(landmark.x, landmark.y) for landmark in results.pose_landmarks.landmark]
                landmarks_list.append(landmarks)

                for x, y in landmarks:
                    h, w, _ = frame.shape
                    x, y = int(x * w), int(y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Display the frame with pose landmarks
            video_placeholder.image(frame, channels="BGR", use_column_width=True)

            # Save the last captured frame
            last_frame = frame.copy()

            # Check if the "Stop Recording" button is pressed
            if stop_button:
                recording = False

        # Release the video capture object when recording is stopped
        cap.release()

    # Print the landmarks after stopping the recording
    if landmarks_list:
        st.write("Pose Landmarks:")
        for idx, landmarks in enumerate(landmarks_list):
            st.write(f"Frame {idx + 1}: {landmarks}")


if __name__ == "_main_":
    main()