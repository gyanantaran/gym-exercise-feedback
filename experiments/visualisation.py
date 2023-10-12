import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Define the video path and open the video capture
video_path = "/Users/vpaudel343/Desktop/par1.mov"
# video_path = "../data/original_quality/2023-09-24/vis1.mp4"
video_capture = cv2.VideoCapture(video_path)

# Initialize Matplotlib figures
fig = plt.figure(figsize=(8, 6))
grid = fig.add_gridspec(2, 2)

ax1, ax2, ax3 = (
    fig.add_subplot(grid[0, 1:]),
    fig.add_subplot(grid[1, 1:]),
    fig.add_subplot(grid[0:, 0]),
)
ax2.set_xlabel("Time")
ax2.set_ylabel("Y Cordinate")
ax1.set_ylabel("X Cordinate")
ax1.set_ylim(0, 1)
ax2.set_ylim(0, 1)
ax1.grid()
ax2.grid()
ax1.set_yticks(np.ceil(np.linspace(0, 856, 10)))
ax2.set_yticks(np.ceil(np.linspace(0, 1528, 10)))
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax2.set_yticklabels([])

fig.tight_layout()

# Prepare lists to store data
timestamps = []
landmark_n_x = []
landmark_n_y = []
landmark_num = 23
landmark_name = "left hip"

# Constants for time control
start_time = 4
time_gaps = 5

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    time = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # Convert the frame to RGB format for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect poses in the frame
    results = pose.process(frame_rgb)

    if results.pose_landmarks is not None:
        # Annotate the frame with pose landmarks
        annotated_frame = frame.copy()
        mp_drawing.draw_landmarks(
            annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        landmark_n = results.pose_landmarks.landmark[landmark_num]

        # Append data to lists
        timestamps.append(time)
        landmark_n_x.append(landmark_n.x)
        landmark_n_y.append(landmark_n.y)

        image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Update the plots
        ax1.plot(timestamps[-2:], [image.shape[1] * i for i in landmark_n_x[-2:]], "b")
        ax2.plot(
            timestamps[-2:],
            [image.shape[0] * (i * -1 + 1) for i in landmark_n_y[-2:]],
            "b",
        )

        ax1.set_xlim(timestamps[-1] - time_gaps, timestamps[-1] + time_gaps)
        ax2.set_xlim(timestamps[-1] - time_gaps, timestamps[-1] + time_gaps)

        ax3.clear()
        ax3.imshow(image)
        ax3.scatter(
            image.shape[1] * landmark_n_x[-1],
            image.shape[0] * landmark_n_y[-1],
            label=f"{landmark_name}",
        )
        ax3.set_xlabel("Pose annotated image")
        ax3.set_yticks([])  # Remove y-ticks
        ax3.set_xticks([])  # Remove x-ticks
        ax3.spines["top"].set_visible(False)  # Remove top spine
        ax3.spines["right"].set_visible(False)  # Remove right spine
        ax3.spines["bottom"].set_visible(False)  # Remove bottom spine
        ax3.spines["left"].set_visible(False)  # Remove left spine

        ax3.legend(loc="upper left")

        plt.pause(0.01)  # Add a small delay to display the frame

# Release video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
