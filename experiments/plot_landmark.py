import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open the video file
video_capture = cv2.VideoCapture("../data/gym clean data/par1.mov")

# Lists to store the time and landmark 0 (nose) x and y coordinates
timestamps = []
landmark_0_x = []
landmark_0_y = []

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # Get the current time (in seconds)
    time = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # Convert the frame to RGB format for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect poses in the frame
    results = pose.process(frame_rgb)

    if results.pose_landmarks is not None:
        # Extract landmark 0 (nose) coordinates
        landmark_0 = results.pose_landmarks.landmark[0]

        # Append data to lists
        timestamps.append(time)
        landmark_0_x.append(landmark_0.x)
        landmark_0_y.append(landmark_0.y)

    # Exit the video by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

# Calculate the y-axis tick positions for both subplots
y_min_x = min(landmark_0_x)
y_max_x = max(landmark_0_x)
y_min_y = min(landmark_0_y)
y_max_y = max(landmark_0_y)

# Calculate the common y-axis tick gap
y_tick_gap = 0.1  # Adjust this value as needed

# Calculate the y-axis tick positions
y_ticks_x = list(np.arange(y_min_x, y_max_x + y_tick_gap, y_tick_gap))
y_ticks_y = list(np.arange(y_min_y, y_max_y + y_tick_gap, y_tick_gap))

# Set Seaborn style
sns.set(style="whitegrid")

# Plot time vs x and time vs y with the same y-axis tick gaps
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
sns.lineplot(x=timestamps, y=landmark_0_x)
plt.title("Time vs Landmark 0 (Nose) X-coordinate")
plt.xlabel("Time (seconds)")
plt.ylabel("X-coordinate")
plt.yticks(y_ticks_x)

plt.subplot(2, 1, 2)
sns.lineplot(x=timestamps, y=landmark_0_y)
plt.title("Time vs Landmark 0 (Nose) Y-coordinate")
plt.xlabel("Time (seconds)")
plt.ylabel("Y-coordinate")
plt.yticks(y_ticks_y)

plt.tight_layout()
plt.show()
