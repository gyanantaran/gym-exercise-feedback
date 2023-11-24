import cv2
import mediapipe as mp

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize MediaPipe Drawing module for annotations
mp_drawing = mp.solutions.drawing_utils

# Open the video file
video_capture = cv2.VideoCapture("../data/gym clean data/par1.mov")

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

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

        # Display the annotated frame
        cv2.imshow("Annotated Video", annotated_frame)

    # Exit the video by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
