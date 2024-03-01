import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate angle between vectors
    angle = np.arccos(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)))
    
    # Convert angle to degrees
    angle_deg = np.degrees(angle)
    
    return angle_deg

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Create a Tkinter root window
root = tk.Tk()
root.withdraw()  # Hide the root window

# Open a file dialog to select the video file
video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mov *.mp4 *.avi *HEIC *.mkv *.flv *.webm *.wmv *.mpg *.mpeg *.3gp *.ogv *.gif *.ts *.m2ts *.mts *.m4v *.vob *.mxf *.rm *.swf *.amv *.m2v *.m4p *.mp2 *.mpv")])

# Open the video file
cap = cv2.VideoCapture(video_path)

# Variable to store the current frame number
frame_number = 0
paused = False

while cap.isOpened():
    if not paused:
        # Read frames from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose model
        results = pose.process(frame_rgb)

        # Draw landmarks on the frame
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Get landmarks
            landmarks = results.pose_landmarks.landmark

            # List of relevant joints
            relevant_joints = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, 
                               mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_HIP, 
                               mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE,
                               mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, 
                               mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_HIP, 
                               mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE]
            
            # Iterate over each relevant joint
            for landmark in relevant_joints:
                landmark_point = landmarks[landmark.value].x, landmarks[landmark.value].y
                # Display the angle on top of the joint
                cv2.putText(frame, str(round(landmark_point[0], 2)), 
                            (int(landmark_point[0] * frame.shape[1]), int(landmark_point[1] * frame.shape[0]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the frame with reduced width and size
    frame_resized = cv2.resize(frame, (int(frame.shape[1] * 0.2), int(frame.shape[0] * 0.2)))
    cv2.imshow('Pose Detection', frame_resized)

    # Keyboard controls
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord(' '):  # Toggle play/pause with spacebar
        paused = not paused

# Release the VideoCapture and destroy all windows
cap.release()
cv2.destroyAllWindows()
