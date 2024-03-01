import cv2
import mediapipe as mp
import numpy as np

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

# Initialize VideoCapture with camera 0
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read frames from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Pose model
    results = pose.process(frame_rgb)

    # Draw landmarks on the frame
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Get landmarks
        landmarks = results.pose_landmarks.landmark

        # Iterate over each landmark
        for landmark in mp_pose.PoseLandmark:
            # Check if the landmark is one of the relevant joints
            if landmark in [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST]:
                landmark_point = landmarks[landmark.value].x, landmarks[landmark.value].y
                # Calculate angle at current landmark
                angle = calculate_angle(landmarks[landmark.value].x, landmarks[landmark.value].y, landmarks[landmark.value].z)
                # Display the angle on top of the joint
                cv2.putText(frame, str(round(angle, 2)), 
                            (int(landmarks[landmark.value].x * frame.shape[1]), int(landmarks[landmark.value].y * frame.shape[0]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
    # Show the frame
    cv2.imshow('Pose Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and destroy all windows
cap.release()
cv2.destroyAllWindows()
