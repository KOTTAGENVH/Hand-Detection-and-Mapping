import cv2
import mediapipe as mp
import time
import numpy as np

mp_hand = mp.solutions.hands

# Initialize MediaPipe Hands with maximum num_hands parameter set to 1.
hands = mp_hand.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawings = mp.solutions.drawing_utils

mp_drawing_styles = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

cap = cv2.VideoCapture(0) # Change based on your camera (0 or 1 or 2 or ...)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height

prev_time = 0

while True:

    success, img = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        continue

    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawings.draw_landmarks(
                img, hand_landmarks,
                mp_hand.HAND_CONNECTIONS,
                mp_drawings.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                mp_drawings.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
            )

            # Detect folded fingers
            folded_fingers = []  # List to store folded fingers
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # Calculate the distance between finger tip and base
                if idx in [3, 6, 10, 14, 18]:  # Indices of thumb, index, middle, ring, and pinky tips
                    tip = np.array([landmark.x, landmark.y, landmark.z])
                    base = np.array([hand_landmarks.landmark[idx - 3].x, hand_landmarks.landmark[idx - 3].y, hand_landmarks.landmark[idx - 3].z])
                    distance = np.linalg.norm(tip - base)
                
                    # Check if finger is folded (distance between tip and base is smaller than a threshold)
                    if distance < 0.03:
                        folded_fingers.append(idx)

                    # Display distance between finger tip and base
                    cv2.putText(img, f'{distance:.2f}', (int(tip[0] * img.shape[1]), int(tip[1] * img.shape[0])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

                    # Calculate distance between adjacent fingertips
                    next_tip_idx = (idx + 1) % 21
                    next_tip = np.array([hand_landmarks.landmark[next_tip_idx].x, hand_landmarks.landmark[next_tip_idx].y, hand_landmarks.landmark[next_tip_idx].z])
                    distance_next = np.linalg.norm(tip - next_tip)

                    # Display distance between current finger and the next one
                    cv2.putText(img, f'{distance_next:.2f}', (int((tip[0] + next_tip[0]) * img.shape[1] / 2), int((tip[1] + next_tip[1]) * img.shape[0] / 2)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

            # Display folded fingers
            folded_fingers_text = "Folded Fingers: " + ", ".join(map(str, folded_fingers))
            cv2.putText(img, folded_fingers_text, (10, 130), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

            # Check the distance between index and middle fingers
            index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y, hand_landmarks.landmark[8].z])
            middle_tip = np.array([hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y, hand_landmarks.landmark[12].z])
            distance_index_middle = np.linalg.norm(index_tip - middle_tip)
            if distance_index_middle <= 0.10:
                cv2.putText(img, "OK, your grip is correct", (10, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Issue: Your fingers are not close", (10, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cur_time = time.time()
    FPS = int(1 / (cur_time - prev_time))
    prev_time = cur_time

    if results.multi_hand_landmarks is None:
        cv2.putText(img, "No Hand Detected", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    cv2.putText(img, f'FPS: {FPS}', (10, 170), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("Hand Detection", img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
