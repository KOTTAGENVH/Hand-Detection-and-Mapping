import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

mp_hand = mp.solutions.hands
hands = mp_hand.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawings = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

@app.route('/process_video', methods=['GET', 'POST'])
def process_video():
    if request.method == 'POST':
        file = request.files['video']
        cap = cv2.VideoCapture(file)
    else:
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

                folded_fingers = []
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    if idx in [3, 6, 10, 14, 18]:
                        tip = np.array([landmark.x, landmark.y, landmark.z])
                        base = np.array([hand_landmarks.landmark[idx - 3].x, hand_landmarks.landmark[idx - 3].y,
                                         hand_landmarks.landmark[idx - 3].z])
                        distance = np.linalg.norm(tip - base)
                        folded_fingers.append(idx)
                        message = f"Finger {idx} distance: {distance:.2f}"
                        return jsonify({"message": message})

                folded_fingers_text = "Folded Fingers: " + ", ".join(map(str, folded_fingers))
                # You can remove the cv2.putText line since we're returning messages
                # cv2.putText(img, folded_fingers_text, (10, 130), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

                index_tip = np.array(
                    [hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y, hand_landmarks.landmark[8].z])
                middle_tip = np.array(
                    [hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y, hand_landmarks.landmark[12].z])
                distance_index_middle = np.linalg.norm(index_tip - middle_tip)
                if distance_index_middle <= 0.10:
                    message = "OK, your grip is correct"
                else:
                    message = "Issue: Your fingers are not close"
                return jsonify({"message": message})

        if results.multi_hand_landmarks is None:
            return jsonify({"message": "No Hand Detected"})

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(debug=True)
