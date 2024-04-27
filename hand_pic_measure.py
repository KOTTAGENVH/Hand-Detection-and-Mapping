import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # Import mp_drawing

def detect_hand(image_path):
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Unable to read the image file.")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            # Draw hand landmarks on the image
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Convert image to PIL format for displaying in Tkinter
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            return image, results.multi_hand_landmarks[0]
        else:
            return None, None
    except Exception as e:
        print("Error:", e)
        return None, None

def calculate_distance(landmarks, point1, point2):
    x1, y1 = landmarks.landmark[point1].x, landmarks.landmark[point1].y
    x2, y2 = landmarks.landmark[point2].x, landmarks.landmark[point2].y
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def open_file():
    file_path = filedialog.askopenfilename()
    print("Selected file:", file_path)  # Add this line for debugging
    if file_path:
        image, landmarks = detect_hand(file_path)
        if image and landmarks:
            # Calculate distance between thumb and index finger
            thumb_tip = 4
            index_finger_tip = 8
            distance_thumb_index = calculate_distance(landmarks, thumb_tip, index_finger_tip)
            print("Distance between thumb and index finger:", distance_thumb_index, "pixels")
            
            # Resize image to fit within window dimensions
            window_width = 800  # Adjust as needed
            window_height = 600  # Adjust as needed
            image = np.array(image.resize((window_width, window_height)))
            
            img_label.image = ImageTk.PhotoImage(image=Image.fromarray(image))
            img_label.config(image=img_label.image)
            img_label.pack()
        else:
            result_label.config(text="No hand detected in the image.")
    else:
        result_label.config(text="No file selected.")  # Add this line for feedback

# Initialize Tkinter window
root = tk.Tk()
root.title("Hand Detection App")
root.geometry("800x600")  # Set width and height of the window

# Create a label for displaying the result
result_label = tk.Label(root, text="")
result_label.pack()

# Create a button for uploading an image
upload_button = tk.Button(root, text="Upload Image", command=open_file)
upload_button.pack()

# Create a label for displaying the image
img_label = tk.Label(root)
img_label.pack()

# Start the Tkinter event loop
root.mainloop()

# Release MediaPipe Hands
hands.close()
