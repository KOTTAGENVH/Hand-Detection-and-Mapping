import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="./movenet_lightning.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Capture video from the camera
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    input_data = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_data = (input_data.astype(np.float32) - 127.5) / 127.5  # Scale to [-1, 1]
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Overlay detected keypoints on the frame
    for keypoints in output_data:
        for keypoint in keypoints:
            # Check confidence score for each keypoint element
            if keypoint[2] > 0.5:  # Adjust the threshold as needed
                # Draw a circle at each keypoint
                cv2.circle(frame, (int(keypoint[1]), int(keypoint[0])), 5, (0, 255, 0), -1)  # Green circle
    
    # Display the frame
    cv2.imshow("Pose Detection", frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
