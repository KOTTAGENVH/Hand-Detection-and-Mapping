import cv2
from openpose import pyopenpose as op

# Configure OpenPose
params = {
    "model_folder": "path/to/openpose/models",  # Path to the OpenPose models folder
    "hand": False,  # Disable hand estimation for faster processing
    "face": False,  # Disable face estimation for faster processing
    "number_people_max": 1,  # Only detect one person
    "display": 0,  # Disable display for faster processing
}

# Initialize OpenPose
openpose = op.WrapperPython()
openpose.configure(params)
openpose.start()

# Open camera
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame with OpenPose
    datum = op.Datum()
    datum.cvInputData = frame
    openpose.emplaceAndPop([datum])

    # Display the pose estimation
    cv2.imshow("Pose Estimation", datum.cvOutputData)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
