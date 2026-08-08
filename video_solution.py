# Project 05: Face Detector - Part 2 (Video Detection)
# Complete Solution

import cv2 as cv
print("OpenCV imported successfully!")

# Load the Haar Cascade classifier
trained_face_data = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
print("Face detector loaded successfully!")

# Silence internal OpenCV C++ backend warning noise
import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
if hasattr(cv, 'utils') and hasattr(cv.utils, 'logging'):
    cv.utils.logging.setLogLevel(cv.utils.logging.LOG_LEVEL_SILENT)

# Initialize webcam capture with robust camera selection & MJPEG decoding
def get_working_camera():
    import numpy as np
    import time
    # Index 1 with DirectShow is the physical USB camera on Windows
    attempts = [
        (1, cv.CAP_DSHOW),
        (1, cv.CAP_ANY),
        (0, cv.CAP_DSHOW),
        (0, cv.CAP_ANY),
    ]
    for idx, backend in attempts:
        cap = cv.VideoCapture(idx, backend)
        if not cap.isOpened():
            continue
        # Set Motion-JPEG encoding to prevent YUV decoding artifacts
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Warm up camera sensor for auto-exposure
        valid = False
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                if np.mean(frame) > 10.0 and np.std(frame) > 10.0:
                    valid = True
                    break
            time.sleep(0.03)
        
        if valid:
            backend_name = "CAP_DSHOW" if backend == cv.CAP_DSHOW else "CAP_ANY"
            print(f"Webcam started on camera index {idx} ({backend_name})... Press 'Q' to quit")
            return cap
        cap.release()
    return None

video = get_working_camera()
if video is None or not video.isOpened():
    print("Error: Could not open any working webcam! Make sure no other application is using the camera.")
    exit()

# Video processing loop
while True:
    # Read a frame from the webcam
    success, frame = video.read()
    if not success:
        print("Error: Could not read frame")
        break

    # Convert to grayscale for face detection
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces in the frame
    face_coordinates = trained_face_data.detectMultiScale(gray_img)

    # Draw green rectangles around detected faces
    for (x, y, w, h) in face_coordinates:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display face count on frame
    cv.putText(frame, f'Faces: {len(face_coordinates)}',
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv.imshow('Live Face Detector', frame)

    # Check if 'Q' is pressed to quit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video.release()
cv.destroyAllWindows()
print("Face detector closed. Goodbye!")
