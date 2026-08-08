# Project 05: Face Detector - Part 2 (Video Detection)
# Student Template - Complete the TODOs!

# ============================================
# Phase 1: Import OpenCV
# ============================================
# WORKING EXAMPLE: Import the computer vision library
import cv2 as cv
print("OpenCV imported successfully!")

# ============================================
# Phase 2: Load Trained Data
# ============================================
# WORKING EXAMPLE: Load the Haar Cascade classifier
trained_face_data = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
print("Face detector loaded successfully!")

# ============================================
# Phase 3: Capture Video
# ============================================
# TODO 4: Initialize webcam capture (Easy)
#
# Start capturing video from the webcam.
# Use 0 for default camera, 1 for external camera.
#
# STEPS:
# 1. Create video capture: video = cv.VideoCapture(0)
# 2. Check if camera opened: if not video.isOpened(): print("Error!")
#
# EXPECTED: Webcam turns on, ready to capture frames
#
# YOUR CODE HERE - starter code provided:
video = cv.VideoCapture(1, cv.CAP_DSHOW)
if not video.isOpened() or not video.read()[0]:
    video.release()
    video = cv.VideoCapture(1)
print("Webcam started on camera index 1... Press 'Q' to quit")

# ============================================
# Phase 4 & 5: Video Processing Loop
# ============================================
# TODO 5: Process each video frame (Medium)
#
# Create a loop that continuously captures frames,
# detects faces, and draws rectangles.
#
# STEPS:
# 1. Start while True loop
# 2. Read frame: success, frame = video.read()
# 3. Check if frame captured: if not success: break
# 4. Convert to grayscale: gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
# 5. Detect faces: face_coordinates = trained_face_data.detectMultiScale(gray_img)
# 6. Draw rectangles: for (x, y, w, h) in face_coordinates: cv.rectangle(...)
# 7. Display frame: cv.imshow('Live Face Detector', frame)
#
# EXPECTED: Live video with green rectangles around detected faces
#
# YOUR CODE HERE - starter code provided:
while True:
    # Read a frame from the webcam
    success, frame = video.read()
    if not success:
        print("Error: Could not read frame")
        break

    # TODO: Convert frame to grayscale (store in 'gray_img')
    # HINT: gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # TODO: Detect faces using trained_face_data.detectMultiScale(gray_img)
    # HINT: face_coordinates = trained_face_data.detectMultiScale(gray_img)

    # TODO: Draw green rectangles around each detected face
    # HINT: for (x, y, w, h) in face_coordinates:
    #           cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame (this part works!)
    cv.imshow('Live Face Detector', frame)

    # TODO 6: Add quit functionality (Easy)
    # Break the loop when 'Q' is pressed
    # HINT: if cv.waitKey(1) & 0xFF == ord('q'): break
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================
# Phase 6: Cleanup
# ============================================
# WORKING EXAMPLE: Release resources
video.release()
cv.destroyAllWindows()
print("Face detector closed. Goodbye!")
