# Project 05: Face Detector - Part 1 (Image Detection)
# Complete Solution

import cv2 as cv

print("OpenCV imported successfully!")

# Load the Haar Cascade classifier
trained_face_data = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
print("Face detector loaded successfully!")

# Load the image
img = cv.imread('outing.jpg')
if img is None:
    print("Error: Could not load image 'outing.jpg'!")
    exit()

# Convert image to grayscale
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
print(f"Found {len(face_coordinates)} face(s) in the image.")

# Draw green rectangles around faces
for (x, y, w, h) in face_coordinates:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display result
cv.imshow('Face Detector - Image', img)
cv.waitKey(0)
cv.destroyAllWindows()
