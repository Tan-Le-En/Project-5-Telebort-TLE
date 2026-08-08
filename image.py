# Project 05: Face Detector - Part 1 (Image Detection)
# Completed version

# ============================================
# Phase 1: Import OpenCV
# ============================================
import cv2 as cv
print("OpenCV imported successfully!")

# ============================================
# Phase 2: Load Trained Data
# ============================================
# TODO 1: Load the Haar Cascade classifier
trained_face_data = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
print("Face detector loaded!")  # confirms the XML parsed without errors

# ============================================
# Phase 3: Image Processing
# ============================================
# TODO 2: Process the image
img = cv.imread('outing.jpg')
print(f"Image loaded: {img.shape}")

# Resize — smaller images mean detectMultiScale has fewer pixels to scan,
# so this is mostly a speed optimization, not an accuracy one.
scale = 0.2
width = int(img.shape[1] * scale)
height = int(img.shape[0] * scale)
img = cv.resize(img, (width, height))
print(f"Image resized to: {width}x{height}")

# Convert to grayscale. Haar cascades were trained on intensity/edge
# patterns (light-dark transitions like eyebrows, nose bridge, mouth),
# not color — grayscale strips out the extra channel data the detector
# doesn't use, which also speeds up the sliding-window scan.
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# ============================================
# Phase 4: Face Detection
# ============================================
# TODO 3: Detect faces and draw rectangles
face_coordinates = trained_face_data.detectMultiScale(gray_img)
print(f"Found {len(face_coordinates)} face(s) in the image")

for (x, y, w, h) in face_coordinates:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# ============================================
# Phase 5: Display Results
# ============================================
cv.imshow('Face Detector - Image', img)
print("Press any key to close...")
cv.waitKey(0)
cv.destroyAllWindows()

print("\n--- Face detection complete! ---")
