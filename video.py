# Project 05: Face Detector - Part 2 (Video Detection)
# Stretch version — eye detection, face counter, FPS counter, smile detection

import cv2 as cv
import time

print("OpenCV imported successfully!")

# ============================================
# Load Trained Data
# ============================================
# Four cascades total. Eyes/smiles are only run *inside* an already-detected
# face rectangle — that's both faster (smaller search area) and more
# accurate (fewer false positives from background clutter).
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier('haarcascade_smile.xml')

for name, cascade in [('face', face_cascade), ('eye', eye_cascade), ('smile', smile_cascade)]:
    if cascade.empty():
        print(f"Error: {name} cascade failed to load. Check the XML filename/path.")
        exit()
print("All cascades loaded successfully!")

# ============================================
# Capture Video
# ============================================
video = cv.VideoCapture(1, cv.CAP_DSHOW)
if not video.isOpened():
    print("Error: Could not open webcam!")
    exit()
print("Webcam started... Press 'Q' to quit")

# FPS tracking — smoothed over the last N frames so the number
# doesn't jitter wildly on screen.
prev_time = time.time()
fps = 0.0

# ============================================
# Video Processing Loop
# ============================================
while True:
    success, frame = video.read()
    if not success:
        print("Error: Could not read frame")
        break

    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw the face box
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Restrict eye/smile search to the region inside this face,
        # in both the grayscale (for detection) and color (for drawing) frames.
        face_gray = gray_img[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]

        # --- Stretch: Eye detection ---
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=10)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

        # --- Stretch: Smile detection ---
        # minNeighbors is set much higher here than for faces/eyes because
        # smile cascades are notoriously trigger-happy on textures like
        # chins and shadows — without this, you get false positives on
        # neutral faces.
        smiles = smile_cascade.detectMultiScale(face_gray, scaleFactor=1.7, minNeighbors=22)
        is_smiling = len(smiles) > 0
        for (sx, sy, sw, sh) in smiles:
            cv.rectangle(face_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

        if is_smiling:
            cv.putText(frame, 'Smiling', (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --- Stretch: Face counter ---
    cv.putText(frame, f'Faces: {len(faces)}', (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- Stretch: FPS counter ---
    current_time = time.time()
    frame_time = current_time - prev_time
    prev_time = current_time
    if frame_time > 0:
        # Simple exponential smoothing so the readout doesn't flicker frame to frame
        instant_fps = 1.0 / frame_time
        fps = fps * 0.9 + instant_fps * 0.1
    cv.putText(frame, f'FPS: {fps:.1f}', (10, 60),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow('Live Face Detector - Stretch', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================
# Cleanup
# ============================================
video.release()
cv.destroyAllWindows()
print("Face detector closed. Goodbye!")
