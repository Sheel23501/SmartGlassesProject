import cv2
import os
import numpy as np
import pickle
from PIL import Image

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
model_path = "face1/trainer/face_detect.yml"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found: {model_path}")
recognizer.read(model_path)

# Load Haar cascade (try local first, fallback to OpenCV's built-in)
local_cascade_path = "face1/trainer/haarcascade_frontalface_default.xml"
if os.path.exists(local_cascade_path):
    cascade_path = local_cascade_path
else:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise IOError(f"❌ Failed to load Haar cascade from {cascade_path}")

# Load label dictionary
labels_path = "face1/trainer/labels.pickle"
if os.path.exists(labels_path):
    with open(labels_path, "rb") as f:
        label_dict = pickle.load(f)
        # Invert dictionary to match IDs -> names
        label_dict = {v: k for k, v in label_dict.items()}
else:
    label_dict = {0: "Unknown"}

# Start webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise IOError("❌ Cannot access the camera.")

print("🔍 Face recognition started. Press 'q' to quit.")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        label_id, confidence = recognizer.predict(roi_gray)
        name = label_dict.get(label_id, "Unknown")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({round(confidence, 2)})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face Recognizer", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
print("jai mata di")