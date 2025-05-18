import cv2
import os
import numpy as np
from PIL import Image

# Load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face1/trainer/face_detect.yml")

# Load Haar cascade for face detection
cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Label mapping (must match what you printed earlier)
label_dict = {0: "sheel"}

# Start the webcam
cam = cv2.VideoCapture(0)
print("🔍 Face recognition started. Press 'q' to quit.")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

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
