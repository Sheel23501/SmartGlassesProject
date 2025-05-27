import os
import cv2
import numpy as np

# Paths
dataset_path = "face1/data"
haar_cascade_path = "haarcascade_frontalface_default.xml"
trainer_path = "trainer"
model_save_path = os.path.join(trainer_path, "face-train.yml")

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Create LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Create training data
faces = []
labels = []
label_map = {}  # name -> id
current_id = 0

for root, dirs, files in os.walk(dataset_path):
    for folder in dirs:
        folder_path = os.path.join(root, folder)
        label = folder
        if label not in label_map:
            label_map[label] = current_id
            current_id += 1
        id_ = label_map[label]

        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(folder_path, filename)
                image = cv2.imread(img_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

                for (x, y, w, h) in faces_rects:
                    roi = gray[y:y + h, x:x + w]
                    faces.append(roi)
                    labels.append(id_)

# Train the recognizer
recognizer.train(faces, np.array(labels))

# Save the model
if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

recognizer.save(model_save_path)

# Optional: save label map for later decoding
import pickle
with open(os.path.join(trainer_path, "labels.pickle"), "wb") as f:
    pickle.dump(label_map, f)

print("✅ Training complete. Model saved to:", model_save_path)
print("jai mata di")