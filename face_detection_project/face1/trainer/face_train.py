import cv2
import numpy as np
from PIL import Image
import os
import pickle

data_path = "face1/data"
trainer_path = "face1/trainer"
os.makedirs(trainer_path, exist_ok=True)

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faces = []
ids = []
label_map = {}
current_id = 0

for person_name in os.listdir(data_path):
    person_path = os.path.join(data_path, person_name)
    if not os.path.isdir(person_path):
        continue

    label_map[current_id] = person_name
    for img_file in os.listdir(person_path):
        path = os.path.join(person_path, img_file)
        gray_img = Image.open(path).convert('L')
        img_np = np.array(gray_img, 'uint8')
        face_rects = detector.detectMultiScale(img_np)

        for (x, y, w, h) in face_rects:
            faces.append(img_np[y:y+h, x:x+w])
            ids.append(current_id)

    current_id += 1

recognizer.train(faces, np.array(ids))
recognizer.save(os.path.join(trainer_path, "face_detect.yml"))

with open(os.path.join(trainer_path, "labels.pickle"), "wb") as f:
    pickle.dump(label_map, f)

print("✅ Training completed. Model saved as face_detect.yml")
print("🧾 Label mapping:", label_map)
