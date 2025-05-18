import cv2
import os
import pickle

# ==== Configuration ====
DATASET_DIR = "face1/data"
LABELS_FILE = "face1/trainer/labels.pickle"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
SAMPLES_PER_PERSON = 100

# ==== Load or initialize label dictionary ====
if os.path.exists(LABELS_FILE):
    with open(LABELS_FILE, 'rb') as f:
        labels = pickle.load(f)
else:
    labels = {}

# ==== Reverse the dictionary for lookup ====
label_ids = {v: k for k, v in labels.items()}
next_id = max(label_ids.values(), default=-1) + 1



# ==== Get person name ====
name = input("👤 Enter name (exactly as before to add more samples, or new name to register): ").strip()

if name in label_ids:
    person_id = label_ids[name]
    print(f"✅ Found existing person ID: {person_id}")
else:
    person_id = next_id
    labels[person_id] = name
    print(f"🆕 New person registered with ID: {person_id}")
    next_id += 1

# ==== Prepare directory ====
person_dir = os.path.join(DATASET_DIR, name)
os.makedirs(person_dir, exist_ok=True)

# ==== Initialize face detector ====
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
cam = cv2.VideoCapture(1)
print("📸 Starting camera. Press 'q' to quit anytime.")

count = 0
while True:
    ret, frame = cam.read()
    if not ret:
        print("⚠️ Camera error")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))

        count += 1
        file_path = os.path.join(person_dir, f"{name}_{count}.png")
        cv2.imwrite(file_path, face_img)

        # Draw & display
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name}: {count}/{SAMPLES_PER_PERSON}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("🧠 Collecting face samples", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= SAMPLES_PER_PERSON:
        break

# ==== Cleanup ====
cam.release()
cv2.destroyAllWindows()

# ==== Save labels ====
with open(LABELS_FILE, 'wb') as f:
    pickle.dump(labels, f)

print(f"\n✅ Collection complete. {count} images saved for '{name}'.")
