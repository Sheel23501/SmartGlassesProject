import cv2
import pickle

# Load model and cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face1/trainer/face_detect.yml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load labels
with open("face1/trainer/labels.pickle", "rb") as f:
    original_labels = pickle.load(f)
    label_dict = {v: k for k, v in original_labels.items()}
# Start webcam
cam = cv2.VideoCapture(1)
print("🔍 Face recognition started. Press 'q' to quit.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (200, 200))
        label, confidence = recognizer.predict(roi_gray)
        print(f"Detected: Label={label}, Confidence={confidence}")

        if confidence < 65:
            name = label_dict.get(label, "Unknown")
        else:
            name = "Unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({round(confidence,2)})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face Recognizer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
