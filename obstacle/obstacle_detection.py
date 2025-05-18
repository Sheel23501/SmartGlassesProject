import cv2
import pyttsx3
import time
import os

# Text-to-speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Load model
prototxt_path = "obstacle/mobilenet_ssd/deploy.prototxt"
model_path = "obstacle/mobilenet_ssd/mobilenet_iter_73000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Classes + known object widths (in cm)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

KNOWN_WIDTHS = {
    "person": 50,
    "chair": 40,
    "bottle": 7,
    "car": 180,
    "dog": 60,
    "bus": 250
}

# Calibrated focal length (in pixels)
FOCAL_LENGTH = 600  

# Create directory to store learning data (if needed)
LEARN_DIR = "obstacle/learned_objects"
os.makedirs(LEARN_DIR, exist_ok=True)

# Start webcam
cam = cv2.VideoCapture(1)
print("🚀 Smart obstacle detection started. Press 'q' to quit.")

last_spoken = ""
last_time = time.time()

while True:
    ret, frame = cam.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    spoken = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")
            x_center = (startX + endX) // 2
            box_height = endY - startY
            box_width = endX - startX

            # Size description
            if box_height > 180:
                size_desc = "large"
            elif box_height > 100:
                size_desc = "medium"
            else:
                size_desc = "small"

            # Position direction and suggestion
            if x_center < w // 3:
                direction = "left"
                suggestion = "please step right"
            elif x_center > 2 * w // 3:
                direction = "right"
                suggestion = "please step left"
            else:
                direction = "front"
                suggestion = "please stop or duck"

            # Distance estimation
            if label in KNOWN_WIDTHS:
                real_width = KNOWN_WIDTHS[label]
                distance_cm = (real_width * FOCAL_LENGTH) / box_width
                distance_m = distance_cm / 100
                distance_text = f"{distance_m:.1f} meters"
            else:
                distance_text = "distance unknown"

            # Draw box and text
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {distance_text}", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Speak smart feedback
            if not spoken and "unknown" not in distance_text:
                text = f"{size_desc.capitalize()} object on the {direction}, {distance_text} ahead. {suggestion}."
                print("🔊", text)
                engine.say(text)
                engine.runAndWait()
                spoken = True

    # Display frame
    cv2.imshow("Smart Obstacle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
