import cv2
import pyttsx3
import time
import os
import random

# ========== Simulated Ultrasonic Sensor ==========
def get_simulated_distance_cm():
    return random.randint(30, 300)  # Between 30cm and 300cm

# ========== TTS Setup ==========
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ========== Load SSD Model ==========
prototxt = "obstacle/mobilenet_ssd/deploy.prototxt"
model = "obstacle/mobilenet_ssd/mobilenet_iter_73000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Camera
cam = cv2.VideoCapture(0)
print("🚀 Running obstacle detection with voice guidance...")

last_spoken_time = time.time()
spoken_recently = ""

while True:
    ret, frame = cam.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,
                                 (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")
            x_center = (startX + endX) // 2

            # Simulated distance from ultrasonic sensor
            distance_cm = get_simulated_distance_cm()
            distance_m = distance_cm / 100.0

            # Object size logic
            width_px = endX - startX
            size_type = "small"
            if width_px > w // 3:
                size_type = "large"
            elif width_px > w // 4:
                size_type = "medium"

            # Direction
            if x_center < w // 3:
                direction = "left"
                escape = "step right"
            elif x_center > 2 * w // 3:
                direction = "right"
                escape = "step left"
            else:
                direction = "front"
                escape = "stop or step back"

            msg = f"{size_type.capitalize()} object on the {direction}, {distance_m:.1f} meters ahead. Please {escape}."

            if time.time() - last_spoken_time > 3 and msg != spoken_recently:
                print("🔊", msg)
                engine.say(msg)
                engine.runAndWait()
                last_spoken_time = time.time()
                spoken_recently = msg

            # Draw box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, msg, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Smart Glasses View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
