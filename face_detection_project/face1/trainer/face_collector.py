import cv2
import os

def collect_faces():
    user_name = input("Enter your name: ")
    user_path = os.path.join("face1/data", user_name)
    os.makedirs(user_path, exist_ok=True)

    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        print("❌ Cannot open webcam")
        return

    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    count = 0
    print("📸 Collecting face samples. Press 'q' to quit early.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y+h, x:x+w]
            file_path = os.path.join(user_path, f"{count}.jpg")
            cv2.imwrite(file_path, face)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{user_name}_{count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Face Collector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Collected {count} face samples for {user_name}.")

if __name__ == "__main__":
    collect_faces()
