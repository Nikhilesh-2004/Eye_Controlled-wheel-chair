import cv2
import dlib
import numpy as np
import os

# --- Setup Folders ---
DATASET_PATH = "dataset_eye"
LABELS = ["left", "right", "center", "closed"]

for label in LABELS:
    path = os.path.join(DATASET_PATH, label)
    if not os.path.exists(path):
        os.makedirs(path)

# --- Dlib Setup ---
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
except RuntimeError:
    print("Error: shape_predictor_68_face_landmarks.dat not found.")
    exit()

cap = cv2.VideoCapture(0)

count = {label: 0 for label in LABELS}


def save_eye(eye_points, landmarks, frame, gray, label):
    """Crops and saves the eye image."""
    # Get eye region
    eye_region = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_points])
    x, y, w, h = cv2.boundingRect(eye_region)

    # Add a little padding
    pad = 5
    x -= pad;
    y -= pad;
    w += pad * 2;
    h += pad * 2

    # Ensure within frame bounds
    h_f, w_f = frame.shape[:2]
    if x < 0: x = 0
    if y < 0: y = 0
    if x + w > w_f: w = w_f - x
    if y + h > h_f: h = h_f - y

    eye_img = gray[y:y + h, x:x + w]

    # Resize for CNN input (standardize size)
    try:
        eye_img = cv2.resize(eye_img, (64, 64))
        filename = f"{DATASET_PATH}/{label}/{label}_{count[label]}.jpg"
        cv2.imwrite(filename, eye_img)
        count[label] += 1
        print(f"Saved {label}: {count[label]}")
    except Exception as e:
        pass


print("--- CONTROLS ---")
print("Press 'l' to save LEFT gaze")
print("Press 'r' to save RIGHT gaze")
print("Press 'c' to save CENTER gaze")
print("Press 'b' to save CLOSED (Blink) gaze")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        # Draw eyes for feedback
        for n in range(36, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('l'):
            save_eye(range(36, 42), landmarks, frame, gray, "left")  # Right eye (screen left)
            save_eye(range(42, 48), landmarks, frame, gray, "left")  # Left eye
        elif key == ord('r'):
            save_eye(range(36, 42), landmarks, frame, gray, "right")
            save_eye(range(42, 48), landmarks, frame, gray, "right")
        elif key == ord('c'):
            save_eye(range(36, 42), landmarks, frame, gray, "center")
            save_eye(range(42, 48), landmarks, frame, gray, "center")
        elif key == ord('b'):
            save_eye(range(36, 42), landmarks, frame, gray, "closed")
            save_eye(range(42, 48), landmarks, frame, gray, "closed")
        elif key == ord('q'):
            break

    cv2.putText(frame, f"L:{count['left']} R:{count['right']} C:{count['center']} B:{count['closed']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Data Collector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()