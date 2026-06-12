# ///////////////////////////////////////////////////////////////////////
# Step 3: Production-Ready Real-time Eye Control
# Features:
# - The "Clutch" (1.2s blink to Pause/Unpause) - UI Timer Added
# - Spectacles / Glasses Optimization
# - Emergency SOS (4s blink to sound alarm)
# - Adaptive Fatigue (Dynamically adjusts sensitivity)
# - Targeted CLAHE (Low-light enhancement safely applied)
# ///////////////////////////////////////////////////////////////////////

import cv2
import dlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.spatial import distance as dist
import time
import threading
import platform

# --- Configuration & Thresholds ---
MODEL_PATH = "eye_gaze_cnn_model.h5"
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
IMG_SIZE = 64
LABELS = ["LEFT", "RIGHT", "CENTER", "CLOSED"]

# --- Spectacles-Optimized Thresholds ---
# Lowered confidence to 0.35 to be forgiving of glasses glare blurring the eye.
MIN_CONFIDENCE = 0.35
BASE_GAZE_FRAMES = 6
BLINK_CONFIRM_FRAMES = 3

# Adjusted to 0.20. Glasses naturally squish the eye bounding box.
# This catches blinks behind thick frames without permanently locking the system.
EAR_THRESHOLD = 0.20

# --- Real-World Feature Variables ---
PAUSE_BLINK_DURATION = 1.2  # FASTER: Only 1.2 seconds to toggle Pause/Unpause
SOS_BLINK_DURATION = 4.0  # Seconds to trigger SOS Alarm
FATIGUE_WINDOW = 60.0  # Track blinks over the last 60 seconds
FATIGUE_LIMIT = 25  # If > 25 blinks in 60s, trigger fatigue mode
FACE_LOST_GRACE_FRAMES = 10  # Grace period for face tracking glitches

is_blinking = False
blink_start_time = 0
pause_toggled = False
sos_triggered = False
is_paused = False
blink_history = []
dynamic_gaze_frames = BASE_GAZE_FRAMES
face_lost_counter = 0

# --- Load Model & Predictors ---
print("[INFO] Loading CNN model...")
try:
    model = load_model(MODEL_PATH)
except:
    print("Error: Model file not found. Run CNN training script first.")
    exit()

detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
except:
    print("Error: Download shape_predictor_68_face_landmarks.dat")
    exit()


# --- Audio Helper (Runs in background thread to prevent lag) ---
def play_alert(alert_type="beep"):
    """Plays cross-platform audio alerts."""
    try:
        if platform.system() == "Windows":
            import winsound
            if alert_type == "sos":
                for _ in range(5):
                    winsound.Beep(2000, 500)
                    time.sleep(0.1)
            else:
                winsound.Beep(800, 200)
        else:
            if alert_type == "sos":
                for _ in range(5):
                    print('\a', end='', flush=True)
                    time.sleep(0.5)
            else:
                print('\a', end='', flush=True)
    except:
        pass


# --- Geometric & Vision Helpers ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def get_head_roll_2d(landmarks):
    left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
    right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
    left_center = left_eye_pts.mean(axis=0)
    right_center = right_eye_pts.mean(axis=0)
    dY = left_center[1] - right_center[1]
    dX = left_center[0] - right_center[0]
    return np.degrees(np.arctan2(dY, dX))


def get_robust_face_and_landmarks(gray, detector, predictor):
    faces = detector(gray, 0)
    if len(faces) > 0:
        landmarks = predictor(gray, faces[0])
        if abs(get_head_roll_2d(landmarks)) < 25:
            return gray, faces[0], landmarks

    faces = detector(gray, 1)
    if len(faces) > 0:
        return gray, faces[0], predictor(gray, faces[0])

    angles = [30, -30]
    h, w = gray.shape[:2]
    center = (int(w // 2), int(h // 2))

    for angle in angles:
        M = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated_gray = cv2.warpAffine(gray, M, (w, h))
        faces = detector(rotated_gray, 0)
        if len(faces) > 0:
            landmarks = predictor(rotated_gray, faces[0])
            if abs(get_head_roll_2d(landmarks)) < 25:
                return rotated_gray, faces[0], landmarks

    return gray, None, None


def preprocess_eye(eye_points, landmarks, gray, roll_angle):
    eye_region = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_points])
    cx, cy = np.mean(eye_region, axis=0)
    cx, cy = int(cx), int(cy)
    eye_width = int(dist.euclidean(eye_region[0], eye_region[3]))
    if eye_width == 0: return None

    pad = int(eye_width * 1.5)
    x1, y1 = max(0, cx - pad), max(0, cy - pad)
    x2, y2 = min(gray.shape[1], cx + pad), min(gray.shape[0], cy + pad)
    patch = gray[y1:y2, x1:x2]
    if patch.shape[0] == 0 or patch.shape[1] == 0: return None

    patch_cx, patch_cy = int(cx - x1), int(cy - y1)
    M_rot = cv2.getRotationMatrix2D((patch_cx, patch_cy), float(roll_angle), 1.0)
    rotated_patch = cv2.warpAffine(patch, M_rot, (patch.shape[1], patch.shape[0]))

    crop_w = int(eye_width * 1.3)
    crop_h = int(crop_w * 0.7)
    rx1 = max(0, patch_cx - crop_w // 2)
    ry1 = max(0, patch_cy - crop_h // 2)
    rx2 = min(rotated_patch.shape[1], patch_cx + crop_w // 2)
    ry2 = min(rotated_patch.shape[0], patch_cy + crop_h // 2)

    final_eye = rotated_patch[ry1:ry2, rx1:rx2]

    try:
        # SPECTACLES FIX: Reduced CLAHE clipLimit from 2.0 to 1.5.
        # This prevents the filter from enhancing the bright glares on glass lenses.
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced_eye = clahe.apply(final_eye)

        eye_img = cv2.resize(enhanced_eye, (IMG_SIZE, IMG_SIZE))
        eye_img = eye_img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        return eye_img / 255.0
    except:
        return None


# --- Main Loop ---
cap = cv2.VideoCapture(0)
consecutive_frames = 0
current_candidate_label = "CENTER"
locked_command = "WAITING"
last_printed_command = ""
color = (0, 255, 0)  # <-- FIX: Initialize default color here

print("[INFO] Real-World Assistive Features Active. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    current_time = time.time()
    blink_history = [t for t in blink_history if current_time - t < FATIGUE_WINDOW]

    if len(blink_history) >= FATIGUE_LIMIT:
        dynamic_gaze_frames = BASE_GAZE_FRAMES * 2
        fatigue_status = "HIGH (Sensors Dampened)"
    else:
        dynamic_gaze_frames = BASE_GAZE_FRAMES
        fatigue_status = "NORMAL"

    working_gray, face, landmarks = get_robust_face_and_landmarks(gray, detector, predictor)

    if face is not None and landmarks is not None:
        face_lost_counter = 0

        left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        avg_ear = (eye_aspect_ratio(left_eye_pts) + eye_aspect_ratio(right_eye_pts)) / 2.0

        predicted_raw_label = "CENTER"
        confidence = 0.0

        # === Time-Based Blink Logic (Clutch & SOS) ===
        if avg_ear < EAR_THRESHOLD:
            predicted_raw_label = "CLOSED"
            confidence = 1.0

            if not is_blinking:
                is_blinking = True
                blink_start_time = time.time()
                pause_toggled = False
                sos_triggered = False

            blink_duration = time.time() - blink_start_time

            # VISUAL CLUTCH TIMER: Show the user how long they have held their blink
            cv2.putText(frame, f"Holding Blink: {blink_duration:.1f}s", (230, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 165, 255), 2)

            # 4 Seconds: SOS
            if blink_duration >= SOS_BLINK_DURATION and not sos_triggered:
                sos_triggered = True
                threading.Thread(target=play_alert, args=("sos",), daemon=True).start()

            # 1.2 Seconds: Pause/Clutch Toggle
            elif blink_duration >= PAUSE_BLINK_DURATION and not pause_toggled and not sos_triggered:
                is_paused = not is_paused
                pause_toggled = True
                threading.Thread(target=play_alert, args=("beep",), daemon=True).start()

        else:
            if is_blinking:
                blink_history.append(time.time())
                is_blinking = False

            if not is_paused and not sos_triggered:
                head_roll = get_head_roll_2d(landmarks)
                left_eye_input = preprocess_eye(range(42, 48), landmarks, working_gray, head_roll)
                right_eye_input = preprocess_eye(range(36, 42), landmarks, working_gray, head_roll)

                inputs = []
                if left_eye_input is not None: inputs.append(left_eye_input[0])
                if right_eye_input is not None: inputs.append(right_eye_input[0])

                if len(inputs) > 0:
                    preds = model.predict(np.array(inputs), verbose=0)
                    avg_pred = np.mean(preds, axis=0)
                    avg_pred[3] = -1
                    class_idx = np.argmax(avg_pred)
                    confidence = avg_pred[class_idx]
                    predicted_raw_label = LABELS[class_idx]

        # === STABILITY & COMMAND LOGIC ===
        if sos_triggered:
            locked_command = "ACTION: STOP (SOS ALARM)"
            color = (0, 0, 255)
        elif is_paused:
            # Inform the user exactly how to unpause
            locked_command = "SYSTEM PAUSED (Hold Blink 1.2s to Drive)"
            color = (0, 255, 255)
        else:
            if confidence > MIN_CONFIDENCE:
                if predicted_raw_label == current_candidate_label:
                    consecutive_frames += 1
                else:
                    consecutive_frames = 0
                    current_candidate_label = predicted_raw_label

                if current_candidate_label == "CLOSED":
                    if consecutive_frames >= BLINK_CONFIRM_FRAMES:
                        locked_command = "ACTION: STOP"
                else:
                    if consecutive_frames >= dynamic_gaze_frames:
                        if current_candidate_label == "LEFT":
                            locked_command = "ACTION: TURN LEFT"
                        elif current_candidate_label == "RIGHT":
                            locked_command = "ACTION: TURN RIGHT"
                        elif current_candidate_label == "CENTER":
                            locked_command = "ACTION: FORWARD"

            color = (0, 255, 0) if "STOP" not in locked_command else (0, 0, 255)

    else:
        face_lost_counter += 1
        if face_lost_counter > FACE_LOST_GRACE_FRAMES:
            locked_command = "ACTION: STOP (FACE LOST)"
            color = (0, 0, 255)
            consecutive_frames = 0
            current_candidate_label = "CENTER"
            cv2.putText(frame, "CRITICAL", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Searching...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # --- Terminal Output ---
    if locked_command != last_printed_command:
        print(f"[COMMAND] {locked_command}")
        last_printed_command = locked_command

    # --- User Interface Overlay ---
    req_frames = BLINK_CONFIRM_FRAMES if current_candidate_label == "CLOSED" else dynamic_gaze_frames
    progress = min(consecutive_frames / req_frames, 1.0) if "FACE LOST" not in locked_command else 1.0

    cv2.rectangle(frame, (10, 60), (10 + int(200 * progress), 80), (255, 255, 0), -1)
    cv2.rectangle(frame, (10, 60), (210, 80), (255, 255, 255), 2)

    cv2.putText(frame, f"Fatigue: {fatigue_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
    if face is not None:
        cv2.putText(frame, f"Signal: {current_candidate_label} (EAR: {avg_ear:.2f})", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(frame, f"LOCKED: {locked_command}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7 if is_paused else 0.8,
                color, 3)

    cv2.imshow("Hybrid Eye Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()