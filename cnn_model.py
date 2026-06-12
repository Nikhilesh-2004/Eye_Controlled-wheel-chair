# Step 2: CNN Model Training & Evaluation
# This script trains a Convolutional Neural Network on the collected eye data.

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATASET_PATH = "dataset_eye"
LABELS = ["left", "right", "center", "closed"]
IMG_SIZE = 64

# --- Load Data ---
print("[INFO] Loading data...")
data = []
labels = []

for label in LABELS:
    path = os.path.join(DATASET_PATH, label)
    class_num = LABELS.index(label)
    if not os.path.exists(path):
        print(f"Warning: Folder {path} not found. Run data collector first.")
        continue

    for img_name in os.listdir(path):
        try:
            img_path = os.path.join(path, img_name)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # --- APPLY CLAHE TO TRAINING DATA ---
            # This ensures the model learns on the same high-contrast
            # images it will see during real-time inference.
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_array = clahe.apply(img_array)
            # ------------------------------------

            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append(resized_array)
            labels.append(class_num)
        except Exception as e:
            pass

data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # Reshape for CNN (Batch, H, W, Channel)
data = data / 255.0  # Normalize pixel values
labels = np.array(labels)

# One-hot encoding
labels_one_hot = to_categorical(labels, num_classes=len(LABELS))

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, labels_one_hot, test_size=0.2, random_state=42)

print(f"[INFO] Training samples: {len(X_train)}")
print(f"[INFO] Testing samples: {len(X_test)}")

# --- Build CNN Model ---
# Architecture: Conv -> Pool -> Conv -> Pool -> Flatten -> Dense -> Output
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),  # Prevent overfitting

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(LABELS), activation='softmax')  # Output layer (4 classes)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- Train Model ---
print("[INFO] Training model...")
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), batch_size=32)

# --- Save Model ---
model.save("eye_gaze_cnn_model.h5")
print("[INFO] Model saved as 'eye_gaze_cnn_model.h5'")

# --- Evaluation & Metrics ---
print("\n[INFO] Evaluating network...")
predictions = model.predict(X_test)
y_pred_classes = np.argmax(predictions, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# 1. Classification Report (Precision, Recall, F1-Score)
print("\n--- PERFORMANCE METRICS ---")
print(classification_report(y_true_classes, y_pred_classes, target_names=LABELS))

# 2. Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
print("\n--- CONFUSION MATRIX ---")

print(cm)

# Optional: Visualize Confusion Matrix
try:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
except:
    print("Could not plot confusion matrix (missing libraries?), but metrics are printed above.")