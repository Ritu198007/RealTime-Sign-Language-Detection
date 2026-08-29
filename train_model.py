import csv
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------
# SETTINGS
# --------------------------------

DATASET = "hand_sign_dataset.csv"
MODEL_FILE = "asl_static_model_normalized.pkl"


# --------------------------------
# NORMALIZE HAND LANDMARKS
# --------------------------------

def normalize_landmarks(values):
    """
    Convert 21 hand landmarks into normalized coordinates.

    Landmark 0 is the wrist.
    We make the wrist the origin so hand position
    in the camera frame doesn't matter as much.
    """

    landmarks = np.array(values, dtype=np.float32).reshape(21, 3)

    # Make wrist the origin
    wrist = landmarks[0].copy()
    landmarks = landmarks - wrist

    # Normalize by hand size
    distances = np.linalg.norm(landmarks, axis=1)

    max_distance = np.max(distances)

    if max_distance > 0:
        landmarks = landmarks / max_distance

    return landmarks.flatten()


# --------------------------------
# LOAD DATASET
# --------------------------------

X = []
y = []

with open(DATASET, "r", newline="") as file:

    reader = csv.reader(file)

    # Skip header
    next(reader)

    for row in reader:

        if not row:
            continue

        label = row[0]

        # 63 values = 21 landmarks × 3 coordinates
        values = [float(value) for value in row[1:]]

        if len(values) != 63:
            continue

        normalized = normalize_landmarks(values)

        X.append(normalized)
        y.append(label)


X = np.array(X)
y = np.array(y)


print("Dataset loaded!")
print("Samples:", len(X))
print("Features per sample:", X.shape[1])
print("Letters:", sorted(set(y)))


# --------------------------------
# SPLIT DATA
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print()
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------
# TRAIN MODEL
# --------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

print()
print("Training normalized model...")

model.fit(X_train, y_train)


# --------------------------------
# TEST MODEL
# --------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print()
print("==============================")
print("NORMALIZED MODEL RESULTS")
print("==============================")
print(f"Accuracy: {accuracy * 100:.2f}%")
print()

print(
    classification_report(
        y_test,
        predictions
    )
)


# --------------------------------
# SAVE MODEL
# --------------------------------

joblib.dump(
    model,
    MODEL_FILE
)

print("==============================")
print(f"Model saved as: {MODEL_FILE}")
print("==============================")