import cv2
import mediapipe as mp
import csv
import os

# -----------------------------
# SETTINGS
# -----------------------------

MODEL_PATH = "hand_landmarker.task"
SAMPLES_PER_LETTER = 50

LETTERS = [
    "A", "B", "C", "D", "E", "F",
    "G", "H", "I", "J", "K", "L",
    "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X",
    "Y", "Z"
]

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

# -----------------------------
# DATASET SETUP
# -----------------------------

file_name = "hand_sign_dataset.csv"

if not os.path.exists(file_name):
    with open(file_name, "w", newline="") as file:
        writer = csv.writer(file)

        header = ["label"]

        for i in range(21):
            header.extend([
                f"x{i}",
                f"y{i}",
                f"z{i}"
            ])

        writer.writerow(header)

# -----------------------------
# CHECK EXISTING DATA
# -----------------------------

counts = {letter: 0 for letter in LETTERS}

with open(file_name, "r", newline="") as file:
    reader = csv.reader(file)

    next(reader, None)

    for row in reader:
        if row and row[0] in counts:
            counts[row[0]] += 1

print("\nCurrent dataset:")

for letter in LETTERS:
    print(f"{letter}: {counts[letter]}/{SAMPLES_PER_LETTER}")

# -----------------------------
# OPEN DATASET
# -----------------------------

file = open(file_name, "a", newline="")
writer = csv.writer(file)

# -----------------------------
# WEBCAM
# -----------------------------

camera = cv2.VideoCapture(0)

timestamp = 0

# -----------------------------
# COLLECT EACH LETTER
# -----------------------------

stopped = False

for letter in LETTERS:

    # How many more samples do we need?
    samples_needed = SAMPLES_PER_LETTER - counts[letter]

    if samples_needed <= 0:
        print(f"\n{letter} already has {counts[letter]} samples. Skipping.")
        continue

    samples = 0

    print("\n" + "=" * 40)
    print(f"GET READY: {letter}")
    print(f"Need {samples_needed} more samples.")
    print("Press SPACE to capture.")
    print("Press O to stop.")
    print("=" * 40)

    # Give yourself time to prepare
    cv2.waitKey(1)

    while samples < samples_needed:

        success, frame = camera.read()

        if not success:
            print("Could not access camera")
            stopped = True
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = detector.detect_for_video(
            mp_image,
            timestamp
        )

        timestamp += 1

        # -----------------------------
        # DISPLAY
        # -----------------------------

        cv2.putText(
            frame,
            f"Sign: {letter}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {samples}/{samples_needed}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # -----------------------------
        # HAND DETECTION
        # -----------------------------

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # Draw landmarks
            for landmark in hand:

                x = int(
                    landmark.x * frame.shape[1]
                )

                y = int(
                    landmark.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

        cv2.imshow(
            "ASL Data Collector",
            frame
        )

        # -----------------------------
        # KEYBOARD
        # -----------------------------

        key = cv2.waitKey(1) & 0xFF

        # STOP
        if key == ord("o"):
            print("\nCollection stopped.")
            stopped = True
            break

        # CAPTURE
        if key == ord(" ") and result.hand_landmarks:

            hand = result.hand_landmarks[0]

            row = [letter]

            for landmark in hand:
                row.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            writer.writerow(row)
            file.flush()

            samples += 1
            counts[letter] += 1

            print(
                f"{letter}: "
                f"{samples}/{samples_needed}"
            )

    if stopped:
        break

    print(f"\n✅ Finished {letter}!")

# -----------------------------
# CLEANUP
# -----------------------------

camera.release()
detector.close()
file.close()
cv2.destroyAllWindows()

print("\n" + "=" * 40)
print("DATA COLLECTION FINISHED")
print("=" * 40)

for letter in LETTERS:
    print(f"{letter}: {counts[letter]} samples")