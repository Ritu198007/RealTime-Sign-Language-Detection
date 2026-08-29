import cv2
import mediapipe as mp
import csv
import os
import time

# -----------------------------
# SETTINGS
# -----------------------------

MODEL_PATH = "hand_landmarker.task"

SEQUENCES_PER_LETTER = 50
FRAMES_PER_SEQUENCE = 30

LETTERS = ["J", "Z"]

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

file_name = "dynamic_hand_sign_dataset.csv"

file_exists = os.path.exists(file_name)

file = open(file_name, "a", newline="")
writer = csv.writer(file)

if not file_exists:
    header = ["label", "sequence"]

    for frame_number in range(FRAMES_PER_SEQUENCE):
        for landmark_number in range(21):
            header.extend([
                f"f{frame_number}_x{landmark_number}",
                f"f{frame_number}_y{landmark_number}",
                f"f{frame_number}_z{landmark_number}"
            ])

    writer.writerow(header)

# -----------------------------
# CHOOSE LETTER
# -----------------------------

letter = input("Enter J or Z: ").strip().upper()

if letter not in LETTERS:
    print("Please enter only J or Z.")
    file.close()
    detector.close()
    exit()

print()
print(f"Preparing to collect {letter}...")
print(f"Each sample contains {FRAMES_PER_SEQUENCE} frames.")
print(f"Target: {SEQUENCES_PER_LETTER} sequences.")
print()

# -----------------------------
# WEBCAM
# -----------------------------

camera = cv2.VideoCapture(0)

timestamp = 0
sequence_count = 0

# -----------------------------
# COLLECTION
# -----------------------------

while sequence_count < SEQUENCES_PER_LETTER:

    success, frame = camera.read()

    if not success:
        print("Could not access camera")
        break

    frame = cv2.flip(frame, 1)

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
        f"Sequences: {sequence_count}/{SEQUENCES_PER_LETTER}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Dynamic Sign Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    # Exit
    if key == ord("o"):
        break

    # Start recording a sequence
    if key == ord(" "):

        sequence = []

        print(
            f"Recording {letter} "
            f"sequence {sequence_count + 1}/"
            f"{SEQUENCES_PER_LETTER}"
        )

        while len(sequence) < FRAMES_PER_SEQUENCE:

            success, frame = camera.read()

            if not success:
                break

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
            # GET HAND LANDMARKS
            # -----------------------------

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                landmarks = []

                for landmark in hand:
                    landmarks.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

                sequence.append(landmarks)

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

            # Recording progress
            cv2.putText(
                frame,
                f"Recording: {len(sequence)}/"
                f"{FRAMES_PER_SEQUENCE}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "Dynamic Sign Collector",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("o"):
                break

        # -----------------------------
        # SAVE SEQUENCE
        # -----------------------------

        if len(sequence) == FRAMES_PER_SEQUENCE:

            row = [
                letter,
                sequence_count + 1
            ]

            for frame_landmarks in sequence:
                row.extend(frame_landmarks)

            writer.writerow(row)
            file.flush()

            sequence_count += 1

            print(
                f"Saved sequence "
                f"{sequence_count}/{SEQUENCES_PER_LETTER}"
            )

        else:
            print("Sequence incomplete. Not saved.")

# -----------------------------
# CLEANUP
# -----------------------------

camera.release()
detector.close()
file.close()
cv2.destroyAllWindows()

print()
print("Dynamic data collection finished.")
print(f"Sign: {letter}")
print(f"Sequences collected: {sequence_count}")