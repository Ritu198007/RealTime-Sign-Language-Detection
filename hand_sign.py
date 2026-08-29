import cv2
import mediapipe as mp

# Model path
MODEL_PATH = "hand_landmarker.task"

# MediaPipe options
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Create detector
detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

# Open webcam
camera = cv2.VideoCapture(0)

timestamp = 0

while True:
    success, frame = camera.read()

    if not success:
        print("Could not access camera")
        break
    
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hands
    result = detector.detect_for_video(
        mp_image,
        timestamp
    )

    timestamp += 1

    # Draw landmarks
    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            points = []

            # Convert landmarks to screen coordinates
            for landmark in hand:

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                points.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Connect hand landmarks
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17)
            ]

            for start, end in connections:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0, 255, 0),
                    2
                )

    cv2.imshow("Hand Sign Detection", frame)

    # Press O to quit
    if cv2.waitKey(1) & 0xFF == ord("o"):
        break

camera.release()
detector.close()
cv2.destroyAllWindows()