import cv2

MODEL = "face_detection_yunet_2023mar.onnx"

detector = cv2.FaceDetectorYN.create(
    MODEL,
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        print("Could not access camera")
        break
    
    frame = cv2.flip(frame, 1)  # 0 → flip vertically, 1 → flip horizontally, -1 → flip both and comment out the code to see the mirror effect(original) of the camera

    height, width = frame.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)

    # Start face counter
    face_count = 0

    if faces is not None:
        for face in faces:
            face_count += 1

            x, y, w, h = face[:4].astype(int)

            # Draw rectangle around face
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

    # Display face count
    cv2.putText(
        frame,
        f"Faces detected: {face_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face Detection", frame)

    # Press O to quit
    if cv2.waitKey(1) & 0xFF == ord("o"):
        break

camera.release()
cv2.destroyAllWindows()