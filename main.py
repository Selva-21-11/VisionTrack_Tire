import cv2
import numpy as np
from ultralytics import YOLO

# Load model
model = YOLO("runs/detect/train4/weights/best.pt")

cap = cv2.VideoCapture("INPUT.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

out = cv2.VideoWriter(
    "output_TEST2.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# Kalman Filter setup
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                     [0, 1, 0, 1],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

tracker = None
tracking = False
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # YOLO detection occasionally
    if not tracking or frame_count % 10 == 0:
        results = model(frame, imgsz=960)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf > 0.3:
                tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                tracking = True

                # Initialize Kalman with center
                cx = x1 + (x2 - x1) // 2
                cy = y1 + (y2 - y1) // 2
                kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
                break

    if tracking and tracker is not None:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)

            # Measurement (center point)
            cx = x + w // 2
            cy = y + h // 2
            measurement = np.array([[np.float32(cx)],
                                     [np.float32(cy)]])
            kalman.correct(measurement)
            prediction = kalman.predict()
            px, py = int(prediction[0][0]), int(prediction[1][0])

            # Draw smoothed box
            cv2.rectangle(frame, (px - w // 2, py - h // 2), (px + w // 2, py + h // 2), (0, 255, 0), 2)
            cv2.putText(frame, "Tire", (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            tracking = False
            tracker = None

    out.write(frame)
    cv2.imshow("Tire", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video saved")