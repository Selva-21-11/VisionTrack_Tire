import cv2
import numpy as np
from ultralytics import YOLO

# Load model
model = YOLO("runs/detect/train4/weights/best.pt")

cap = cv2.VideoCapture("Video.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

out = cv2.VideoWriter(
    "output_FINAL_PRO.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# 🔥 Kalman Filter
kalman = cv2.KalmanFilter(4, 2)

kalman.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]], np.float32)

kalman.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]], np.float32)

kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

tracker = None
tracking = False
frame_count = 0

# 🔥 Store last known position
prev_px, prev_py = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 🔥 Dynamic confidence
    if frame_count < 50:
        conf_threshold = 0.2
    else:
        conf_threshold = 0.4

    # 🔥 Detection logic
    if frame_count < 50 or not tracking or frame_count % 10 == 0:

        results = model(frame, imgsz=960)[0]

        best_box = None

        # 🔥 If we already have tracking → use distance filter
        if prev_px is not None:
            min_dist = float("inf")

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                if conf > conf_threshold:
                    cx = x1 + (x2-x1)//2
                    cy = y1 + (y2-y1)//2

                    dist = (cx - prev_px)**2 + (cy - prev_py)**2

                    # 🔥 Ignore large objects (person filter)
                    area = (x2 - x1) * (y2 - y1)
                    if area > 80000:
                        continue

                    if dist < min_dist:
                        min_dist = dist
                        best_box = (x1, y1, x2, y2)

        else:
            # 🔥 First detection → use highest confidence
            best_conf = 0
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                if conf > conf_threshold:
                    if conf > best_conf:
                        best_conf = conf
                        best_box = (x1, y1, x2, y2)

        # 🔥 Initialize tracker
        if best_box is not None:
            x1, y1, x2, y2 = best_box

            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(frame, (x1, y1, x2-x1, y2-y1))
            tracking = True

            cx = x1 + (x2-x1)//2
            cy = y1 + (y2-y1)//2

            kalman.statePre = np.array([[cx],[cy],[0],[0]], np.float32)

            prev_px, prev_py = cx, cy

    # 🔥 Tracking + Kalman
    if tracking and tracker is not None:
        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = map(int, bbox)

            cx = x + w//2
            cy = y + h//2

            measurement = np.array([[np.float32(cx)],
                                    [np.float32(cy)]])

            kalman.correct(measurement)
            prediction = kalman.predict()

            px, py = int(prediction[0]), int(prediction[1])

            prev_px, prev_py = px, py

            cv2.rectangle(frame,
                          (px - w//2, py - h//2),
                          (px + w//2, py + h//2),
                          (0,255,0), 2)

            cv2.putText(frame, "Tire",
                        (px, py-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

        else:
            tracking = False
            tracker = None

    out.write(frame)

    cv2.imshow("FINAL PRO TRACKING", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ output_FINAL_PRO.mp4 saved")

# import cv2
# import numpy as np

# cap = cv2.VideoCapture("Video_EASY.mp4")

# tracker = None
# tracking = False

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.resize(frame, (640, 480))

#     if not tracking:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Threshold for dark object
#         _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

#         # Remove noise
#         thresh = cv2.medianBlur(thresh, 5)

#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         for cnt in contours:
#             area = cv2.contourArea(cnt)

#             if area > 100:  # tune this
#                 x, y, w, h = cv2.boundingRect(cnt)

#                 aspect_ratio = h / float(w)

#                 # Tire is tall-ish (ellipse)
#                 if 1.2 < aspect_ratio < 4.0:
#                     tracker = cv2.TrackerCSRT_create()
#                     tracker.init(frame, (x, y, w, h))
#                     tracking = True
#                     print("Detected and tracking started")
#                     break

#     else:
#         success, bbox = tracker.update(frame)

#         if success:
#             x, y, w, h = [int(v) for v in bbox]
#             cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
#             cv2.putText(frame, "Tracking Tire", (x,y-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
#         else:
#             tracking = False  # re-detect

#     cv2.imshow("Tire Tracking", frame)

#     if cv2.waitKey(30) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

# #-------------------------------------------------------------------------------------

# # import cv2
# # from ultralytics import YOLO
# # import supervision as sv

# # # Load YOLO model
# # model = YOLO("yolov8n.pt")  # pretrained model

# # cap = cv2.VideoCapture("Video_Very_easy.mp4")

# # tracker = sv.ByteTrack()
# # box_annotator = sv.BoxAnnotator()

# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break

# #     # Run detection
# #     results = model(frame)[0]

# #     # Convert to supervision format
# #     detections = sv.Detections.from_ultralytics(results)

# #     # Track objects
# #     detections = tracker.update_with_detections(detections)

# #     # Draw boxes
# #     frame = box_annotator.annotate(
# #         scene=frame,
# #         detections=detections
# #     )

# #     cv2.imshow("YOLO Tracking", frame)

# #     if cv2.waitKey(1) & 0xFF == 27:
# #         break

# # cap.release()
# # cv2.destroyAllWindows()

# #------------------------------------------------------------------
# # import cv2
# # import numpy as np

# # video_path = "Video_Very_easy.mp4"
# # cap = cv2.VideoCapture(video_path)

# # tracker = None
# # initialized = False

# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break

# #     frame = cv2.resize(frame, (640, 480))
# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #     gray = cv2.GaussianBlur(gray, (9, 9), 2)

# #     if not initialized:
# #         # Detect circles (tire)
# #         circles = cv2.HoughCircles(
# #             gray,
# #             cv2.HOUGH_GRADIENT,
# #             dp=1.2,
# #             minDist=100,
# #             param1=100,
# #             param2=30,
# #             minRadius=30,
# #             maxRadius=150
# #         )

# #         if circles is not None:
# #             circles = np.round(circles[0, :]).astype("int")

# #             # Take first detected circle
# #             x, y, r = circles[0]

# #             # Convert circle → bounding box
# #             bbox = (x - r, y - r, 2*r, 2*r)

# #             tracker = cv2.TrackerCSRT_create()
# #             tracker.init(frame, bbox)

# #             initialized = True
# #             print("Tire detected and tracking started")

# #     else:
# #         success, bbox = tracker.update(frame)

# #         if success:
# #             x, y, w, h = [int(v) for v in bbox]

# #             cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

# #             cx = x + w // 2
# #             cy = y + h // 2
# #             cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)

# #             cv2.putText(frame, "Tracking Tire", (20,40),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

# #         else:
# #             cv2.putText(frame, "Lost", (20,40),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
# #             initialized = False  # Try detecting again

# #     cv2.imshow("Auto Tire Tracking", frame)

# #     if cv2.waitKey(30) & 0xFF == 27:
# #         break

# # cap.release()
# # cv2.destroyAllWindows()