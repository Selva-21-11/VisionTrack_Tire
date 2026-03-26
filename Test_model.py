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
    "output_FINAL.mp4",
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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 🔥 Dynamic confidence (important)
    if frame_count < 50:
        conf_threshold = 0.2   # early frames → easier detection
    else:
        conf_threshold = 0.4   # later → stable detection

    # 🔥 Force detection at start + periodic correction
    if frame_count < 50 or not tracking or frame_count % 10 == 0:

        results = model(frame, imgsz=960)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if conf > conf_threshold:

                tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(frame, (x1, y1, x2-x1, y2-y1))
                tracking = True

                # Initialize Kalman
                cx = x1 + (x2-x1)//2
                cy = y1 + (y2-y1)//2

                kalman.statePre = np.array([[cx],[cy],[0],[0]], np.float32)
                break

    # 🔥 Tracking + Kalman smoothing
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

            cv2.rectangle(frame,
                          (px - w//2, py - h//2),
                          (px + w//2, py + h//2),
                          (0,255,0), 2)

            cv2.putText(frame, "Tire",
                        (px, py-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

        else:
            # 🔥 DO NOT immediately reset → allow recovery
            tracking = False
            tracker = None

    out.write(frame)

    cv2.imshow("Tire Tracking FINAL", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ output_FINAL.mp4 saved")

# import cv2
# import numpy as np
# from ultralytics import YOLO

# # Load model
# model = YOLO("runs/detect/train4/weights/best.pt")

# cap = cv2.VideoCapture("Video.mp4")

# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)

# if fps == 0:
#     fps = 30

# out = cv2.VideoWriter(
#     "output_TEST2.mp4",
#     cv2.VideoWriter_fourcc(*"mp4v"),
#     fps,
#     (width, height)
# )

# # 🔥 Kalman Filter setup
# kalman = cv2.KalmanFilter(4, 2)

# kalman.measurementMatrix = np.array([[1,0,0,0],
#                                      [0,1,0,0]], np.float32)

# kalman.transitionMatrix = np.array([[1,0,1,0],
#                                     [0,1,0,1],
#                                     [0,0,1,0],
#                                     [0,0,0,1]], np.float32)

# kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

# tracker = None
# tracking = False
# frame_count = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1

#     # YOLO detection occasionally
#     if not tracking or frame_count % 10 == 0:
#         results = model(frame, imgsz=960)[0]

#         for box in results.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = float(box.conf[0])

#             if conf > 0.3:
#                 tracker = cv2.legacy.TrackerCSRT_create()
#                 tracker.init(frame, (x1, y1, x2-x1, y2-y1))
#                 tracking = True

#                 # Initialize Kalman with center
#                 cx = x1 + (x2-x1)//2
#                 cy = y1 + (y2-y1)//2

#                 kalman.statePre = np.array([[cx],[cy],[0],[0]], np.float32)
#                 break

#     if tracking and tracker is not None:
#         success, bbox = tracker.update(frame)

#         if success:
#             x, y, w, h = map(int, bbox)

#             # Measurement (center point)
#             cx = x + w//2
#             cy = y + h//2

#             measurement = np.array([[np.float32(cx)],
#                                     [np.float32(cy)]])

#             kalman.correct(measurement)

#             prediction = kalman.predict()

#             px, py = int(prediction[0]), int(prediction[1])

#             # Draw smoothed box
#             cv2.rectangle(frame,
#                           (px - w//2, py - h//2),
#                           (px + w//2, py + h//2),
#                           (0,255,0), 2)

#             cv2.putText(frame, "Tire",
#                         (px, py-10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6, (0,255,0), 2)

#         else:
#             tracking = False
#             tracker = None

#     out.write(frame)

#     cv2.imshow("Tire", frame)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# out.release()
# cv2.destroyAllWindows()

# print("✅ output_kalman.mp4 saved")

#-----------------------------------------------------------------------------------------------------------


# import cv2
# from ultralytics import YOLO

# # Load model
# model = YOLO("runs/detect/train7/weights/best.pt")

# cap = cv2.VideoCapture("Video.mp4")

# # 🔥 Get original video properties
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)

# # 🔥 Fix FPS issue (very important)
# if fps == 0:
#     fps = 30

# print("Width:", width, "Height:", height, "FPS:", fps)

# # 🔥 Video writer (MP4 working)
# out = cv2.VideoWriter(
#     "output_TEST.mp4",
#     cv2.VideoWriter_fourcc(*"mp4v"),  # correct codec
#     fps,
#     (width, height)   # MUST match frame size
# )

# tracker = None
# tracking = False
# frame_count = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1

#     # 🔥 Detection only when needed (or every 10 frames)
#     if not tracking or frame_count % 10 == 0:

#         results = model(frame, imgsz=640)[0]

#         for box in results.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = float(box.conf[0])

#             if conf > 0.5:
#                 tracker = cv2.legacy.TrackerCSRT_create()
#                 tracker.init(frame, (x1, y1, x2-x1, y2-y1))
#                 tracking = True
#                 break

#     # 🔥 Tracking
#     if tracking and tracker is not None:
#         success, bbox = tracker.update(frame)

#         if success:
#             x, y, w, h = map(int, bbox)

#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
#             cv2.putText(frame, "Tracking",
#                         (x, y-10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6, (0,255,0), 2)

#         else:
#             tracking = False
#             tracker = None

#     # 🔥 Write frame (CRITICAL)
#     out.write(frame)

#     # Optional preview
#     cv2.imshow("Tracking", frame)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# out.release()
# cv2.destroyAllWindows()

# print("✅ Video saved as output_tracked.mp4")


# import cv2
# from ultralytics import YOLO

# # Load model
# model = YOLO("runs/detect/train6/weights/best.pt")

# cap = cv2.VideoCapture("your_video.mp4")

# tracker = None
# tracking = False

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Resize for speed
#     frame = cv2.resize(frame, (640, 480))

#     if not tracking:
#         # Run YOLO detection
#         results = model(frame, imgsz=640)[0]

#         for box in results.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = float(box.conf[0])

#             if conf > 0.5:
#                 # Initialize tracker
#                 tracker = cv2.legacy.TrackerCSRT_create()
#                 tracker.init(frame, (x1, y1, x2-x1, y2-y1))
#                 tracking = True
#                 break

#     else:
#         # Update tracker
#         success, bbox = tracker.update(frame)

#         if success:
#             x, y, w, h = map(int, bbox)

#             # Draw tracking box
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             cv2.putText(frame, "Tracking", (x, y-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

#         else:
#             # Tracker lost → go back to YOLO
#             tracking = False
#             tracker = None

#     cv2.imshow("Tire Tracking", frame)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()



#----------------------------------------------------------------------------
# import cv2
# from ultralytics import YOLO

# # Load model
# model = YOLO("runs/detect/train7/weights/best.pt")

# cap = cv2.VideoCapture("Video_EASY.mp4")

# # Video writer
# width = int(cap.get(3))
# height = int(cap.get(4))
# fps = int(cap.get(cv2.CAP_PROP_FPS))

# out = cv2.VideoWriter("output_tracked.mp4",
#                       cv2.VideoWriter_fourcc(*"mp4v"),
#                       fps, (width, height))

# tracker = None

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # YOLO detection
#     results = model(frame, imgsz=416)[0]

#     detected = False

#     for box in results.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         conf = float(box.conf[0])

#         if conf > 0.5:
#             tracker = cv2.legacy.TrackerCSRT_create()
#             tracker.init(frame, (x1, y1, x2-x1, y2-y1))
#             detected = True
#             break

#     # If detected → update tracker
#     if tracker is not None:
#         success, bbox = tracker.update(frame)

#         if success:
#             x, y, w, h = map(int, bbox)
#             cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
#             cv2.putText(frame, "Tracking Tire",
#                         (x, y-10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6, (0,255,0), 2)

#     out.write(frame)

# cap.release()
# out.release()

# print("✅ output_tracked.mp4 saved")

#--------------------------------------------------------------------------------------

# import cv2
# from ultralytics import YOLO

# # Load trained model
# model = YOLO("runs/detect/train7/weights/best.pt")

# # Load video
# cap = cv2.VideoCapture("video_Very_easy.mp4")

# # Get video properties
# width = int(cap.get(3))
# height = int(cap.get(4))
# fps = int(cap.get(cv2.CAP_PROP_FPS))

# # Output video writer
# out = cv2.VideoWriter(
#     "output.mp4",
#     cv2.VideoWriter_fourcc(*"mp4v"),
#     fps,
#     (width, height)
# )

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # 🔥 Run detection (GPU used automatically)
#     results = model(frame, imgsz=416)[0]

#     # Draw boxes
#     for box in results.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         conf = float(box.conf[0])

#         if conf > 0.5:  # confidence filter
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
#             cv2.putText(frame, f"Tire {conf:.2f}",
#                         (x1, y1-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6,
#                         (0,255,0), 2)

#     # Save frame
#     out.write(frame)

# # Release everything
# cap.release()
# out.release()

# print("✅ Output video saved as output.mp4")



# import cv2
# from inference_sdk import InferenceHTTPClient

# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="GVQMX46zQkcToT7ya2wQ"
# )

# model_id = "tire-detection-k2b74/1"

# cap = cv2.VideoCapture("Video.mp4")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     result = CLIENT.infer(frame, model_id=model_id)

#     for pred in result["predictions"]:
#         x = int(pred["x"])
#         y = int(pred["y"])
#         w = int(pred["width"])
#         h = int(pred["height"])

#         x1 = int(x - w/2)
#         y1 = int(y - h/2)
#         x2 = int(x + w/2)
#         y2 = int(y + h/2)

#         cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
#         cv2.putText(frame, "Tire", (x1,y1-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

#     cv2.imshow("Tire Detection", frame)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()


# # from roboflow import Roboflow

# # rf = Roboflow(api_key="GVQMX46zQkcToT7ya2wQ")
# # project = rf.workspace().project("tire-detection-k2b74")
# # model = project.version("1").model

# # job_id, signed_url, expire_time = model.predict_video(
# #     "Video_Very_easy.mp4",
# #     fps=5,
# #     prediction_type="batch-video",
# # )

# # results = model.poll_until_video_results(job_id)

# # print(results)
