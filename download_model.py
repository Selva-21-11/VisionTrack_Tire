import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture("Video_Very_easy.mp4")

# Get video properties
width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create output video writer
out = cv2.VideoWriter(
    "output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, "Tire", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Write frame instead of showing
    out.write(frame)

cap.release()
out.release()

print("✅ Output video saved as output.mp4")

# from roboflow import Roboflow

# rf = Roboflow(api_key="GVQMX46zQkcToT7ya2wQ")

# project = rf.workspace().project("tire-detection-k2b74")
# dataset = project.version(1).download("yolov8")

# print(project.versions())