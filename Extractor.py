import cv2
import os

video_path = "Video_Very_easy.mp4"
output_folder = "frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

count = 0
frame_skip = 5  # take every 5th frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % frame_skip == 0:
        cv2.imwrite(f"{output_folder}/frame_{count}.jpg", frame)

    count += 1

cap.release()
print("Frames extracted")