# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

# from ultralytics import YOLO
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="Tire_Detection.v2i.yolov8/data.yaml",
        epochs=30,
        imgsz=640,
        device=0,
        workers=0
    )

if __name__ == "__main__":
    main()


# import cv2
# import numpy as np

# print(cv2.__version__)
# print(np.__version__)
# print(hasattr(cv2, 'legacy'))





