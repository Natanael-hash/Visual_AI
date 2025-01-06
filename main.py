from ultralytics import YOLO

model = YOLO("image_processing/train/weights/best.pt")

model.predict(source="image_processing/4K Road traffic video for object detection and tracking - free download now! [MNn9qKG2UFI].webm", save=True, imgsz=320, conf=0.5, device="mps", show=True)