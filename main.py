from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(0, save=True, imgsz=320, conf=0.5, device="mps", show=True)