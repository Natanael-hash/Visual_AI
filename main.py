from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(0, save=True, imgsz=640, conf=0.5, device="mps", show=True)

# results = model("data_preparation/data_images/test/0f64529c4d33acdb206935b0e84ab7d3_jpg.rf.bd243132c3d75391507f79a10397e25e.jpg")
# results[0].show()