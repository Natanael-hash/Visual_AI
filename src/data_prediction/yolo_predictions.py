import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

class ObjectDetection:
    def __init__(self, capture_index):
        self.colors = None
        self.capture_index = capture_index

        self.device = 'mps' if torch.mps.is_available() else 'cpu'
        self.model = self.load_model()


    def load_model(self):
        model = YOLO("/Users/natanaelhordon/Desktop/Visual-AI/src/data_prediction/train7/weights/best.pt")
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame):
        xyxys = []
        confidences = []
        class_ids = []

        for result in results:
            boxes = result.boxes.cpu().numpy()
            
            xyxys.append(boxes.xyxy)
            confidences.append(boxes.conf)
            class_ids.append(boxes.cls)

        return results[0].plot(line_width=5, font_size=5), xyxys, confidences, class_ids

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened(), "Error: Could not open video capture."

        while True:
            ret, frame = cap.read()
            assert ret, "Error: Could not read frame."

            start_time = time()
            results = self.predict(frame)
            frame, _,_,_ = self.plot_bboxes(results, frame)
            end_time = time()

            fps = 1 / (end_time - start_time)
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 90), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

            cv2.imshow('YOLOv8 Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()




