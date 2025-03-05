import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import yaml
from yaml.loader import SafeLoader

class ObjectDetection:
    def __init__(self, capture_index, data_yaml):
        self.colors = None
        self.capture_index = capture_index

        self.device = 'mps' if torch.mps.is_available() else 'cpu'
        self.model = self.load_model()

        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.nc = data_yaml['nc']

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
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxys[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                self.colors = self.generate_colors(class_id)

                xyxys.append((x1, y1, x2, y2))
                confidences.append(confidence)
                class_ids.append(class_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors, 2)
                label = f'{self.model.names[class_id]} {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors, 2)

        return frame, xyxys, confidences, class_ids

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened(), "Error: Could not open video capture."

        while True:
            ret, frame = cap.read()
            assert ret, "Error: Could not read frame."

            start_time = time()
            results = self.predict(frame)
            frame, _, _, _ = self.plot_bboxes(results, frame)
            end_time = time()

            fps = 1 / (end_time - start_time)
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 5)

            cv2.imshow('YOLOv8 Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    def generate_colors(self, ID):
        np.random.seed(10)
        colors = np.random.randint(100, 255, size=(self.nc, 3)).tolist()
        return tuple(colors[ID])


