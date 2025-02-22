import cv2
import os
import numpy as np
import yaml
from yaml.loader import SafeLoader

INPUT_WH_YOLO = 640

with open("/Users/natanaelhordon/Desktop/Visual-AI/src/data_preparation/data.yaml", "r") as fr:
    data_yaml = yaml.load(fr, Loader=SafeLoader)

labels = data_yaml['names']
print(len(labels))

yolo = cv2.dnn.readNetFromONNX("/Users/natanaelhordon/Desktop/Visual-AI/src/train/weights/best.onnx")
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

image = cv2.imread("/Users/natanaelhordon/Desktop/Visual-AI/src/data_prediction/street_image.jpg")
image_resize = cv2.resize(image, (640,640))
blob = cv2.dnn.blobFromImage(image_resize, 1/255, swapRB=True, crop=False)
yolo.setInput(blob)
preds = yolo.forward()
print(preds.shape)
# detections = preds[0]
# boxes = []
# confidences = []
# classes = []
#
# image_w ,image_h = image.shape[:2]
# x_factor =   image_w / INPUT_WH_YOLO
# y_factor = image_h / INPUT_WH_YOLO
#
#
# for item in range(len(detections)):
#     row = detections[item]
#     confindence = row[4]
#
#     if confindence > 0.5:
#         class_scores = row[5:]
#         class_score = class_scores.max()
#         class_id = class_scores.argmax()
#
#         if class_score > 0.25:
#             cx, cy, w, h = row[0:4]
#             left = int((cx - 0.5 * w) * x_factor)
#             top = int((cy - 0.5 * h) * y_factor)
#             width = int(w * x_factor)
#             height = int(h * y_factor)
#
#             box = np.array([left, top, width, height])
#
#             confidences.append(confindence)
#             boxes.append(box)
#             classes.append(class_id)
#
# boxes_np = np.array(boxes).tolist()
# confidences_np = np.array(confidences).tolist()
#
# index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.5)
#
# for ind in index:
#     x, y, w, h = boxes_np[ind]
#     bb_conf = int(confidences_np[ind])
#     classes_id = classes[ind]
#     class_name = labels[classes_id]
#
#     text = f'{class_name}: {bb_conf}%'
#     print(text)
#
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 255, 255), -1)
#
#     cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)
#
# cv2.imshow('original',image)
# cv2.imshow('yolo_prediction',image_resize)
# cv2.waitKey(0)
# cv2.destroyAllWindows()






