import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader

INPUT_WIDTH_YOLO = 640
INPUT_HEIGHT_YOLO = 640


with open("/Users/natanaelhordon/Desktop/Visual-AI/data_preparation/data.yaml", 'r') as fr:
    data_yaml = yaml.load(fr, Loader=SafeLoader)

labels = data_yaml["names"]

yolo = cv2.dnn.readNetFromONNX("/Users/natanaelhordon/Desktop/Visual-AI/data_prediction/train17/weights/best.onnx")
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

img = cv2.imread("street_image.jpg")
image = img.copy()
row, col, d = image.shape
max_rc = max(row, col)
input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
input_image[0:row, 0:col] = image
blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH_YOLO,INPUT_HEIGHT_YOLO), swapRB=True, crop=False)
yolo.setInput(blob)
preds = yolo.forward()

