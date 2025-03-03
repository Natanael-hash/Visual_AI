from src.data_prediction.yolo_predictions import ObjectDetection


detector = ObjectDetection(capture_index=0, data_yaml='/Users/natanaelhordon/Desktop/Visual-AI/src/data_preparation/data.yaml')
detector()
