import cv2
from object_counting_api import ObjectCountingAPI

options = {"model": "yolo-tiny.cfg", "load": "yolo-tiny_10000.weights", "threshold": 0.6, "gpu": 1.0}
VIDEO_PATH = "test.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
counter = ObjectCountingAPI(options)

counter.count_objects_on_video(cap, show=True)
