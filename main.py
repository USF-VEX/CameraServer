import numpy as np
import cv2
import time

video_source = "tcp://192.168.1.92:8798"
cam = cv2.VideoCapture(video_source)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 2)

while True:
    ret, frame = cam.read()
    if not ret: continue
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)