import cv2
import random

import numpy as np


# Use a phone ip camera on local address as input
cap = cv2.VideoCapture('http://192.168.1.54:8080/video')
# Use laptop camera
# cap = cv2.VideoCapture(0)


# Render captured video on a window
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break