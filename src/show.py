# Functions for displaying results
import cv2
from config import *

def open_window(width, height):
    print("Opening new window with name ", WINDOW_NAME)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'Test')

def read_cam(frame):
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(10)
        if key == 27:
            break
