'''
part of: cloud-image-segmentation
by: Daniel Casado Herraez

____________show.py____________
Display the results in an OpenCV window 
'''
import cv2
from config import *

def open_window(width, height):
    print("Opening new window with name ", WINDOW_NAME)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'Test')
