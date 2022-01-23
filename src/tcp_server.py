from threading import Thread
import cv2
from time import sleep
import signal
import sys
import os

from config import *
from tcp import *

def start_server():
    filename="videoplayback.mp4"
    # filename =""
    
    print("[SERVER] Starting on {} port {}".format(*server_connection))
    
    if (filename == ""):
        open_cam_pc(server_connection)
    else:
        stream_video_file(filename)

