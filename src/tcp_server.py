from threading import Thread
import cv2
from time import sleep
import signal
import sys
import os

from config import *
from tcp import *

def start_server(args):
    print("[SERVER] Starting on {} port {}".format(*server_connection))

    use_jetson = False

    if args.jetson:
        use_jetson = True

    # if using video file
    if args.video_file:
        if use_jetson:
            stream_video_file_from_jetson(args.video_file)
        else:
            stream_video_file(args.video_file)
    # if using camera
    elif args.use_cam:
        if use_jetson:
            stream_video_file_from_jetson(args.video_file)
        else:
            open_cam_pc(server_connection)
    
    
    

