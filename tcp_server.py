# ==========================================================
# video_pipeline.py
# This tool was made for testing GStreamer on Ubuntu 20.04
# webcam. The server thread creates the sending pipeline
# while the client thread creates the reciving pipeline. 
# Modify the device accordingly to your camera.
# ==========================================================

from threading import Thread
import cv2
from time import sleep
import signal
import sys
import os

from config import *


# stream a video file
def stream_video_file(filename):
    print("Streaming video file " , filename)
    # create TCP server pipeline to send H264 videos
    os.system(('gst-launch-1.0 filesrc location={} do-timestamp=TRUE ! decodebin ! timeoverlay valignment=1 halignment=1 ! \
                videoscale ! videoconvert ! "video/x-raw,format=I420" ! \
                x264enc byte-stream=TRUE tune=zerolatency ! \
                "video/x-h264,alignment=au,stream-format=byte-stream" ! \
                rtph264pay ! rtpstreampay ! tcpserversink host={} port={}').format(filename, *server_connection)) 

# stream pc cam
def open_cam_pc():
    # create TCP server pipeline to send H264 videos
    print('Using device: {}'.format(cam_device))
    os.system(('gst-launch-1.0 v4l2src device={} do-timestamp=TRUE ! timeoverlay valignment=1 halignment=1 ! \
                videoscale ! videoconvert ! "video/x-raw,format=I420,framerate=30/1" ! \
                x264enc byte-stream=TRUE tune=zerolatency ! \
                "video/x-h264,alignment=au,stream-format=byte-stream" ! \
                rtph264pay ! rtpstreampay ! tcpserversink host={} port={}').format(cam_device, *server_connection)) 

def test_on_vlc():
    # create TCP server pipeline to send H264 videos
    print('Using device: {}'.format(cam_device))
    os.system(('gst-launch-1.0 -v videotestsrc ! \
                x264enc key-int-max=12 byte-stream=true ! \
                mpegtsmux ! tcpserversink host={} port={} ').format(*server_connection)) 

def start_server():
    filename="videoplayback.mp4"
    # filename =""
    
    print("[SERVER] Starting on {} port {}".format(*server_address))
    
    if (filename == ""):
        open_cam_pc(server_address)
    else:
        stream_video_file(filename)

