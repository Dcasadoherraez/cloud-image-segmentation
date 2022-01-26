'''
part of: cloud-image-segmentation
by: Daniel Casado Herraez

____________tcp.py____________
Functions to send/receive video on the TCP server
'''
import cv2
import os

from config import *

# receive video from TCP server
def get_tcp_video():
    # create TCP client pipeline to receive H264 videos
    print("Getting TCP video on {} port {}".format(*client_connection))
    gst_str = 'tcpclientsrc host={} port={} do-timestamp=TRUE ! \
                capsfilter caps="application/x-rtp-stream" ! rtpstreamdepay ! \
                rtph264depay ! decodebin ! videoconvert ! \
                appsink sync=false'.format(*client_connection)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

# stream a video file from the Jetson Nano
def stream_video_file_from_jetson(filename):
    
    print("Streaming video file {} on {}:{}".format(filename, *server_connection))
    # create TCP server pipeline to send H264 videos
    launch_gstreamer = 'gst-launch-1.0 filesrc location={} do-timestamp=TRUE !  \
                        qtdemux ! decodebin ! videoscale ! \
                        nvvidconv ! videoconvert ! "video/x-raw,format=I420" ! \
                        x264enc byte-stream=TRUE tune=zerolatency ! \
                        "video/x-h264,alignment=au,stream-format=byte-stream" ! \
                        rtph264pay ! rtpstreampay ! tcpserversink host={} port={}'.format(filename, *server_connection)
    os.system(launch_gstreamer) 

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
