import sys
from argparse import ArgumentParser

from show import *
from infer_local import *
from config import *

# receive video from TCP server
def get_tcp_video():
    # create TCP client pipeline to receive H264 videos
    print("Getting TCP camera on {} port {}".format(*client_connection))
    gst_str = 'tcpclientsrc host={} port={} do-timestamp=TRUE ! \
                capsfilter caps="application/x-rtp-stream" ! rtpstreamdepay ! \
                rtph264depay ! decodebin ! videoconvert ! \
                appsink sync=false'.format(*client_connection)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def start_client():
    display = False
    if args.display:
        display = True
        
    print("[CLIENT] Connecting to {} port {}".format(*client_connection))

    cap = get_tcp_video(client_connection)
    if not cap.isOpened():
        sys.exit('Failed to open camera!')

    # perform inference
    parser = ArgumentParser()
    parser.add_argument("-d", dest="display", type=bool)
    parser.add_argument("-c", dest="client", type=bool)
    
    args = parser.parse_args(["-display", "FALSE", "-c", "TRUE"])

    infer_local(args)


