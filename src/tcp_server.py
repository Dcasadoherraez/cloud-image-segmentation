'''
part of: cloud-image-segmentation
by: Daniel Casado Herraez

____________tcp_server.py____________
Initialize the TCP server for local inference
'''
from config import *
from tcp import *

def start_server(args):
    print("[SERVER] Starting on {} port {}".format(*server_connection))

    use_jetson = False

    if args.jetson:
        use_jetson = True

    # if using video file
    if args.video_file:
        print("Streaming video ", args.video_file)
        if use_jetson:
            stream_video_file_from_jetson(args.video_file)
        else:
            stream_video_file(args.video_file)
    # if using camera
    elif args.use_cam:
        print("Straming camera ", cam_device)
        if use_jetson:
            stream_video_file_from_jetson(args.video_file)
        else:
            open_cam_pc()
    
    
    

