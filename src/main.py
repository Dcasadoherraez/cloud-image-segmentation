'''
part of: cloud-image-segmentation
by: Daniel Casado Herraez

____________main.py____________
Manages command calls to the cloud-image-segmentation package

main.py [-h] [-t TRAIN] [-l LOCAL] [-i IMAGE_PATH] [-v VIDEO_PATH] [-d DISPLAY] [-s SERVER]
               [-c CLIENT] [-njetson JETSON] [-cam USE_CAM]

optional arguments:
  -h, --help       show this help message and exit
  -t TRAIN         train the network in this machine
  -l LOCAL         infer in local machine
  -i IMAGE_PATH    input image file for inference
  -v VIDEO_PATH    input video file for inference
  -d DISPLAY       display the output result
  -s SERVER        start TCP server
  -c CLIENT        start TCP client
  -njetson JETSON  use Nvidia jetson nano as server
  -cam USE_CAM     use computer camera
'''

# utility libraries
import numpy as np

# system libraries
import sys
from argparse import ArgumentParser, ArgumentTypeError
import os

# custom libraries
from infer_local import *
from utils import *
from tcp_client import *
from tcp_server import *

# set environment variables
os.environ['TORCH_HOME'] = HOME_PATH
os.environ["WANDB_RUN_GROUP"] = "image-seg"

        
# main function        
def main(argv):
    parser = ArgumentParser()
    parser.add_argument("-t", dest="train", required=False,
                    help="train the network in this machine", type=bool)
    parser.add_argument("-l", dest="local", required=False,
                    help="infer in local machine", type=bool)
    parser.add_argument("-i", dest="image_file", required=False,
                    help="input image file for inference", metavar="IMAGE_PATH",
                    type=is_valid_file)
    parser.add_argument("-v", dest="video_file", required=False,
                    help="input video file for inference", metavar="VIDEO_PATH", 
                    type=is_valid_file)
    parser.add_argument("-d", dest="display", required=False,
                    help="display the output result", type=bool)
    parser.add_argument("-s", dest="server", required=False,
                    help="start TCP server", type=bool)
    parser.add_argument("-c", dest="client", required=False,
                    help="start TCP client", type=bool)
    parser.add_argument("-njetson", dest="jetson", required=False,
                    help="use Nvidia jetson nano as server", type=bool)
    parser.add_argument("-cam", dest="use_cam", required=False,
                    help="use computer camera", type=bool)
    args = parser.parse_args()

    if args.local:
        print("Starting local inference module...")
        infer_local(args)
    elif args.train:
        print("Starting training module...")
        print("-- Using parameters from config.py --")
        # setup_and_train()
    elif args.server:
        start_server(args)
    elif args.client:
        start_client(args)
    else:
        raise ArgumentTypeError("No valid arguments")

# main call
if __name__ == "__main__": 
    main(sys.argv[1:])