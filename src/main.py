# deep learning libraries
import argparse
from getopt import getopt
import torch
import torch.nn as nn 
import torchvision 
from torchvision import transforms
import cv2

# image manipulation libraries
from PIL import Image

# utility libraries
import numpy as np

# system libraries
import sys
from argparse import ArgumentParser, ArgumentTypeError
import gc
import os
import copy
from time import time

# custom libraries
from infer_local import *
# from train_local import *
from tcp_client import *
from tcp_server import *

os.environ['TORCH_HOME'] = HOME_PATH
os.environ["WANDB_RUN_GROUP"] = "image-seg"

def is_valid_file(arg):
    print(arg)
    if not os.path.exists(str(arg)):
        raise argparse.ArgumentTypeError("{0} does not exist".format(arg))
    else:
        return arg
        
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
        raise argparse.ArgumentTypeError("No valid arguments")


if __name__ == "__main__": 
    main(sys.argv[1:])