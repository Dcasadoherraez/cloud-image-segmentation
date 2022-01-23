# deep learning libraries
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
from argparse import ArgumentParser
import gc
import os
import copy
from time import time

# custom libraries
from infer_local import *
from infer_tcp import *
from train_local import *
from tcp_client import *
from tcp_server import *

os.environ['TORCH_HOME'] = HOME_PATH
os.environ["WANDB_RUN_GROUP"] = "image-seg"

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg
        
def main(argv):
    parser = ArgumentParser()
    parser.add_argument("-t", dest="train", required=False,
                    help="train", type=bool)
    parser.add_argument("-l", dest="local", required=False,
                    help="infer local", type=bool)
    parser.add_argument("-i", dest="image_file", required=False,
                    help="input file with two matrices", metavar="IMAGE_PATH",
                    type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-v", dest="video_file", required=False,
                    help="input file with two matrices", metavar="VIDEO_PATH", 
                    type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-d", dest="display", required=False,
                    help="display the output result", type=bool)
    parser.add_argument("-s", dest="server", required=False,
                    help="start TCP server", type=bool)
    parser.add_argument("-c", dest="client", required=False,
                    help="start TCP client", type=bool)
    args = parser.parse_args()

    if args.local:
        print("Starting local inference module...")
        infer_local(args)
    elif args.train:
        print("Starting training module...")
        print("-- Using parameters from config.py --")
        setup_and_train()
    elif args.server:
        start_server()
    elif args.client:
        start_client()


if __name__ == "__main__": 
    main(sys.argv[1:])