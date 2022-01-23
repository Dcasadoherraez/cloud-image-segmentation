# deep learning libraries
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
import gc
import os
import copy
from time import time

# custom libraries
from labels import *
from config import *


def get_pretrained_model():
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    # Replace classifier
    model.classifier[4] = nn.Conv2d(
        in_channels = 256,
        out_channels = len(labels_dict),
        kernel_size =(1,1),
        stride=(1,1)
    )
    return model