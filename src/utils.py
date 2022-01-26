'''
part of: cloud-image-segmentation
by: Daniel Casado Herraez

____________utils.py____________
Common tools used by main, inference and training
'''

# deep learning libraries
import cv2
import torch
import torch.nn as nn 
import torchvision 

# utility libraries
import numpy as np
from argparse import ArgumentTypeError
import os

# custom libraries
from labels import *
from config import *

# check valid input
def is_valid_file(arg):
    print(arg)
    if not os.path.exists(str(arg)):
        raise ArgumentTypeError("{0} does not exist".format(arg))
    else:
        return arg

# get pretrained deeplabv3_resnet50 model
def get_pretrained_model():
    # download from hub if does not exist 
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)

    # Replace classifier
    model.classifier[4] = nn.Conv2d(
        in_channels = 256,
        out_channels = len(labels_dict),
        kernel_size =(1,1),
        stride=(1,1)
    )
    return model

# transform from class id to RGB color, used for display
def apply_class2color(frame):
    size = frame.shape
    c = 3 # channels
    converted = np.zeros((size[0], size[1], c), dtype=np.float32)
    for x in range(size[0]):
        for y in range(size[1]):
            converted[x,y,:] = np.array(id2color[frame[x,y].item()][::-1])/255
    return converted

# get output sample with masked labels
def score_frame_for_display(frame, model, device, display=True):
    t_frame = data_transform(frame)
    # print("New image shape is ", t_frame.shape)
    input_to_model = t_frame.unsqueeze(0).to(device)
    results = model(input_to_model)

    if display:
        mask = torch.argmax(results['out'][0], 0).detach().cpu().numpy()
        mask = apply_class2color(mask)
        original = show_data_transform(frame).permute(1, 2, 0).numpy()
        overlap = cv2.addWeighted(original,0.9,mask,0.5,0)
        numpy_horizontal_concat = np.concatenate((overlap, original), axis=0)

        return numpy_horizontal_concat
    else:
        return results