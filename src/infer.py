# Functions to perform inference 

# deep learning libraries
import torch
import torch.nn as nn 
import torchvision 
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


def load_model():
    model = get_pretrained_model()

    if torch.cuda.is_available():
        device = 'cuda' 
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Memory specs: ")
        show_memory_specs()
    else:
        device = 'cpu'
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Using device ", device)
    model.to(device)
    print("The model has ", get_model_params(model), " parameters")
    model.eval()

    return model, device


def get_video_stream():
    return cv2.VideoCapture(0)


def get_file_video(filename):
    print('Opening video from file')
    stream = cv2.VideoCapture(filename) #create a opencv video stream.
    return stream

def test_on_img(image_name, model, device):
    image = cv2.imread(image_name)

    res = score_frame_for_display(image, model, device)
    open_window(500, 500)

    cv2.imshow(WINDOW_NAME, res)
    cv2.waitKey(0)


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
    
def apply_class2color(frame):
    size = frame.shape
    c = 3 # channels
    converted = np.zeros((size[0], size[1], c), dtype=np.float32)
    for x in range(size[0]):
        for y in range(size[1]):
            converted[x,y,:] = np.array(id2color[frame[x,y].item()][::-1])/255
    return converted

def show_memory_specs():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved

    print("Total memory:     ", t)
    print("Reseved memory:   ", r)
    print("Allocated memory: ", a)
    print("Free memory:      ", f)


def get_model_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp