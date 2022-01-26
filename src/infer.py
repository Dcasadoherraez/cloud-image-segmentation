'''
part of: cloud-image-segmentation
by: Daniel Casado Herraez

____________infer.py____________
Complementary functions to infer_local.py to perform
image segmentation.
'''
# deep learning libraries
import cv2
import torch

# custom libraries
from labels import *
from config import *
from utils import *
from show import *

# load model from checkpoint into available device
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

# get computer camera video
def get_video_stream():
    return cv2.VideoCapture(0)

# get video from file
def get_file_video(filename):
    print('Opening video from file')
    stream = cv2.VideoCapture(filename) #create a opencv video stream.
    return stream

# test image segmentation on an image
def test_on_img(image_name, model, device):
    print("Testing on image")
    image = cv2.imread(image_name)

    res = score_frame_for_display(image, model, device)
    open_window(500, 500)

    cv2.imshow(WINDOW_NAME, res)
    cv2.waitKey(0)

# show the memory allocation
def show_memory_specs():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved

    print("Total memory:     ", t)
    print("Reseved memory:   ", r)
    print("Allocated memory: ", a)
    print("Free memory:      ", f)

# get the number of parameters of the model
def get_model_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp