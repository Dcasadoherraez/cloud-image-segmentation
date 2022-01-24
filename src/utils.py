# deep learning libraries
import cv2
import torch
import torch.nn as nn 
import torchvision 
from torchvision import transforms

# utility libraries
import numpy as np

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


def apply_class2color(frame):
    size = frame.shape
    c = 3 # channels
    converted = np.zeros((size[0], size[1], c), dtype=np.float32)
    for x in range(size[0]):
        for y in range(size[1]):
            converted[x,y,:] = np.array(id2color[frame[x,y].item()][::-1])/255
    return converted


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