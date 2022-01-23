# deep learning libraries
import torch
import torch.nn as nn 
import torchvision 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F 

# image manipulation libraries
from PIL import Image

# utility libraries
import math
import numpy as np
from tqdm import tqdm # progress bar

# system libraries
import sys
import gc
import os
import copy

# Create custom pytorch dataset class
class CityscapesCustom(Dataset):
    def __init__(self, root, image_base, mask_base, 
              split = 'train', mode = 'gtFine', target_type = 'labelIds'):
        self.root = root
        self.image_base = image_base
        self.mask_base = mask_base
        self.split = split # "train" or "val"
        self.mode = mode # 'gtFine'
        self.target_type = target_type # "labelIds", "instanceIds", "color"

        self.image_list = []
        for i in os.listdir(os.path.join(self.image_base, self.split)):
            for image in os.listdir(os.path.join(self.image_base, self.split, i)):
                self.image_list.append(os.path.join(self.image_base, self.split, i, image))

        self.mask_list = []
        for i in os.listdir(os.path.join(self.mask_base, self.split)):
            for mask in os.listdir(os.path.join(self.mask_base, self.split, i)):
                self.mask_list.append(os.path.join(self.mask_base, self.split, i, mask))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = self.image_list[index].rstrip()
        mask_path = '_'.join(img_path.split('_')[:-1]) + '_' + self.mode + '_' + self.target_type + '.png'
        mask_path = mask_path.replace('leftImg8bit', self.mode)
 
        return img_path, mask_path

class PILToTensor:
  # Transform mask to tensor with integer values corresponding to class IDs
    def __call__(self, target):
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return target

# get mean and std of the dataset
def get_mean_std(loader):
    # var(x) = E(x²) - E(x)²
    # std(x) = sqrt(var(x))
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    num_norm_batches = 2
    count = 0
  
    for data, target in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3]) # [sample, height, width]
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
        gc.collect()
        count += 1
        if count == num_norm_batches:
            break

    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5

    return mean,std