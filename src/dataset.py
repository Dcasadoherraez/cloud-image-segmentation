'''
part of: cloud-image-segmentation
by: Daniel Casado Herraez

____________dataset.py____________
Class and functions for the Cityscapes dataset
'''

# deep learning libraries
import torch
import torch.nn as nn 
from torch.utils.data import Dataset

# image manipulation libraries

# utility libraries
import numpy as np

# system libraries
import gc
import os

# Custom pytorch dataset class
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
        # append all the image names 
        for i in os.listdir(os.path.join(self.image_base, self.split)):
            for image in os.listdir(os.path.join(self.image_base, self.split, i)):
                self.image_list.append(os.path.join(self.image_base, self.split, i, image))

        self.mask_list = []
        # append all the labelled ground truths
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

 # transform mask to tensor with integer values corresponding to class IDs
class PILToTensor:
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