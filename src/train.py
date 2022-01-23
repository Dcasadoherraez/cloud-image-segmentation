# deep learning libraries
import torch
import torch.nn as nn 
import torchvision 
from torchvision import transforms, io
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F 
import cv2

# image manipulation libraries
from PIL import Image

# utility libraries
import math
import numpy as np
from tqdm import tqdm # progress bar
import wandb

# system libraries
import sys
import gc
import os
import copy
from datetime import datetime

# custom libraries
from utils import *
from config import *

def load_image_batch(image_names, label_names, labels_dict, 
                     data_transform=None, target_transform=None, 
                     one_hot = False):
    
    # load image batch and perform transformations
    if data_transform == None or target_transform == None:
        raise Exception("You must provide a transform")

    images = []
    masks = []
    
    for idx in range(0, len(image_names)):
        with Image.open(image_names[idx]) as new_img:
            new_img = data_transform(new_img)
            images.append(new_img)
          
        with Image.open(label_names[idx]) as new_mask:
            new_mask = target_transform(new_mask)
            if one_hot:
                new_mask = torch.nn.functional.one_hot(
                    new_mask,
                    len(labels_dict),
                )
            masks.append(new_mask)
        gc.collect()
        
    images = torch.stack(images)
    masks = torch.stack(masks)

    return images.clone().detach(), masks.clone().detach()

def save_checkpoint(state_dict, path):
    torch.save(state_dict, path)

def get_latest_model(path):
    files = os.listdir(path)
    files.sort()
    return files[-1]

def compare_size(t1, t2):
    l1 = list(t1)
    l2 = list(t2)

    if len(l1) != len(l2):
        return False

    for i in range(len(l1)):
        if l1[i] != l2[i]:
            return False

    return True

def add_weight_decay(net, l2_value, skip_list=()):
    # https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/
    # https://github.com/fregu856/deeplabv3/blob/master/utils/utils.py
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_value}]


def score_frame(frame, model, device):
    t_frame = data_transform(frame)
    # print("New image shape is ", t_frame.shape)
    input_to_model = t_frame.unsqueeze(0).to(device)
    results = model(input_to_model)
    mask = torch.argmax(results['out'][0], 0).detach().cpu().numpy()
    mask = apply_class2color(mask)
    original = data_transform(frame).permute(1, 2, 0).numpy()
    overlap = cv2.addWeighted(original,0.7,mask,0.8,0)
    return overlap


def train_eval(device, train_loader, val_loader,
               model, model_name, criterion, optimizer,
               scaler, num_epochs, batch_size, labels_dict,
               data_transform, target_transform, 
               num_channels, img_h, img_w, save_path, wandb_obj):

    n_train = len(train_loader)
    n_val = len(val_loader)

    dataloaders = {'train': train_loader, 'val': val_loader}
    losses = {'train': [], 'val': []}
    accuracies = {'train': [], 'val': []}

    n_running = 10

    

    for epoch in range(num_epochs):
        dir_name = datetime.now().strftime("%d_%m_%Y_%H_%M")
        print("Epoch: ", epoch)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                print("Training...")
                model.train()
            else:
                print("Evaluating...")
                model.eval()
            loop = tqdm(dataloaders[phase])

            running_loss_total = 0.0
            running_acc_total = 0.0

            val_loss_total = 0.0
            val_acc_total = 0.0

            # Iterate over data
            for i, (image_paths, label_paths) in enumerate(loop):
                images, labels = load_image_batch(image_paths, label_paths, labels_dict, data_transform, target_transform, one_hot=False)
                if not compare_size(images.size(), torch.Size([batch_size, num_channels, img_h, img_w])):
                    print("[ERROR] Image not valid")
                    continue
                if not compare_size(labels.size(), torch.Size([batch_size, img_h, img_w])):
                    print("[ERROR] Label not valid")
                    continue

                images = images.to(device)
                labels = labels.to(device)

                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast(): # to use float16 training
                        outputs = model(images)['out']
                        loss = criterion(outputs, labels)
                
                    # find the running loss and accuracy
                    running_loss = loss.item()
                    _, predictions = torch.max(outputs, 1) # get indices of predicted classes
                    running_acc = (predictions == labels).sum().item()/(img_h*img_w*batch_size) # image size * batch_size

                    losses[phase].append(running_loss)
                    accuracies[phase].append(running_acc) 

                    running_loss_total += running_loss
                    running_acc_total += running_acc

                    val_loss_total += running_loss
                    val_acc_total += running_acc

                    # backward
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        if (i + 1) % n_running == 0:
                            wandb_image = score_frame(images[0], model, device)
                            wandb_obj.log({'Train accuracy': running_acc_total/n_running, 'Train loss': running_loss_total/n_running})
                            # wandb_obj.load("img": [wandb.Image(wandb_image , caption="Training image")])
                            running_loss_total = 0.0
                            running_acc_total = 0.0
                            

                        loop.set_postfix({'loss': running_loss, 'acc': running_acc})

            if phase == 'val':
                wandb_obj.log({'Val accuracy': val_acc_total/n_val, 'Val loss': val_loss_total/n_val})
                loop.set_postfix({'acc': val_acc_total/n_val, 'loss': val_loss_total/n_val})
                
                val_loss_total = 0.0
                val_acc_total = 0.0
        
        if epoch % 20: 
            print( "saving model...")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, os.path.join(save_path, dir_name + model_name + '_e' + str(epoch) + '_of' + str(num_epochs) + '.pth'))
