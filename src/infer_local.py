# deep learning libraries
import cv2
import torch
import torch.nn as nn 
import torchvision 
from torchvision import transforms

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
from labels import *
from show import *
from infer import *
from config import *
from utils import *
from tcp import *

def infer_local(args):
    image_mode = False
    video_mode = False
    tcp_client_mode = False

    if args.image_file:
        image_mode = True
        image_file = args.image_file
        print("Image selected in ", image_file)
    elif args.video_file:
        video_mode = True
        video_file = args.video_file
        print("Video selected in ", video_file)
    elif args.client:
        tcp_client_mode = True
    
    display = False
    if args.display:
        display = True


    # Load pretrained model
    model, device = load_model()

    if image_mode:
        test_on_img(image_file, model, device)
        exit(0)

    if video_mode or tcp_client_mode: 
         
        if video_mode:      
            cap = get_file_video(video_file) 
        elif tcp_client_mode:
            if args.use_cam:
                cap = open_cam_pc()
            else:
                cap = get_tcp_video()

        assert cap.isOpened() 

        if display:
            open_window(500, 500)

        x_shape = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('Input shape is ' + str((x_shape, y_shape)))
        
        while True: # Run until stream is out of frames
            ret, frame = cap.read() # Read the first frame.
            if ret != True:
                break
            start_time = time() # We would like to measure the FPS.
            results = score_frame_for_display(frame, model, device, display) # Score the Frame
            
            
            # Display results
            if display:
                if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
                    print("Image could not be displayed")
                    break
                else:
                    cv2.imshow(WINDOW_NAME, results)
                    key = cv2.waitKey(10)
                    if key == 27:
                        print("System stopped")
                        break   
            else:
                del results

            end_time = time()
            fps = 1/np.round(end_time - start_time, 3) #Measure the FPS.
            print(f"Frames Per Second : {fps}")