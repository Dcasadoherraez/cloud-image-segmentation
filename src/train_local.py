'''
part of: cloud-image-segmentation
by: Daniel Casado Herraez

____________train_local.py____________
Setup the dataset and train on the local machine
'''
# deep learning libraries
import torch
import torch.nn as nn 

# utility libraries
import numpy as np
import wandb

# custom libraries
from dataset import *
from utils import *
from labels import *
from config import *
from train import *

def setup_and_train():
    print('Using PyTorch', torch.__version__)
    wandb_obj = wandb.init(project="image-segmentation")

    # train parameters
    batch_size = 5
    num_epochs = 100
    num_channels = 3
    learning_rate = 1e-4

    # create instance of dataset classes
    train_dataset = CityscapesCustom(CITYSCAPES_PATH, IMAGES_PATH, MASKS_PATH, 
                                    split = 'train', 
                                    mode = 'gtFine', 
                                    target_type = 'labelIds')

    val_dataset = CityscapesCustom(CITYSCAPES_PATH, IMAGES_PATH, MASKS_PATH, 
                                    split = 'val', 
                                    mode = 'gtFine', 
                                    target_type = 'labelIds')

    # load train and val data
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    # sample images to get sizes
    examples = iter(train_loader)
    example_data, example_targets = examples.next()
    show_img, show_mask = load_image_batch(example_data[0:2], example_targets[0:2], labels_dict, data_transform, target_transform, one_hot=False)
    size = show_img.size()

    # select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # batch_size defined earlier    
    img_h = size[2]
    img_w = size[3]

    print(f"Input image dimensions: ({list(size)})")

    if img_h <= 0 or img_w <= 0: 
        raise Exception("Dataset dimensions not valid")

    # with open("/root/deeplabv3/data/cityscapes/meta/class_weights.pkl", "rb") as file: # (needed for python3)
    #     class_weights = np.array(pickle.load(file))
    # class_weights = torch.from_numpy(class_weights)
    # class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()

    # # loss function
    # loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # GradScaler for fp16 training
    scaler = torch.cuda.amp.GradScaler()    
    criterion = nn.CrossEntropyLoss()

    # Get model and change classifier to the number of classes we have
    model_name = 'deeplabv3_resnet50'
    model = get_pretrained_model()

    params = add_weight_decay(model, l2_value=0.0001)
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    if LOAD_MODEL_TRAINING:
        checkpoint = torch.load(get_latest_model(SAVE_PATH))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model = model.to(device)

    # training loop
    train_eval(device, train_loader, val_loader,
            model, model_name, criterion, optimizer,
            scaler, num_epochs, batch_size, labels_dict,
            data_transform, target_transform, 
            num_channels, img_h, img_w, SAVE_PATH, wandb_obj)