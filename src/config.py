from torchvision import transforms
from dataset import PILToTensor

# train variables
DATASET_PATH = '/'
CITYSCAPES_PATH = DATASET_PATH + '/cityscapes'  

# Obtained Cityscapes mean and std
# mean = [0.2868, 0.3250, 0.2838]
# std = [0.1869, 0.1901, 0.1872]

# ImageNet normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Initialize the dataset
IMAGES_PATH = './cityscapes/leftImg8bit/'
MASKS_PATH = './cityscapes/gtFine/'
HOME_PATH = './models'
SAVE_PATH = 'checkpoints/'
LOAD_MODEL_TRAINING = False

# inference variables
WINDOW_NAME='Test'
HOME_PATH = './models'
MODEL_PATH = './models/trim-pyramid/21_12_2021_08_04deeplabv3_resnet50_e16_of100.pth'

# dataset transforms
image_smaller_size = 520
image_smaller_size = (250, 500) # if the GPU is low memory set smaller size
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(image_smaller_size),
    transforms.Normalize(mean, std)
])
target_transform = transforms.Compose([
    transforms.Resize(image_smaller_size),
    PILToTensor(),
])

# display transform
show_data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((250, 500)),
    transforms.ToTensor(),
])

# client and server parameters
client_connection = ('192.168.1.40', 5004)
server_connection = ('192.168.1.40', 5004)
cam_device = '/dev/video0'