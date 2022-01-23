# FlowCloud: An Image Segmentation Pipeline for Edge Devices in the Cloud

Image segmentation is a computer vision field that
has been widely explored for different applications such as disease
detection, self-driving vehicles and aerial map creation. It’s main
concept is to extract information from the image by identifying
different regions that correspond to different objects or classes.
However, computing power plays an important role on being able
to perform this pixelwise inference, thus being a limiting factor
for small devices (wearables, phones, surveillance cameras...).
This project solves this limitation for internet-connected edge de-
vices by taking the image segmentation inference step to a remote
GPU-capable machine and transmitting the encoded video via a
TCP client-server network. The data flow pipeline is presented
from the video capturing step to the image segmentation.

## Preparation
1. Create docker image using Dockerfile (Check CUDA compatibility with your device)
2. chmod +rwx docker_run.sh

## Usage

```
./docker_run.sh [-h] [-t TRAIN] [-l LOCAL] [-i IMAGE_PATH] [-v VIDEO_PATH] [-d DISPLAY] [-s SERVER] [-c CLIENT]

optional arguments:
  -h, --help     show this help message and exit
  -t TRAIN       train the network in this machine
  -l LOCAL       infer in local machine
  -i IMAGE_PATH  input image file for inference
  -v VIDEO_PATH  input video file for inference
  -d DISPLAY     display the output result
  -s SERVER      start TCP server
  -c CLIENT      start TCP client

```

Run server:
```
./docker_run.sh -s TRUE
```

Run client with display:
```
./docker_run.sh -c1 -d1
```

Run client without display:
```
./docker_run.sh -c1
```

Run local inference on image:
```
./docker_run.sh -l1 -i image_test1.png
```

Run inference on video:
```
./docker_run.sh -l1 -v video.mp4
```

## Folder structure
```
├── docker
│   └── Dockerfile
├── docker_run.sh
├── FlowCloud.pdf
├── image_test1.png
├── models
│   ├── hub
│   │   └── checkpoints
│   │       └── deeplabv3_resnet50_coco-cd0a2569.pth
│   └── trim-pyramid
│       ├── 21_12_2021_08_04deeplabv3_resnet50_e16_of100.pth
│       └── 21_12_2021_08_04deeplabv3_resnet50_e16_of100.zip
├── README.md
├── src
│   ├── config.py
│   ├── dataset.py
│   ├── infer_local.py
│   ├── infer.py
│   ├── labels.py
│   ├── main.py
│   ├── __pycache__
│   │   ├── config.cpython-38.pyc
│   │   ├── dataset.cpython-38.pyc
│   │   ├── infer.cpython-38.pyc
│   │   ├── infer_local.cpython-38.pyc
│   │   ├── labels.cpython-38.pyc
│   │   ├── show.cpython-38.pyc
│   │   ├── tcp_client.cpython-38.pyc
│   │   ├── tcp.cpython-38.pyc
│   │   ├── tcp_server.cpython-38.pyc
│   │   ├── train_local.cpython-38.pyc
│   │   └── utils.cpython-38.pyc
│   ├── show.py
│   ├── tcp_client.py
│   ├── tcp.py
│   ├── tcp_server.py
│   ├── train_local.py
│   ├── train.py
│   └── utils.py
└── videoplayback.mp4
```