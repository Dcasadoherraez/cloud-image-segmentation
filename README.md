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

The pipeline structure is the following:

<img src="https://github.com/Dcasadoherraez/cloud-image-segmentation/blob/main/media/pipeline.png" alt="Structure of the pipeline" height="400"/>


Some sample results of the image segmentation network:

![](https://github.com/Dcasadoherraez/cloud-image-segmentation/blob/main/media/results1.png) | ![](https://github.com/Dcasadoherraez/cloud-image-segmentation/blob/main/media/results2.png)

## Preparation
1. Create docker image using Dockerfile (Check CUDA compatibility with your device. When using Jetson Nano, packages were manually installed instead of using Docker)
2. chmod +rwx docker_run.sh
3. Setup corresponding parameters in config.py

## Usage

### Training 

### TCP Client/Server

* Setup IP addresses in config.py.
* Make sure ports are forwarded in router
```
./docker_run.sh [-h] [-t TRAIN] [-l LOCAL] [-i IMAGE_PATH] [-v VIDEO_PATH] [-d DISPLAY] [-s SERVER] [-c CLIENT] [-njetson JETSON] [-cam USE_CAM]

optional arguments:
  -h, --help       show this help message and exit
  -t TRAIN         train the network in this machine
  -l LOCAL         infer in local machine
  -i IMAGE_PATH    input image file for inference
  -v VIDEO_PATH    input video file for inference
  -d DISPLAY       display the output result
  -s SERVER        start TCP server
  -c CLIENT        start TCP client
  -njetson JETSON  use Nvidia jetson nano as server
  -cam USE_CAM     use computer camera


```

Run video server on PC:
```
./docker_run.sh -s TRUE -v videoplayback.mp4
```

Run camera server on PC (configure camera device in config.py):
```
./docker_run.sh -s TRUE -cam TRUE
```

Run video server on Nvidia Jetson Nano:
```
./docker_run.sh -s TRUE -njetson TRUE -v videoplayback.mp4
```

Run client with display:
```
./docker_run.sh -c TRUE -d TRUE
```

Run client without display:
```
./docker_run.sh -c TRUE
```

### Local inference

Run local inference on image:
```
./docker_run.sh -l TRUE -i image_test1.png
```

Run local inference on video:
```
./docker_run.sh -l TRUE -v video.mp4
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
│       └── 21_12_2021_08_04deeplabv3_resnet50_e16_of100.pth
├── README.md
├── src
│   ├── config.py
│   ├── main.py
│   └── ...
└── videoplayback.mp4
```

