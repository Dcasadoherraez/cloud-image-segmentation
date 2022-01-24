# Build using
# sudo docker build -t edge-image-segmentation .

# **************** https://github.com/anibali/docker-pytorch **************** 

FROM nvidia/cuda:10.2-base-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8.1 \
 && conda clean -ya

# CUDA 10.2-specific steps
# Pytorch version: https://anaconda.org/pytorch/pytorch/files
# Torchvision version: https://anaconda.org/pytorch/torchvision/files?version=0.11.2&type=&page=3
RUN conda install -y -c pytorch \
    cudatoolkit=10.2 \
    "pytorch=1.10.1=py3.8_cuda10.2_cudnn7.6.5_0" \
    "torchvision=0.11.2=py38_cu102" \
 && conda clean -ya

# Set the default command to python3
CMD ["python3"]

# *************************************************************************** 

## OpenCV compilation
# Set up time zone.
ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

# Install system libraries required by OpenCV.
RUN sudo apt-get update \
 && sudo apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 \
 && sudo rm -rf /var/lib/apt/lists/*

# Basic utilities
RUN sudo apt-get update && sudo apt-get install \
    build-essential \
    cmake \
    pkg-config \
    unzip \
    yasm \
    git \
    checkinstall -y

# Image libs
RUN sudo apt install \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavresample-dev -y

# Encoding libs
RUN sudo apt install \
    libxvidcore-dev \
    x264 \
    libx264-dev \
    libfaac-dev \
    libmp3lame-dev \
    libtheora-dev \
    libfaac-dev \
    libmp3lame-dev \
    libvorbis-dev \
    libdc1394-22 \
    libdc1394-22-dev \
    libxine2-dev \
    libv4l-dev \
    v4l-utils -y

# GUI
RUN sudo apt-get install libgtk-3-dev -y

# Gstreamer
RUN sudo apt-get install \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-doc \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio -y

# OpenCV can be compiled with gcc up to v8
RUN sudo apt remove gcc -y
RUN sudo apt-get install gcc-8 g++-8 -y
RUN sudo ln -s /usr/bin/gcc-8 /usr/bin/gcc
RUN sudo ln -s /usr/bin/g++-8 /usr/bin/g++
RUN sudo ln -s /usr/bin/gcc-8 /usr/bin/cc
RUN sudo ln -s /usr/bin/g++-8 /usr/bin/c++

WORKDIR /home/user/miniconda/
RUN git clone https://github.com/opencv/opencv.git
RUN git clone https://github.com/opencv/opencv_contrib.git
RUN cd opencv && \
    mkdir build && \
    cd build &&\
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D WITH_TBB=ON \
      -D WITH_EIGEN=ON \
      -D WITH_CUDA=OFF \
      -D WITH_CUDNN=OFF \
      -D OPENCV_DNN_CUDA=OFF \
      -D WITH_V4L=ON \
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D WITH_GSTREAMER=ON \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D OPENCV_EXTRA_MODULES_PATH=/home/user/miniconda/opencv_contrib/modules \
      -D BUILD_EXAMPLES=ON \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_python3=ON \
      -D BUILD_NEW_PYTHON_SUPPORT=ON \
      -D OPENCV_PYTHON3_VERSION=3.8.1 \
      -D PYTHON3_EXECUTABLE=$(which python3) \
      -D PYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
      -D PYTHON_INCLUDE_DIR2=$(python3 -c "from os.path import dirname; from distutils.sysconfig import get_config_h_filename; print(dirname(get_config_h_filename()))") \
      -D PYTHON_LIBRARY=$(python3 -c "from distutils.sysconfig import get_config_var;from os.path import dirname,join ; print(join(dirname(get_config_var('LIBPC')),get_config_var('LDLIBRARY')))") \
      -D PYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \
      -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
      .. && \
    make -j$(nproc) && \
    sudo make install && \
    sudo ldconfig

WORKDIR /app
RUN echo "DONE!"
