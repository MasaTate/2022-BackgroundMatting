FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y tzdata
# timezone setting
ENV TZ=Asia/Tokyo 


RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    git 

RUN apt-key del 3bf863cc
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN pip3 install torch torchvision
RUN pip3 install matplotlib
RUN pip3 install imageio
RUN pip3 install tqdm
RUN pip3 install imageio-ffmpeg
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

RUN pip3 install opencv-python

ENV LIBRARY_PATH /user/local/cuda/lib64/stubs