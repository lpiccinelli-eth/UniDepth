FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set the timezone
ENV TZ Europe/Berlin
ARG DEBIAN_FRONTEND noninteractive

# Update the system timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set the working directory inside the container
ENV APP_WORKDIR /home/depth_estimation
WORKDIR $APP_WORKDIR

RUN apt update -y && apt upgrade -y 
RUN apt-get install libopenblas-dev liblapack-dev -y

# needed for cv2 
RUN apt-get install ffmpeg libsm6 libxext6 -y

# install pip
RUN apt install python3-pip -y
RUN python3 -m pip install --upgrade setuptools pip wheel

COPY ./requirements.txt $APP_WORKDIR/requirements.txt

# Install UniDepth and dependencies
RUN pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

EXPOSE 80

CMD ["/bin/bash"]
