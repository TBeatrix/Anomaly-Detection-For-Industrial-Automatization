FROM nvidia/cuda:11.7.1-base-ubuntu22.04

# Set Proxy ...

WORKDIR /autoencoder

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Budapest

RUN apt-get update 
RUN apt-get install python3.10 python3-pip -y
RUN apt-get install -y libgl1-mesa-glx libgl1 libglib2.0-0

COPY configs/requirements_autoencoder.txt requirements.txt
RUN pip install --upgrade pip
RUN apt-get update
RUN apt-get install  libsm6 libxext6  -y  
RUN pip install pyparsing
RUN pip install -r requirements.txt

CMD ["python3", "main.py"]
