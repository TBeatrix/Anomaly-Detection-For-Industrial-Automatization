FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Budapest
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


WORKDIR /SAA
RUN apt-get update 

ENV CUDA_HOME=/usr/local/cuda-11.3
ENV PATH=/usr/local/cuda-11.3/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:${LD_LIBRARY_PATH}

RUN apt-get update 
RUN apt-get install python3.9 python3-pip -y
RUN apt-get install -y libgl1-mesa-glx libgl1 libglib2.0-0
RUN apt-get install  libsm6 libxext6  -y  

RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install torch==1.8.1+cu101 torchvision==0.7.0+cu101  -f https://download.pytorch.org/whl/cu101/torch_stable.html


COPY GroundingDINO /SAA/GroundingDINO
COPY SAM /SAA/SAM
# If the cuda is not avaiable on the machine while build, change it or run these command later in the created repository.
WORKDIR /SAA/GroundingDINO
RUN pip install -e .

WORKDIR /SAA/SAM
RUN pip install -e .


WORKDIR /SAA
COPY requirements_SAA.txt /SAA/requirements_SAA.txt
RUN pip install -r requirements_SAA.txt

RUN pip install --upgrade diffusers[torch]


CMD ["python3", "main.py"]








# # Use an official CUDA runtime with Conda as a parent image
# FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
# # Proxy (BP0ECF01)
# ENV http_proxy=http://172.17.0.1:3128
# ENV https_proxy=http://172.17.0.1:3128
# ENV NO_PROXY=*.bosch.com,127.0.0.1
# ENV CUDA_HOME=/usr/local/cuda \
#      TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
#      SETUPTOOLS_USE_DISTUTILS=stdlib

# RUN conda update conda -y
# RUN conda install -c "nvidia/label/cuda-12.1.1" cuda -y
# ENV CUDA_HOME=/opt/conda/pkgs/cuda-12.1.0-0

# ARG DEBIAN_FRONTEND=noninteractive
# ENV TZ=Europe/Budapest
# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone



# WORKDIR /SAA





# RUN apt-get update 

# RUN apt-get install -y libgl1-mesa-glx libgl1 libglib2.0-0
# RUN apt-get install  libsm6 libxext6  -y  
# RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# COPY GroundingDINO /SAA/GroundingDINO
# COPY SAM /SAA/SAM
# WORKDIR /SAA/GroundingDINO
# RUN pip install -e .

# WORKDIR /SAA/SAM
# RUN pip install -e .


# WORKDIR /SAA
# COPY requirements_SAA.txt /SAA/requirements_SAA.txt
# RUN pip install -r requirements_SAA.txt

# RUN pip install --upgrade diffusers[torch]





# CMD ["python3", "main.py"]

