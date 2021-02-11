#FROM nvidia/cuda:11.0-devel-ubuntu18.04-rc
FROM nvidia/cuda:10.2-devel-ubuntu18.04

# Install curl
RUN apt-get update && apt-get install -y curl

# Install python3.6 and pip
RUN apt-get update && apt-get install -y \
    python3.6 \
    python3-pip \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \ 
    python3.6-dev

# Update pip
RUN python3.6 -m pip install --upgrade pip

#symlink for convenience
RUN ln -s /usr/bin/python3.6 /usr/bin/python

#Install torch/torchvision and dependencies
# RUN pip install \
#     torch==1.3.1 \
#     pytorch-lightning==0.6.0 \
#     pillow==7.0.0 \
#     future==0.17.1 \
#     numpy==1.18.3 \
#     pandas \
#     fvcore \
#     cython \
#     opencv-python

# RUN git clone https://github.com/cocodataset/cocoapi.git
# RUN cd cocoapi/PythonAPI/ && python setup.py build develop

#Install other libraries from requirements.txt
COPY requirements.txt /tmp/
RUN cd /tmp/ && pip install -r requirements.txt

WORKDIR /home/run_pipeline
COPY ./ /home/run_pipeline/
RUN cd /home/run_pipeline/
RUN ls
RUN python setup.py build develop

CMD ["modelrun", "/data/lwll_datasets"]

ENTRYPOINT python full_detection_tasks.py
