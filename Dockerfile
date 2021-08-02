#FROM nvidia/cuda:11.0-devel-ubuntu18.04-rc
#FROM nvidia/cuda:10.2-devel-ubuntu18.04
FROM nvidia/cuda:10.2-base

# Install curl
RUN apt-get update && apt-get install -y curl

# Install python3.6 and pip
RUN apt-get update && apt-get install -y \
    python3.7 \
    python3-pip \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \ 
    python3.7-dev

# Update pip
RUN python3.7 -m pip install --upgrade pip

#symlink for convenience
RUN ln -s /usr/bin/python3.7 /usr/bin/python

#Install other libraries from requirements.txt
COPY requirements.txt /tmp/
RUN cd /tmp/ && pip install -r requirements.txt

WORKDIR /home/run_pipeline
COPY ./ /home/run_pipeline/
RUN cd /home/run_pipeline/
RUN ls
# RUN python setup.py build develop

# CMD ["modelrun", "/data/lwll_datasets"]

ENTRYPOINT python full_detection_tasks.py
