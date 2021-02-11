## CORAL

This repository hosts USC ISI / CORAL teams systems for LWLL evaluations.

Documentation is available at: https://usc-isi.lollllz.com/coral/ 

This folder is for CORAL TEAM Object Detection System. Currently, this is the Dry Run version.

### Usage:

For building the docker for evaluation:

    docker build -t detection_pipeline .

For running the Object Detection Pipeline:

    docker run --rm -ti --gpus '"device=0"' --env-file env.list --ipc=host -v /lwll/evaluation:/lwll/evaluation detection_pipeline

Assuming LWLL* environment variables are correctly set. If the above command is used, then it will be read from `env.list` file.

Assuming all datasets are stored at `/lwll/evaluation/`.

The system uses only one GPU. Please make sure to give access to only one GPU using `--gpus '"device=0"'`.