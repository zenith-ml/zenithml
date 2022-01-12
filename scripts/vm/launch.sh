#!/bin/bash
while nvidia-smi -eq 0; do
    sleep 5
done
sudo docker run --gpus=all --rm -d --name zenith -p 8787:8787 -p 8999:8999 -p 8888:8080 -v /tmp/ml:/home/ml --ipc=host $CX_DOCKER_IMAGE