#!/bin/bash
# bash build_and_push.sh torch 21.2
# bash build_and_push.sh tf 21.2
FRAMEWORK=$1
VER=$2
if [ "$FRAMEWORK" == "tf" ]
then
  docker build . -f docker/Dockerfile -t condortools/tf-base:$VER --build-arg BASE_VER=$VER --build-arg BASE_IMAGE=nvcr.io/nvidia/merlin/merlin-tensorflow-training
  docker push condortools/tf-base:$VER
elif [ "$FRAMEWORK" == "torch" ]
then
  docker build . -f docker/Dockerfile -t condortools/torch-base:$VER --build-arg BASE_VER=$VER --build-arg BASE_IMAGE=nvcr.io/nvidia/merlin/merlin-pytorch-training
  docker push condortools/torch-base:$VER
else
  echo "Invalid FRAMEWORK. Only tf or torch allowed."
fi

