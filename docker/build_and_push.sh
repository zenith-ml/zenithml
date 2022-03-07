#!/bin/bash
set -e
# bash ./docker/build_and_push.sh torch 0.0.0
# bash ./docker/build_and_push.sh tf 0.0.0
FRAMEWORK=$1
BASEIMAGE_VER=22.03
VER=$2
if [ "$FRAMEWORK" == "tf" ]
then
  docker build . -f docker/Dockerfile -t zenithml/zenithml-tf:$VER --build-arg BASE_VER=$BASEIMAGE_VER --build-arg BASE_IMAGE=nvcr.io/nvidia/merlin/merlin-tensorflow-training
  docker push zenithml/zenithml-tf:$VER
elif [ "$FRAMEWORK" == "torch" ]
then
  docker build . -f docker/Dockerfile -t zenithml/zenithml-torch:$VER --build-arg BASE_VER=$BASEIMAGE_VER --build-arg BASE_IMAGE=nvcr.io/nvidia/merlin/merlin-pytorch-training
  docker push zenithml/zenithml-torch:$VER
else
  echo "Invalid FRAMEWORK. Only tf or torch allowed."
fi

