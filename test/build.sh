#!/bin/sh

MAGE_NAME=$1

cd ./procedural_ml_pipe
echo "Building docker image..."

docker build -t $IMAGE_NAME --rm .