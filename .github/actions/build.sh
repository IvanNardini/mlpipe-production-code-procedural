#!/bin/sh

IMAGE_NAME=$1

cd /test
echo "Building docker image..."

docker build -t $IMAGE_NAME --rm .