#!/bin/bash

docker build --no-cache -t mnist-classifier-image .

docker run \
    -v /$(pwd):/app \
    --name mnist-classifier-container \
    mnist-classifier-image
