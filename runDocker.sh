#! /bin/bash

docker run \
    --name explaneatGPU \
    --mount type=bind,source="$(pwd)",target=/root/app \
    --runtime=nvidia\
    --rm \
    -p 9999:9999 \
    -it \
    mikethemerry/explaneatgpu
