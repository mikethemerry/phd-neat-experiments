#! /bin/bash

docker run \
    --name explaneatGPU \
    --mount type=bind,source="$(pwd)",target=/root/app \
    --mount type=bind,source="$(pwd)"/../../data,target=/root/data \
    --runtime=nvidia\
    --rm \
    -p 9999:9999 \
    -it \
    mikethemerry/explaneat-gpu ./app/runJupyter.sh
