#! /bin/bash

cd dockers/explaneatgpu
#cp dockers/explaneatgpu/dockerfile ../
#cd ../
docker build -t mikethemerry/explaneat-gpu-2 .
cd ../../
# rm dockerfile
# cd phd-neat-experiments
