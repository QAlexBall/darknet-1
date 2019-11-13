# Using

change Makefile
GPU=1
CUDNN=1
OPENCV=1



make
rt PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
