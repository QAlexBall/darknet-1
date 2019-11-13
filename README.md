# Using

change Makefile
GPU=1
CUDNN=1
OPENCV=1


```shell
export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
make

# run voc/tf_record_to_voc ...

./darknet detector train voc/voc.data voc/yolov3-voc.cfg (path_to_pretrain)

./darknet detector demo 
```
