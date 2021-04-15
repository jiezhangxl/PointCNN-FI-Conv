# FI-Conv: Feature interpolation convolution for point cloud analysis

Created by Jie Zhang, Jian Liu, Xiuping Liu, Wei Jiang, Junjie Cao, Kewei Tang.

## Introduction

FI-Conv is a simple and general framework for feature learning from point cloud. 
This code is based on the architecture of PointCNN. (PointCNN+FI-Conv)

* per voxel labelling accuracy on ScanNet (**85.1%**).results shown in table 4.

## PointCNN+FI-Conv Usage

The codeis implemented and tested with Tensorflow 1.12 in python3 scripts. 
It has dependencies on some python packages such as transforms3d, h5py, plyfile, and maybe more if it complains. Install these packages before the use of PointCNN+FI-Conv.

Here we list the commands for training/evaluating PointCNN+FI-Conv on segmentation tasks on ScanNet dataset.

  * #### ScanNet
  Please refer to [data_conversions](data_conversions/README.md) for downloading ScanNet, then:
  ```
  cd data_conversions
  python3 prepare_scannet_seg_data.py
  python3 prepare_scannet_seg_filelists.py
  cd ../pointcnn_seg
  ./train_val_scannet.sh -g 0 -x scannet_x8_2048_fps
  ./test_scannet.sh -g 0 -x scannet_x8_2048_fps -l ../../models/seg/FIConv+XConv+Random+Optimization_scannet_x8_2048_fps_xxxx/ckpts/iter-xxxxx -r 4
  cd ../evaluation
  python3 eval_scannet.py -d ../../data/scannet/seg/test -p ../../data/scannet/seg/scannet_test.pickle
  ```
  ####
  we also provide a Pretrained model.  Before using it, you need to install "git-lfs" first, and then clone the remote repository to the local through "git lfs clone https://github.com/jiezhangxl/PointCNN-FI-Conv.git".
  Then, You can use it by the following code.
  ```
  ./test_scannet.sh -g 0 -x scannet_x8_2048_fps -l ../model/iter-232000 -r 4
  cd ../evaluation
  python3 eval_scannet.py -d ../../data/scannet/seg/test -p ../../data/scannet/seg/scannet_test.pickle
  ```
