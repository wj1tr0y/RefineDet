A higher performance [NVCaffe](https://github.com/nvidia/caffe) implementation of [Single-Shot Refinement Neural Network for Object Detection](https://arxiv.org/abs/1711.06897 ). This code is based on NVCaffe version 17.0.2. The official and original Caffe code can be found [here](https://github.com/sfzhang15/RefineDet). 

### Table of Contents
- <a href='#performance'>Performance</a>
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-refinedet'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#todo'>Future Work</a>
- <a href='#references'>Reference</a>

&nbsp;
&nbsp;
&nbsp;
## Performance
This code is verified on Ubuntu 16.04 LTS 64bit, CUDA 9.0 and cudnn 7.0. 

#### VOC2007 Test 

##### mAP (*Single Scale Test*)

| Arch | Paper | Caffe Version | Our NVCaffe Version |
|:-:|:-:|:-:|:-:|
| RefineDet320 | 80.0% | 79.52% | 79.98% |
| RefineDet512 | 81.8% | 81.85% | 81.8% | 
| RefineDet320 from Scratch | - | - | 72.27% |


## Installation
- Clone this repository.
- Then download the dataset by following the [instructions](#datasets) below.

```Shell
make -j{cpu_core_num} all 
make pycaffe
```

## Training RefineDet
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:              https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
- By default, we assume you have downloaded the file in the `RefineDet.PyTorch/weights` dir:

```Shell
cd models/VGGNet
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- To train RefineDet320 or RefineDet512 using the train scripts `train_refinedet320.sh` and `train_refinedet512.sh`. You can manually change them as you want.

```Shell
./refinedet_train.sh  
```

- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * For instructions on Visdom usage/installation, see the <a href='#installation'>Installation</a> section.


## Evaluation
To evaluate a trained network:

```Shell
python test/test_refinedet.py 
```

## TODO


## References
- [Original Implementation (CAFFE)](https://github.com/sfzhang15/RefineDet)
- [NVCAFFE](https://github.com/nvidia/caffe)
- A list of other great SSD ports that were sources of Readme.md:
  * [luuuyi/RefineDet.PyTorch](https://github.com/luuuyi/RefineDet.PyTorch/edit/master/README.md
)

# NVCaffe

NVIDIA Caffe ([NVIDIA Corporation &copy;2017](http://nvidia.com)) is an NVIDIA-maintained fork
of BVLC Caffe tuned for NVIDIA GPUs, particularly in multi-GPU configurations.
Here are the major features:
* **16 bit (half) floating point train and inference support**.
* **Mixed-precision support**. It allows to store and/or compute data in either 
64, 32 or 16 bit formats. Precision can be defined for every layer (forward and 
backward passes might be different too), or it can be set for the whole Net.
* **Layer-wise Adaptive Rate Control (LARC) and adaptive global gradient scaler** for better
 accuracy, especially in 16-bit training.
* **Integration with  [cuDNN](https://developer.nvidia.com/cudnn) v7**.
* **Automatic selection of the best cuDNN convolution algorithm**.
* **Integration with v2.2 (or higher) of [NCCL library](https://github.com/NVIDIA/nccl)**
 for improved multi-GPU scaling.
* **Optimized GPU memory management** for data and parameters storage, I/O buffers 
and workspace for convolutional layers.
* **Parallel data parser, transformer and image reader** for improved I/O performance.
* **Parallel back propagation and gradient reduction** on multi-GPU systems.
* **Fast solvers implementation with fused CUDA kernels for weights and history update**.
* **Multi-GPU test phase** for even memory load across multiple GPUs.
* **Backward compatibility with BVLC Caffe and NVCaffe 0.15 and higher**.
* **Extended set of optimized models** (including 16 bit floating point examples).
* _Experimental feature (no official support)_ **Multi-node training** (since v0.17.1, NCCL 2.2 and OpenMPI 2 required).
* _Experimental feature (no official support)_ **TRTLayer** (since v0.17.1, can be used as inference plugin).

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
   
## Contributions

Please read, sign and attach enclosed agreement **NVIDIA_CLA_v1.0.1.docx**
to your PR.

## Useful notes

Libturbojpeg library is used since 0.16.5. It has a packaging bug. Please execute the following (required for Makefile, optional for CMake):
```
sudo apt-get install libturbojpeg
sudo ln -s /usr/lib/x86_64-linux-gnu/libturbojpeg.so.0.1.0 /usr/lib/x86_64-linux-gnu/libturbojpeg.so
```
