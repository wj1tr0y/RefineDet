TensorRT inference 1ayer
1. Download and install TensorRT from https://developer.nvidia.com/tensorrt
2. Download and unzip https://github.com/NVIDIA/caffe/blob/models/RN50-FP16-20180201/weights.tar.gz into examples/TensorRT/
3. Dowload ImageNet (original JPG files) and make sure that examples/TensorRT/val-jpeg_map.txt points to the right place
4. Build Caffe with the following keys:
   cmake .. -DUSE_TRT=ON -DUSE_CUDNN=ON -DUSE_NCCL=ON
   You might try -DTRT_ROOT_DIR=<path to TRT> if fails to find it
5. Run
   caffe time --phase=TEST -model=examples/TensorRT/trt.prototxt -iterations=50000 -gpu=0
6. It should print something like

I0528 16:53:26.206513 22014 caffe.cpp:608] *** Benchmark begins ***
I0528 16:53:26.206521 22014 caffe.cpp:609] Testing for 50000 iterations.
I0528 16:53:36.082623 22014 caffe.cpp:528] Iterations: 0-10000 average forward-backward time: 0.987606 ms.
I0528 16:53:45.860116 22014 caffe.cpp:528] Iterations: 10000-20000 average forward-backward time: 0.977745 ms.
I0528 16:53:55.679301 22014 caffe.cpp:528] Iterations: 20000-30000 average forward-backward time: 0.981914 ms.
I0528 16:54:05.516075 22014 caffe.cpp:528] Iterations: 30000-40000 average forward-backward time: 0.983661 ms.
I0528 16:54:15.373744 22014 caffe.cpp:528] Iterations: 40000-50000 average forward-backward time: 0.985763 ms.
I0528 16:54:15.373767 22014 caffe.cpp:649] Average time per layer:
I0528 16:54:15.373770 22014 caffe.cpp:652]       data	forward: 0.00287586 ms.
I0528 16:54:15.373775 22014 caffe.cpp:655]       data	backward: 0.0001152 ms.
I0528 16:54:15.373777 22014 caffe.cpp:652]        TRT	forward: 0.979098 ms.
I0528 16:54:15.373780 22014 caffe.cpp:655]        TRT	backward: 0.00015376 ms.
I0528 16:54:15.373786 22014 caffe.cpp:660] Average Forward pass: 0.982514 ms.
I0528 16:54:15.373790 22014 caffe.cpp:662] Average Backward pass: 0.0005854 ms.
I0528 16:54:15.373792 22014 caffe.cpp:664] Average Forward-Backward: 0.98334 ms.
I0528 16:54:15.373795 22014 caffe.cpp:666] Total Time: 49167 ms.
I0528 16:54:15.373798 22014 caffe.cpp:667] *** Benchmark ends ***
I0528 16:54:15.547689 22014 caffe.cpp:671] *** Stats ***
I0528 16:54:15.547708 22014 caffe.cpp:672]
Labels:         1000
Iterations:     50005
Top 1 hits:     38169
Top 1 accuracy: 0.763304
Top 5 hits:     46534
Top 5 accuracy: 0.930587

Process finished with exit code 0
