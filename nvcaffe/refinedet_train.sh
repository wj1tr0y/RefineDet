cd ~/RefineDet
./build/tools/caffe train \
--solver="models/VGGNet/refinedet_vgg16_320x320/solver.prototxt" \
--gpu all 2>&1 | tee models/VGGNet/refinedet_vgg16_320x320/refinedet_320x320.log

