cd /home/wangjilong/pedestrian/RefineDet/examples/refinedet
./build/tools/caffe train \
--solver="models/ResNet/coco/refinedet_resnet101_512x512/solver.prototxt" \
--weights="models/ResNet/ResNet-101-model.caffemodel" \
--gpu 0,1,2,3 2>&1 | tee jobs/ResNet/coco/refinedet_resnet101_512x512/coco_refinedet_resnet101_512x512.log
