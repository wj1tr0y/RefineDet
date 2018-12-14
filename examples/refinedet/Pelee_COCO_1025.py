from __future__ import print_function
import sys
sys.path.append("./python")
import caffe
from caffe.model_libs import *
from google.protobuf import text_format

import math
import os
import shutil
import stat
import subprocess

# Add extra layers on top of a "base" network (e.g. VGGNet or ResNet).
def AddExtraLayers(net, arm_source_layers=[], use_batchnorm=True):
    use_relu = True

    # Add additional convolutional layers.
    # 512/32: 16 x 16
    last_layer = net.keys()[-1]
    from_layer = last_layer

    # 512/64: 8 x 8
    from_layer = dense_block(net, from_layer, 6, 32, name='stage5', bottleneck_width=4)
    with_pooling=True
    transition_block(net, from_layer, 896, name='stage5_tb', with_pooling=with_pooling)

    arm_source_layers.reverse()
    num_p = 5
    for index, layer in enumerate(arm_source_layers):
        from_layer = layer
        out_layer = "TL{}_{}".format(num_p, 1)
        ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1)

        if num_p == 5:
            from_layer = out_layer
            out_layer = "TL{}_{}".format(num_p, 2)
            ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1)

            from_layer = out_layer
            out_layer = "P{}".format(num_p)
            ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1)

        else:
            from_layer = out_layer
            out_layer = "TL{}_{}".format(num_p, 2)
            ConvBNLayer(net, from_layer, out_layer, use_batchnorm, False, 256, 3, 1, 1)

            from_layer = "P{}".format(num_p+1)
            out_layer = "P{}-up".format(num_p+1)
            DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, False, 256, 2, 0, 2)

            from_layer = ["TL{}_{}".format(num_p, 2), "P{}-up".format(num_p+1)]
            out_layer = "Elt{}".format(num_p)
            EltwiseLayer(net, from_layer, out_layer)
            relu_name = '{}_relu'.format(out_layer)
            net[relu_name] = L.ReLU(net[out_layer], in_place=True)
            out_layer = relu_name

            from_layer = out_layer
            out_layer = "P{}".format(num_p)
            ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1)
        num_p = num_p - 1

    return net


### Modify the following parameters accordingly ###
# The directory which contains the caffe code.
# We assume you are running the script at the CAFFE_ROOT.
caffe_root = os.getcwd()

# Set true if you want to start training right after generating all files.
run_soon = True
# Set true if you want to load from most recently saved snapshot.
# Otherwise, we will load from the pretrain_model defined below.
resume_training = True
# If true, Remove old model files.
remove_old_models = False

# The database file for training data. Created by data/coco/create_data.sh
train_data = "examples/zhili_coco_posneg/zhili_coco_posneg_train_lmdb"
# The database file for testing data. Created by data/coco/create_data.sh
test_data = "examples/coco/coco_val_lmdb"
# Specify the batch sampler.
resize_width = 1024
resize_height = 1024
resize = "{}x{}".format(resize_width, resize_height)
batch_sampler = [
        {
                'sampler': {
                        },
                'max_trials': 1,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.1,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.3,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.5,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.7,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.9,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'max_jaccard_overlap': 1.0,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        ]
train_transform_param = {
        'mirror': True,
        'mean_value': [104, 117, 123],
        'force_color': True,
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': resize_height,
                'width': resize_width,
                'interp_mode': [
                        P.Resize.LINEAR,
                        P.Resize.AREA,
                        P.Resize.NEAREST,
                        P.Resize.CUBIC,
                        P.Resize.LANCZOS4,
                        ],
                },
        'distort_param': {
                'brightness_prob': 0.5,
                'brightness_delta': 32,
                'contrast_prob': 0.5,
                'contrast_lower': 0.5,
                'contrast_upper': 1.5,
                'hue_prob': 0.5,
                'hue_delta': 18,
                'saturation_prob': 0.5,
                'saturation_lower': 0.5,
                'saturation_upper': 1.5,
                'random_order_prob': 0.0,
                },
        'expand_param': {
                'prob': 0.5,
                'max_expand_ratio': 4.0,
                },
        'emit_constraint': {
            'emit_type': caffe_pb2.EmitConstraint.CENTER,
            }
        }
test_transform_param = {
        'mean_value': [104, 117, 123],
        'force_color': True,
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': resize_height,
                'width': resize_width,
                'interp_mode': [P.Resize.LINEAR],
                },
        }


# A learning rate for batch_size = 1, num_gpus = 1.
base_lr = 0.00004 #0.00004

# Modify the job name if you want.
job_name = "refinedet_pelee_{}".format(resize)
# The name of the model. Modify it if you want.
model_name = "coco_{}".format(job_name)

# Directory which stores the model .prototxt file.
save_dir = "models/ResNet/coco/{}".format(job_name)
# Directory which stores the snapshot of models.
snapshot_dir = "models/ResNet/coco/{}".format(job_name)
# Directory which stores the job script and log file.
job_dir = "jobs/ResNet/coco/{}".format(job_name)
# Directory which stores the detection results.
output_result_dir = "{}/data/RefineDet/coco/results/{}".format(os.environ['HOME'], job_name)

# model definition files.
train_net_file = "{}/train.prototxt".format(save_dir)
test_net_file = "{}/test.prototxt".format(save_dir)
deploy_net_file = "{}/deploy.prototxt".format(save_dir)
solver_file = "{}/solver.prototxt".format(save_dir)
# snapshot prefix.
snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
# job script path.
job_file = "{}/{}.sh".format(job_dir, model_name)

# Stores the test image names and sizes. Created by data/coco/create_list.sh
name_size_file = "data/coco/val2017_name_size.txt"
# The pretrained ResNet101 model from https://github.com/KaimingHe/deep-residual-networks.
# pretrain_model = "models/ResNet/coco/refinedet_resnet18_1024x1024/res18_1024x1024_5w.caffemodel"
# Stores LabelMapItem.
label_map_file = "data/zhili_coco_posneg/labelmap_coco.prototxt"

# MultiBoxLoss parameters.
num_classes = 2
share_location = True
background_label_id = 0
train_on_diff_gt = False
normalization_mode = P.Loss.VALID
code_type = P.PriorBox.CENTER_SIZE
ignore_cross_boundary_bbox = False
mining_type = P.MultiBoxLoss.MAX_NEGATIVE
neg_pos_ratio = 3.
loc_weight = (neg_pos_ratio + 1.) / 4.
multibox_loss_param = {
    'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
    'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
    'loc_weight': loc_weight,
    'num_classes': num_classes,
    'share_location': share_location,
    'match_type': P.MultiBoxLoss.PER_PREDICTION,
    'overlap_threshold': 0.5,
    'use_prior_for_matching': True,
    'background_label_id': background_label_id,
    'use_difficult_gt': train_on_diff_gt,
    'mining_type': mining_type,
    'neg_pos_ratio': neg_pos_ratio,
    'neg_overlap': 0.5,
    'code_type': code_type,
    'ignore_cross_boundary_bbox': ignore_cross_boundary_bbox,
    'objectness_score': 0.01,
    }
loss_param = {
    'normalization': normalization_mode,
    }

# parameters for generating priors.
# minimum dimension of input image
# min_dim = 1024
# res3b_relu ==> 64 x 64
# res4b_relu ==> 32 x 32
# res5b_relu ==> 16 x 16
# res5b_relu/conv1_2_relu ==> 8 x 8
arm_source_layers = ['stage2_tb', 'stage3_tb', 'stage4_tb', 'stage5_tb']
odm_source_layers = ['P2', 'P3', 'P4', 'P5']
min_sizes = [32, 64, 128, 256]
max_sizes = [[], [], [], []]
steps = [16, 32, 64, 128]
aspect_ratios = [[2], [2], [2], [2]]
# variance used to encode/decode prior bboxes.
if code_type == P.PriorBox.CENTER_SIZE:
  prior_variance = [0.1, 0.1, 0.2, 0.2]
else:
  prior_variance = [0.1]
flip = True
clip = False

# Solver parameters.
# Defining which GPUs to use.
gpus = "3,4,5,6,7"
gpulist = gpus.split(",")
num_gpus = len(gpulist)

# Divide the mini-batch to different GPUs.
batch_size = 100
accum_batch_size = 100
iter_size = accum_batch_size / batch_size
solver_mode = P.Solver.CPU
device_id = 0
batch_size_per_device = batch_size
if num_gpus > 0:
  batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))
  iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
  solver_mode = P.Solver.GPU
  device_id = int(gpulist[0])

if normalization_mode == P.Loss.NONE:
  base_lr /= batch_size_per_device
elif normalization_mode == P.Loss.VALID:
  base_lr *= 25. / loc_weight
elif normalization_mode == P.Loss.FULL:
  # Roughly there are 2000 prior bboxes per image.
  # TODO(weiliu89): Estimate the exact # of priors.
  base_lr *= 2000.

# Evaluate on whole test set.
num_test_image = 2693
test_batch_size = 1
test_iter = num_test_image / test_batch_size

solver_param = {
    # Train parameters
    'base_lr': base_lr,
    'weight_decay': 0.0005,
    'lr_policy': "multistep",
    'stepvalue': [100000, 150000, 340000],
    'gamma': 0.1,
    'momentum': 0.9,
    'iter_size': iter_size,
    'max_iter': 340000,
    'snapshot': 1000,
    'display': 10,
    'average_loss': 10,
    'type': "SGD",
    'solver_mode': solver_mode,
    'device_id': device_id,
    'debug_info': False,
    'snapshot_after_train': True,
    # Test parameters
    # 'test_iter': [test_iter],
    # 'test_interval': 10000,
    # 'eval_type': "detection",
    # 'ap_version': "11point",
    # 'test_initialization': False,
    }

# parameters for generating detection output.
det_out_param = {
    'num_classes': num_classes,
    'share_location': share_location,
    'background_label_id': background_label_id,
    'nms_param': {'nms_threshold': 0.50, 'top_k': 1000},
    'keep_top_k': 500,
    'confidence_threshold': 0.01,
    'code_type': code_type,
    'objectness_score': 0.01,
    }

# parameters for evaluating detection results.
det_eval_param = {
    'num_classes': num_classes,
    'background_label_id': background_label_id,
    'overlap_threshold': 0.5,
    'evaluate_difficult_gt': False,
    'name_size_file': name_size_file,
    }

### Hopefully you don't need to change the following ###
# Check file.
check_if_exist(train_data)
check_if_exist(test_data)
check_if_exist(label_map_file)
# check_if_exist(pretrain_model)
make_if_not_exist(save_dir)
make_if_not_exist(job_dir)
make_if_not_exist(snapshot_dir)

# Create train net.
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(train_data, batch_size=batch_size_per_device,
        train=True, output_label=True, label_map_file=label_map_file,
        transform_param=train_transform_param, batch_sampler=batch_sampler)

PeleeNetBody(net, from_layer='data')
AddExtraLayers(net, arm_source_layers, use_batchnorm=True)

print(net.to_proto())