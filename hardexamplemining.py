'''
@Author: Jilong Wang
@Date: 2019-01-05 14:44:14
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@LastEditTime: 2019-04-08 16:36:50
@Description: In this script, we will load a RefineDet model to detect pedestriancd .
'''
#coding:utf-8
from __future__ import print_function
import argparse
import os
import sys
import numpy as np
import skimage.io as io
import cv2
import time
import matplotlib.pyplot as plt
import json
os.environ['GLOG_minloglevel'] = '3'
# Make sure that caffe is on the python path:x
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import matplotlib.pyplot as plt
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2


class PeopleDetection:
    def __init__(self, modelDeployFile,  modelWeightsFile,  gpuid=0,  threshold=0.60,  img_resize=512, batch_size=25):
        caffe.set_device(int(gpuid))
        caffe.set_mode_gpu()
        self.img_resize = img_resize
        self.batch_size = batch_size
        self.threshold = threshold
        self.net = None
        self.transformer = None
        self.net = caffe.Net(modelDeployFile, modelWeightsFile, caffe.TEST)

        # detection image preprocessing
        self.net.blobs['data'].reshape(self.batch_size, 3, img_resize, img_resize)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

    def detect(self, img_dir):
        print('Processing {}:'.format(img_dir))

        # get all image names and sorted by name
        im_names = os.listdir(img_dir)
        im_names = [x.split('/')[-1][:-5]+'.jpg' for x in im_names if 'json' in x]

        im_names.sort()
        im_names = im_names[:52478]
        frame_result = []
        batch_size = self.batch_size
        names = []
        shapes = []
        last_c = 0
        total = len(im_names)
        for count, im_name in enumerate(im_names):
            image_file = os.path.join(img_dir, im_name)
            try:
                image = caffe.io.load_image(image_file)
            except:
                continue
            transformed_image = self.transformer.preprocess('data', image)
            self.net.blobs['data'].data[(count - last_c) % batch_size, ...] = transformed_image
            shapes.append(image.shape)
            print(count, total)
            names.append(im_name)
            if (count + 1 - last_c) % batch_size == 0:
                last_c = count + 1
                detections = self.net.forward()['detection_out']
                for i in range(batch_size):
                    det_label = detections[0, 0, 500*i:500*(i+1), 1]
                    det_conf = detections[0, 0, 500*i:500*(i+1), 2]
                    det_xmin = detections[0, 0, 500*i:500*(i+1), 3]
                    det_ymin = detections[0, 0, 500*i:500*(i+1), 4]
                    det_xmax = detections[0, 0, 500*i:500*(i+1), 5]
                    det_ymax = detections[0, 0, 500*i:500*(i+1), 6]

                    # print('processing {}'.format(names[i]), end='')
                    result = np.column_stack([det_xmin, det_ymin, det_xmax, det_ymax, det_conf, det_label])
                    frame_result.append((names[i], result, shapes[i]))
                names = []
                shapes = []
                # change batch_size when there is no enough images to fill initial batch_size
                if len(im_names) - count <= batch_size:
                    batch_size = len(im_names) - count - 1
        print('Detection done! Total:{} frames'.format(len(frame_result)))
        self.frame_result = frame_result

    def save_results(self, save_dir):
        for im_name, results, shape in self.frame_result:
            annotations = []
            for i in range(0, results.shape[0]):
                score = results[i, -2]
                if self.threshold and score < self.threshold:
                    continue
                label = int(results[i, -1])
                if label != 1:
                    continue
                name = str(label)
                xmin = int(round(results[i, 0] * shape[1]))
                ymin = int(round(results[i, 1] * shape[0]))
                xmax = int(round(results[i, 2] * shape[1]))
                ymax = int(round(results[i, 3] * shape[0]))
                annotations.append({'bbox': [xmin, ymin, xmax, ymax]})
            with open(os.path.join(save_dir, im_name[:-4]+'_dets.json'), 'w') as f:
                json.dump(annotations, f, indent=2)
                print('Saved: ' + os.path.join(save_dir, im_name[:-4] + '_dets.json'))

    def get_output(self, img_dir, save_dir):
        self.detect(img_dir)
        self.save_results(save_dir)


def net_init(batch_size, gpuid=0):
    '''
    @description: load detection & openpose & segementation models
    @param {None} 
    @return: three instances of det_net, op_net, seg_net
    '''
    # load detection model
    modelDeployFile = 'models/ResNet/coco/refinedet_resnet18_1024x1024/deploy.prototxt'
    modelWeightsFile = 'models/ResNet/coco/refinedet_resnet18_1024x1024/coco_refinedet_resnet18_1024x1024_iter_307000.caffemodel'
    det_net = PeopleDetection(modelDeployFile, modelWeightsFile, gpuid=gpuid, img_resize=1024, batch_size=batch_size, threshold=0.20)

    return det_net

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "run test and get all result")
    parser.add_argument("--gpuid",
        help = "The gpu chosen to run the model.", required=True)
    parser.add_argument("--save_dir",
        help = "The output directory where we store the result.", required=True)
    parser.add_argument("--test_set", 
        help = "which sets your wanna run test.", required=True)

    args = parser.parse_args()
    # gpu preparation
    assert len(args.gpuid) == 1, "You only need to choose one gpu. But {} gpus are chosen.".format(args.gpuid)
    
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    img_dir = args.test_set
    if not os.path.exists(img_dir):
        print("{} doesn't exists".format(img_dir))
        sys.exit(0)

    det = net_init(50, gpuid=args.gpuid)
    det.get_output(img_dir, save_dir)
