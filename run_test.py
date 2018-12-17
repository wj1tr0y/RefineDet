'''
In this example, we will load a RefineDet model and use it to detect objects.
'''
import argparse
import os
import sys
import numpy as np
import skimage.io as io
import cv2
# Make sure that caffe is on the python path:
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import threading

def ShowResults(im_name, image_file, results, save_dir, threshold=0.6, save_fig=False):
    img = cv2.imread(image_file)
    for i in range(0, results.shape[0]):
        score = results[i, -2]
        if threshold and score < threshold:
            continue
        label = int(results[i, -1])
        if label != 1:
            continue
        name = str(label)
        xmin = int(round(results[i, 0] * img.shape[1]))
        ymin = int(round(results[i, 1] * img.shape[0]))
        xmax = int(round(results[i, 2] * img.shape[1]))
        ymax = int(round(results[i, 3] * img.shape[0]))
        coords = (xmin, ymin), xmax - xmin, ymax - ymin
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 3)
        display_text = '%s: %.2f' % (name, score)
        cv2.putText(img, display_text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255,255,255), thickness=3)
    if save_fig:
        cv2.imwrite(os.path.join(save_dir, im_name[:-4] + '_dets.jpg'), img)
        print 'Saved: ' + os.path.join(save_dir, im_name[:-4] + '_dets.jpg')

def get_output(det, name, img_dir, save_dir):
    for j in range(det.shape[2]//500):
        det_label = det[0, 0, 500*j:500*(j+1), 1]
        det_conf = det[0, 0, 500*j:500*(j+1), 2]
        det_xmin = det[0, 0, 500*j:500*(j+1), 3]
        det_ymin = det[0, 0, 500*j:500*(j+1), 4]
        det_xmax = det[0, 0, 500*j:500*(j+1), 5]
        det_ymax = det[0, 0, 500*j:500*(j+1), 6]
        result = np.column_stack([det_xmin, det_ymin, det_xmax, det_ymax, det_conf, det_label])

        # show result
        ShowResults(name[j], os.path.join(img_dir, name[j]), result, save_dir, 0.40, save_fig=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "run test and get all result")
    parser.add_argument("--gpuid",
        help = "The gpu chosen to run the model.", required=True)
    parser.add_argument("--out-dir",
        help = "The output directory where we store the result.", required=True)
    parser.add_argument("--test-set", 
        help = "which sets your wanna run test.", required=True)

    args = parser.parse_args()
    # gpu preparation
    assert len(args.gpuid) == 1, "You only need to choose one gpu. But {} gpus are chosen.".format(args.gpuid)
    caffe.set_device(int(args.gpuid))
    caffe.set_mode_gpu()

    
    save_dir = args.out_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # load model
    model_def = 'models/ResNet/coco/refinedet_resnet18_addneg_1024x1024/deploy.prototxt'
    model_weights = 'models/ResNet/coco/refinedet_resnet18_addneg_1024x1024/coco_refinedet_resnet18_addneg_1024x1024_iter_129000.caffemodel'
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    # image preprocessing
    img_resize = 1024
    batch_size = 50
    net.blobs['data'].reshape(batch_size, 3, img_resize, img_resize)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

    test_set = args.test_set
    test_set = test_set.split(',')
    for i in test_set:
        print('Processing test/{}/:'.format(i))
        img_dir = '../dataset/test/' + str(i)
        im_names = os.listdir(img_dir)
        im_names = [x for x in im_names if 'dets' not in x]
        total = len(im_names)
        names = []
        images = []
        threads = []
        for count, im_name in enumerate(im_names):
            if total - count < batch_size:
                batch_size = total - count
                net.blobs['data'].reshape(batch_size, 3, img_resize, img_resize)
            image_file = os.path.join(img_dir, im_name)
            image = caffe.io.load_image(image_file)
            transformed_image = transformer.preprocess('data', image)
            net.blobs['data'].data[count % batch_size, ...] = transformed_image
            names.append(im_name)
            if (count + 1) % batch_size == 0:
                for t in threads:
                    t.join()
                detections = net.forward()['detection_out']
                
                threads = []
                t = threading.Thread(target=get_output,
                    args=(detections, names, img_dir, save_dir))
                threads.append(t)
                t.start()
                names = []
