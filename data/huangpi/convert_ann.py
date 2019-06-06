#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jilong Wang
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@Description: file content
@Date: 2019-04-19 13:33:38
@LastEditTime: 2019-04-19 14:02:25
'''
import argparse
from collections import OrderedDict
import json
import os
from pprint import pprint
import shutil
import sys
import cv2
sys.path.append(os.path.dirname(sys.path[0]))
from tqdm import tqdm

HOMEDIR = os.path.expanduser("~")
DATASETDIR = os.path.join(HOMEDIR, 'Data/pedestrian/huangpi/')
ann_dirs = os.listdir(os.path.join(DATASETDIR, 'json'))
print(ann_dirs)
if __name__ == "__main__":

    for ann_dir in ann_dirs:
        anno = json.load(open(os.path.join(DATASETDIR, 'json', ann_dir, 'multi_person.json'), 'r'))
        for obj in tqdm(anno):
            ann = obj['annotations']
            im_filename = obj['filename']
            with open(os.path.join(DATASETDIR, 'Annotations', im_filename[:-4] + '.json'), 'w') as new_ann:
                try:
                    img = cv2.imread(os.path.join(DATASETDIR, 'pic', ann_dir, im_filename))
                    height, width, _ = img.shape
                except:
                    print(os.path.join(DATASETDIR, 'pic', ann_dir, im_filename))
                    sys.exit(1)
                json_format = {"annotation": [], 
                                "image": {
                                    "file_name": im_filename,
                                    "height": height,
                                    "width": width}}
                del img
                for bbox in ann:
                    x = bbox['x'] if bbox['x'] > 0 else 0
                    y = bbox['y'] if bbox['y'] > 0 else 0
                    w = bbox['width'] if bbox['width'] > 0 else 0
                    h = bbox['height'] if bbox['height'] > 0 else 0
                    x, y, w, h = list(map(lambda x:int(x), [x, y, w, h]))
                    if w == 0 or h == 0:
                        print('Found irregular bbox annotation. This image has been skipped.{}/{}'.format(ann_dir, im_filename))
                        continue
                    if x > width:
                        x = width
                    if y > height:
                        y = height
                    if x + w > width:
                        w = width - x
                    if y + h > height:
                        h = height - y
                    temp = {"category_id": 1, "iscrowd": 0, "bbox": [x, y, w, h],}
                    json_format["annotation"].append(temp)
                new_ann.writelines(json.dumps(json_format, sort_keys=True, indent=2, ensure_ascii=False))
    
            shutil.copy(os.path.join(DATASETDIR, 'pic', ann_dir, im_filename), os.path.join(DATASETDIR, 'ImageSet', im_filename))
    img_names = [x[:-4] for x in os.listdir(os.path.join(DATASETDIR, 'ImageSet'))]
    img_names.sort()
    with open('huangpi.txt', 'w') as f:
        f.write("\n".join(img_names))
