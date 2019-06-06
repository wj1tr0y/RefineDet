#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jilong Wang
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@Description: file content
@Date: 2019-03-14 11:13:35
@LastEditTime: 2019-03-14 11:23:41
'''
import os
import cv2
import matplotlib
import json
import numpy as np
from tqdm import tqdm

#ann_list = list(open('hardexample.txt', 'r').readlines())
#ann_list = list(map(lambda x:x.strip('\n'), ann_list))
ann_list = [x for x in os.listdir('./easycoco_out') if 'json' in x]
ann_list = np.random.choice(ann_list, 1000)

img_list = [x[:-10]+'.jpg' for x in ann_list]
save_dir = './easycoco_jpg'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
pbar = tqdm(total=1000)
for i in range(1000):
    img_dir = os.path.join('/home/wangjilong/data/easycoco/ImageSet', img_list[i])
    ann_dir = os.path.join('./easycoco_out', ann_list[i])
    ann = json.load(open(ann_dir, 'r'))
    img = cv2.imread(img_dir, cv2.IMREAD_COLOR)

    for j in range(len(ann)):
        xmin = int(ann[j]['bbox'][0])
        ymin = int(ann[j]['bbox'][1])
        xmax = int(ann[j]['bbox'][2])
        ymax = int(ann[j]['bbox'][3])
        coords = (xmin, ymin), xmax - xmin, ymax - ymin
        person = img[ymin:ymax, xmin:xmax]
        cv2.imwrite('seg/'+img_list[i][:-4]+str(xmin)+str(ymin)+'.jpg',person)
#        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
#        display_text = '%s' % ('Person')
#        cv2.putText(img, display_text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(255,0,0), thickness=2)

#    cv2.imwrite(os.path.join(save_dir, img_list[i][:-4] + '_dets.jpg'), img)
    pbar.set_description('Saved: {}'.format(os.path.join(save_dir, img_list[i][:-4]+'_dets.jpg')))
    pbar.update(1)
