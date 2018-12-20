import shutil
import os
img_dir = '/home/wangjilong/data/zhili_coco_posneg/ImageSet'
ann_dir = '/home/wangjilong/data/zhili_coco_posneg/Annotations'
hard_dir = '/home/wangjilong/data/hardexamples'
with open('hardexample.txt', 'r') as hd:
    for i in hd.readlines():
        shutil.copyfile(os.path.join(ann_dir, i[:-11] + '.json'), os.path.join(hard_dir, 'Annotations/'+i[:-11] + '.json'))
        shutil.copyfile(os.path.join(img_dir, i[:-11] + '.jpg'), os.path.join(hard_dir, 'ImageSet/'+i[:-11] + '.jpg'))
print('copy done')
