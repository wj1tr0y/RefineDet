import shutil
import os
img_dir = '/home/wangjilong/data/zhili_coco_posneg/ImageSet'
ann_dir = '/home/wangjilong/data/zhili_coco_posneg/Annotations'
de_ann_dir = '/home/wangjilong/data/zhili_coco_posneg/COCO_useless_annotations'
with open('hardexample.txt', 'r') as hd:
    count = 0
    for i in hd.readlines():
        try:
            a = int(i[:12])
            shutil.move(os.path.join(ann_dir, i[:-11] + '.json'), os.path.join(de_ann_dir, i[:-11] + '.json'))
            count += 1
            print(count)
        except:
            pass
     
print('copy done')
