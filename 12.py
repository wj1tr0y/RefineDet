import shutil
import os
img_dir = '/home/wangjilong/data/zhili_coco_posneg/ImageSet'
ann_dir = '/home/wangjilong/data/zhili_coco_posneg/Annotations'
de_ann_dir = '/home/wangjilong/data/hardexamples/Annotations'
de_img_dir = '/home/wangjilong/data/hardexamples/ImageSet'
with open('hardexample.txt', 'r') as hd:
    count = 0
    for i in hd.readlines():
        try:
            a = int(i[:12])
        except:
            shutil.copy(os.path.join(de_ann_dir, i[:-11] + '.json'), os.path.join(ann_dir, i[:-11] + '.json'))
            shutil.copy(os.path.join(de_img_dir, i[:-11] + '.jpg'), os.path.join(img_dir, i[:-11] + '.jpg'))
            count += 1
            print(count)
     
print('copy done')
