import shutil
import os
img_dir = '/home/wangjilong/data/zhili_coco_posneg/ImageSet'
ann_dir = '/home/wangjilong/data/zhili_coco_posneg/Annotations'
de_ann_dir = '/home/wangjilong/data/hardexamples/Annotations'
de_img_dir = '/home/wangjilong/data/hardexamples/ImageSet'
fake_neg = '/home/wangjilong/data/neg/Annotations'
with open('hardexample.txt', 'r') as hd:
    count = 0
    fake = os.listdir(fake_neg)
    for i in hd.readlines():
        try:
            int(i[:12])
            # shutil.copy(os.path.join(ann_dir, i[:-11] + '.json'), os.path.join(de_ann_dir, i[:-11] + '.json'))
            # shutil.copy(os.path.join(img_dir, i[:-11] + '.jpg'), os.path.join(de_img_dir, i[:-11] + '.jpg'))
            # count += 1
            # print(count)
        except:
            # pass
            if i[:-11] + '.json' not in fake:
                shutil.copy(os.path.join(ann_dir, i[:-11] + '.json'), os.path.join(de_ann_dir, i[:-11] + '.json'))
                shutil.copy(os.path.join(img_dir, i[:-11] + '.jpg'), os.path.join(de_img_dir, i[:-11] + '.jpg'))
                count += 1
                print(count)
     
print('copy done')
