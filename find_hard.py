import threading
import json
import os
import threading
import shutil

def compute_iou(rec1, rec2):
    one_x, one_y, one_w, one_h = rec1
    two_x, two_y, two_w, two_h = rec2
    if((abs(one_x - two_x) < ((one_w + two_w) / 2.0)) and (abs(one_y - two_y) < ((one_h + two_h) / 2.0))):
        lu_x_inter = max((one_x - (one_w / 2.0)), (two_x - (two_w / 2.0)))
        lu_y_inter = min((one_y + (one_h / 2.0)), (two_y + (two_h / 2.0)))

        rd_x_inter = min((one_x + (one_w / 2.0)), (two_x + (two_w / 2.0)))
        rd_y_inter = max((one_y - (one_h / 2.0)), (two_y - (two_h / 2.0)))

        inter_w = abs(rd_x_inter - lu_x_inter)
        inter_h = abs(lu_y_inter - rd_y_inter)

        inter_square = inter_w * inter_h
        union_square = (one_w * one_h) + (two_w * two_h) - inter_square

        calcIOU = inter_square / union_square * 1.0
        return calcIOU
    else:
        return 0

def find_hard(det_names, count):
    hard_name = []
    for det in det_names:
        det_file = os.path.join(det_dir, det)
        ann_file = os.path.join(ann_dir, det[:-10]+'.json')

        ann = json.load(open(ann_file, 'r'))
        result = json.load(open(det_file, 'r'))
        result = result['results']
        ann = ann['annotation']

        for res in result:
            res['count'] = 0
        for gt in ann:
            gt['count'] = 0
                 
        for i in range(len(result)):
            bbox2 = result[i]['bbox']
            rect2 = [bbox2[0], bbox2[1], bbox2[2], bbox2[3]]
            max_iou = (-1, 0)
            for j in range(len(ann)):
                bbox = ann[j]['bbox']
                rect = [bbox[0], bbox[1], bbox[2], bbox[3]]
                if compute_iou(rect, rect2) > max_iou[1] and compute_iou(rect, rect2) > 0.5:
                    max_iou = (j, compute_iou(rect, rect2))
            if max_iou[0] != -1:
                ann[max_iou[0]]['count'] += 1
                result[i]['count'] += 1


        multi_bbox = 0.
        mismatch_bbox = 0.
        lost_bbox = 0.
        for gt in ann:
            if gt['count'] > 1:
                multi_bbox += 1
            if gt['count'] == 0:
                lost_bbox +=1
        for res in result:
            if res['count'] == 0:
                mismatch_bbox += 1
        if len(ann) == 0:
            if mismatch_bbox > 3:
                hard_name.append(det)
        elif mismatch_bbox/len(ann) > 0.5 or multi_bbox/len(ann) > 0.5 or lost_bbox/len(ann) > 0.5:
            hard_name.append(det)
    with open('thread{}'.format(count), 'w') as f:
        f.writelines('\n'.join(hard_name))
    print('Done')


if __name__ == "__main__":
    img_dir = '/home/wangjilong/data/zhili_coco_posneg/ImageSet'
    ann_dir = '/home/wangjilong/data/zhili_coco_posneg/Annotations'
    hard_dir = '/home/wangjilong/data/hardexamples'
    det_dir = './detout'
    det_name = os.listdir(det_dir)
    threads = []
    for j, i in enumerate(range(0, len(det_name), 10000)):
        t = threading.Thread(target=find_hard, args=(det_name[i: i+10000], j))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    
    name = os.listdir('.')
    name = [x for x in name if 'thread' in x]
    with open('hardexample.txt', 'w') as hd:
        for i in name:
            with open(i, 'r') as f:
                for line in f.readlines():
                    hd.write(line)
            hd.write('\n')
    
# img_dir = '/home/wangjilong/data/zhili_coco_posneg/ImageSet'
# ann_dir = '/home/wangjilong/data/zhili_coco_posneg/Annotations'
# hard_dir = '/home/wangjilong/data/hardexamples'
# with open('hardexample.txt', 'r') as hd:
#     for i in hd.readlines():
#         shutil.copyfile(os.path.join(ann_dir, i[:-11] + '.json'), os.path.join(hard_dir, 'Annotations/'+i[:-11] + '.json'))
#         shutil.copyfile(os.path.join(img_dir, i[:-11] + '.jpg'), os.path.join(hard_dir, 'ImageSet/'+i[:-11] + '.jpg'))
# print('copy done')

