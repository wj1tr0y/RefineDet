import threading
import json
import os
import threading
import shutil

def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)

def find_hard(det_name, count):
    hard_name = []
    for det in det_name:
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
            
        for gt in ann:
            bbox = gt['bbox']
            rect = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
            for res in result:
                bbox2 = res['bbox']
                rect2 = [bbox2[0], bbox2[1], bbox2[0]+bbox2[2], bbox2[1]+bbox2[3]]
                if compute_iou(rect, rect2) > 0.5:
                    gt['count'] += 1
                    res['count'] += 1

        multi_bbox = 0
        mismatch_bbox = 0
        lost_bbox = 0
        for gt in ann:
            if gt['count'] > 1:
                multi_bbox += 1
            if gt['count'] == 0:
                lost_bbox +=1
        for res in result:
            if res['count'] == 0:
                mismatch_bbox += 1

        if mismatch_bbox > 12 or multi_bbox > 12 or lost_bbox > 12:
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
    
    with open('hardexample.txt', 'w') as hd:
        for i in hd.readlines():
            shutil.copyfile(os.path.join(ann_dir, i[:-10] + '.json'), os.path.join(hard_dir, 'Annotations/'))
            shutil.copyfile(os.path.join(img_dir, i[:-10] + '.jpg'), os.path.join(hard_dir, 'ImageSet/'))
    print('copy done')

