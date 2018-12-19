import threading
import json
import os

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

if __name__ == "__main__":
    img_dir = '/home/wangjilong/data/zhili_coco_posneg/ImageSet'
    ann_dir = '/home/wangjilong/data/zhili_coco_posneg/Annotations'
    det_dir = './detout'
    det_name = os.listdir(det_dir)
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
        for gt in ann:
            if gt['count'] > 1:
                multi_bbox += 1
        for res in result:
            if res['count'] == 0:
                mismatch_bbox += 1
        
        if mismatch_bbox > 2 or multi_bbox > 5:
            hard_name.append(det)
        
        print(mismatch_bbox, multi_bbox)
    print(hard_name)
