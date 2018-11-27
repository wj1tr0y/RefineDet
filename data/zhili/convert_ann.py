import argparse
from collections import OrderedDict
import json
import os
from pprint import pprint
import sys
import cv2
sys.path.append(os.path.dirname(sys.path[0]))


HOMEDIR = os.path.expanduser("~")
DATASETDIR = os.path.join(HOMEDIR, 'data/zhili/')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "convert txt annotation to MSCOCO-style json annotation file.")
    parser.add_argument("--anno-dir",
            help = "The file which contains all the annotations for a dataset in txt format.")
    parser.add_argument("--out-dir",
            help = "The output directory where we store the annotation per image.")
    parser.add_argument("--imageset-dir",
        help = "The output directory where we store the annotation per image.")
    args = parser.parse_args()

    annofile_dir = os.path.join(DATASETDIR, args.anno_dir)
    if not os.path.exists(annofile_dir):
        print("{} does not exist!".format(annofile_dir))
        sys.exit()

    out_dir = os.path.join(DATASETDIR, args.out_dir)
    if out_dir:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    imageset_dir = os.path.join(DATASETDIR, args.imageset_dir)
    if not os.path.exists(imageset_dir):
        print("{} does not exist!".format(imageset_dir))
        sys.exit()

    ann_filenames = os.listdir(annofile_dir)
    total = len(ann_filenames)
    for count, ann_file in enumerate(ann_filenames):
        with open(os.path.join(out_dir, ann_file[:-4] + '.json'), 'w') as new_ann:
            print('Processing {}/{}: '.format(count,total) + os.path.join(imageset_dir, ann_file[:-4] + ".jpg"))
            img = cv2.imread(os.path.join(imageset_dir, ann_file[:-4] + ".jpg"))
            height, width, _ = img.shape
            json_format = {"annotation": [], 
                            "image": {
                                "file_name": ann_file[:-4] + ".jpg",
                                "height": height,
                                "width": width}}
            del img
            # convert annotations and filter out irregular bbox annotations
            with open(os.path.join(annofile_dir, ann_file), 'r') as f:
                anno = f.readlines()[1:]
                if anno:
                    for anno_perline in anno:
                        anno_perline = anno_perline.split(' ')
                        if anno_perline[0] == 'person':
                            x = anno_perline[1] if anno_perline[1] > 0 else 0
                            y = anno_perline[2] if anno_perline[2] > 0 else 0
                            w = anno_perline[3] if anno_perline[3] > 0 else 0
                            h = anno_perline[4] if anno_perline[4] > 0 else 0
                            x, y, w, h = list(map(lambda x:int(x), [x, y, w, h]))
                            if w == 0 or h == 0:
                                print('Found irregular bbox annotation. This image has been skipped.')
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
    
    img_names = [x[:-4]+'.jpg' for x in os.listdir(annofile_dir)]
    img_names.sort()
    with open('zhili.txt', 'w') as f:
        f.write("\n".join(img_names))
