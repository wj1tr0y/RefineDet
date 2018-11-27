import argparse
from collections import OrderedDict
import json
import os
from pprint import pprint
import sys
import cv2
sys.path.append(os.path.dirname(sys.path[0]))


HOMEDIR = os.path.expanduser("~")
DATASETDIR = os.path.join(HOMEDIR, 'data/neg/')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "convert txt annotation to MSCOCO-style json annotation file.")
    parser.add_argument("--out-dir",
            help = "The output directory where we store the annotation per image.")
    parser.add_argument("--imageset-dir",
        help = "The output directory where we store the annotation per image.")
    args = parser.parse_args()

    out_dir = os.path.join(DATASETDIR, args.out_dir)
    if out_dir:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    imageset_dir = os.path.join(DATASETDIR, args.imageset_dir)
    if not os.path.exists(imageset_dir):
        print("{} does not exist!".format(imageset_dir))
        sys.exit()

    ann_filenames = os.listdir(imageset_dir)
    total = len(ann_filenames)
    for count, ann_file in enumerate(ann_filenames):
        with open(os.path.join(out_dir, ann_file[:-4] + '.json'), 'w') as new_ann:
            print('Processing {}/{}: '.format(count + 1, total) + os.path.join(imageset_dir, ann_file))
            img = cv2.imread(os.path.join(imageset_dir, ann_file))
            height, width, _ = img.shape
            json_format = {"annotation": [], 
                            "image": {
                                "file_name": ann_file,
                                "height": height,
                                "width": width}}
            del img
            # convert annotations and filter out irregular bbox annotations
            new_ann.writelines(json.dumps(json_format, sort_keys=True, indent=2, ensure_ascii=False))
    
    img_names = [x[:-4] for x in os.listdir(imageset_dir)]
    img_names.sort()
    with open('zhili.txt', 'w') as f:
        f.write("\n".join(img_names))
