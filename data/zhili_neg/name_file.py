import os
HOMEDIR = os.path.expanduser("~")
DATASETDIR = os.path.join(HOMEDIR, 'data/zhili_neg/')

im_names = os.listdir(os.path.join(DATASETDIR, 'Annotations'))
im_names = [x for x in im_names if 'json' in x]
im_names = [x[:-5] for x in im_names]
im_names.sort()

with open(os.path.join(DATASETDIR,'ImageSet/zhili_neg.txt'), 'w') as f:
    f.write("\n".join(im_names))
