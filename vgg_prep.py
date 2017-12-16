import os
import numpy as np
import sys

def get_label(path):
    x = path.split('.')[0].split('-')[0]
    return x

region = sys.argv[1]
SL_PATH = "/Users/aarya/Atlas/tracks/"+str(region)+"/slines"
labels = [get_label(f) for f in os.listdir(SL_PATH) if not f.startswith('.')]

paths = [os.path.join(SL_PATH, f) for f in os.listdir(SL_PATH) if not f.startswith('.')]
foo = np.load(paths[0])

shape = (len(labels), foo.shape[0], foo.shape[1])
region_slines = np.zeros(shape)

idx = 0
for path in paths:
    foo = np.load(path)
    region_slines[idx] = foo
    idx += 1

save_path = SL_PATH+"/all_slines/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_path1 = save_path+str(region)+"-slines"
save_path2 = save_path+str(region)+"-labels"
np.save(save_path1, region_slines)
np.save(save_path2, np.asarray(labels))


