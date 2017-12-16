
# coding: utf-8

# In[1]:

import numpy as np
import dipy.tracking.utils as dtu
import pandas as pd
import nibabel as nib
import sys


# In[3]:
region = sys.argv[1]

DATA_PATH = "/Users/aarya/Atlas/tracks/"+str(region)

import os
import os.path as op
from glob import glob


# In[19]:

labels = [f for f in os.listdir(DATA_PATH) if not f.startswith('.')]


# In[5]:

bundle_fnames = glob(op.join(DATA_PATH, '*.trk'))


# In[7]:

t1_shape = (256, 256, 150)


# In[8]:
sl_sum = 0
sls = []
print('Bundles and number of streamlines: \n')
for b_idx, bundle in enumerate(bundle_fnames):
    tgram = nib.streamlines.load(op.join(DATA_PATH, bundle))
    print(bundle, len(tgram.streamlines))
    sls.append(len(tgram.streamlines))
    sl_sum += len(tgram.streamlines)

minsls = min(sls)
print('\n\n')
# In[10]:

def get_bname(f):
    return f.split('/')[-1].split('.')[0]

# In[14]:

ii = 0
newpath = DATA_PATH+'/slines'
if not os.path.exists(newpath):
    os.makedirs(newpath)
for b_idx, bundle in enumerate(bundle_fnames):
    tgram = nib.streamlines.load(op.join(DATA_PATH, bundle))
    print(bundle, len(tgram.streamlines))
    for sl_idx, sl in enumerate(list(dtu.move_streamlines(tgram.streamlines, np.linalg.inv(tgram.affine)))):
            if sl_idx > minsls:
                break
            bname = get_bname(bundle)
            savepath = newpath+"/"+bname+'-'+str(sl_idx)
            if not np.mod(sl_idx, 100):
                print("Streamline {0} at index {1} of Bundle {2} saved at path {3}".format(sl_idx, ii, bname, savepath))
            vol = np.zeros(t1_shape + (1,), dtype=bool)
            sl = np.round(sl).astype(int).T
            vol[sl[0], sl[1], sl[2]] = 1
            a1 = np.max(vol, 0).squeeze()
            for x in range(1,3):
                a1 = np.concatenate((a1, np.max(vol, x).squeeze()), axis=1)
            np.save(savepath, a1)
            ii += 1


# In[15]:

def get_label(path):
    x = path.split('/')[-1].split('_')[:2]
    return '_'.join(x)


# In[16]:

def get_dataframe(directory):
    paths = [os.path.join(directory, f) for f in os.listdir(directory) if not f.startswith('.')]
    labels = [get_label(path) for path in paths if not path.startswith('.')]
    return pd.DataFrame({'paths': paths, 'labels': labels})




