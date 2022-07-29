# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 00:28:03 2022

@author: kirstenl
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

path_seg = r'frames\ISBI\Fluo-N2DL-HeLa\01_GT\SEG'
path_tra = r'frames\ISBI\Fluo-N2DL-HeLa\02_GT\TRA'

#%%

paths_seg = glob(path_seg+'/*.tif')
paths_tra = glob(path_tra+'/*.tif')

#%%

for path in glob(r'frames\Fluo-N2DL-HeLa\02_RES\*.tif'):
    os.rename(path, path.replace('man_track', 'mask'))

#%%


for ps, pt in zip(paths_seg, paths_tra):
    
    seg = cv2.imread(ps, -1)
    tra = cv2.imread(pt, -1)

    if len(np.unique(seg))!=len(np.unique(tra)):
        print(ps, pt)

#%%

ps = r'frames\ISBI\Fluo-N2DL-HeLa\01_GT\SEG\man_seg066.tif'
pt = r'frames\ISBI\Fluo-N2DL-HeLa\01_GT\TRA\man_track066.tif'

seg = cv2.imread(ps, -1)
tra = cv2.imread(pt, -1)

img = cv2.imread(r'frames\ISBI\Fluo-N2DL-HeLa\01\t066.tif', -1)




