# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 18:27:46 2022

@author: kirstenl
"""

import os
import cv2
import shutil
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

from joblib import Parallel, delayed
import multiprocessing
NUM_CORES = multiprocessing.cpu_count()

#%%

dataset = 'Fluo-N2DL-HeLa'
lineage = '02'
augment = True

#%%

path_images = sorted(glob(f'./frames/{dataset}/{lineage}_GT/SEG/*.tif'))
path_save_ann = f'./frames/{dataset}/{lineage}/annotations/dota_format'
path_res = f'./frames/{dataset}/{lineage}_RES'
tracklets = open(f'./frames/{dataset}/{lineage}_GT/TRA/man_track.txt').read()
tracklets = [list(map(int, track.split(' '))) \
                for track in tracklets.split('\n')[:-1]]
    
#%%

if augment:
    def _resize_image(path):
        image = cv2.imread(path, -1)
        # augment image
        image = cv2.resize(image, (image.shape[1]*4,image.shape[0]*4))
        
        cv2.imwrite(path, image)
        
    with Parallel(n_jobs=NUM_CORES, prefer='threads') as parallel:
        _ = parallel(delayed(_resize_image)(path)\
                     for path in tqdm(glob(f'./frames/{dataset}/{lineage}/images/*.tif')))

# for path in path_images:
#     shutil.copyfile(path, path.replace('TRA\\man_track', 'SEG/man_seg'))
    
#%% iterate over tracklets to find mitoses

mitoses = {} # cell idx : frame position
# for track in tracklets[::-1]:
#     if track[0] in mitoses:
#         mitoses[track[0]] = track[-2]
#     elif track[-1]==0:
#         continue
#     else:
#         mitoses[track[-1]] = -1

#%%
os.makedirs(path_res, exist_ok=True)
os.makedirs(path_save_ann, exist_ok=True)

def _adjust_segmentation(frame, path):
    
    if not path.endswith('.tif'):
        return
    
    img_name = os.path.split(path)[-1]
    file = open(os.path.join(path_save_ann, 't'+img_name[-7:-4]+'.txt'), 'w')
    file.write('imagesource:ISBI\ngsd:0\n')
    
    image = cv2.imread(path, -1)
    # augment image
    if augment:
        image = cv2.resize(image, (image.shape[1]*4,image.shape[0]*4),interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(path, image)
        
        # augment for tracking
        path_tra = path.replace('SEG\\man_seg', 'TRA\\man_track')
        tra = cv2.imread(path_tra, -1)
        tra = cv2.resize(tra, (image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(path_tra, tra)
    
    vals = np.unique(image)[1:]
    
    draw = np.zeros_like(image)
    for sval in vals:
        y,x = np.where(image==sval)
        if len(x)<2 or len(y)<2:
            continue
        
        points = np.array(list(zip(x,y)))
        (cx,cy), (w,h), ang = cv2.minAreaRect(points)
        
        ellipse = ((cx,cy),(w,h),ang)
        
        draw = cv2.ellipse(draw, ellipse, (int(sval),), -1)
        
        box = cv2.boxPoints(ellipse)
        box = np.int0(box).reshape(-1)
        box[::2]  = np.clip(box[::2], 0, image.shape[1])
        box[1::2] = np.clip(box[1::2], 0, image.shape[0])
        
        cell_class = 'normal_cell'
        if (sval in mitoses) and (frame == mitoses[sval]):
            cell_class = 'mitoses'
        
        file.write(' '.join(map(lambda x:str(x), box))+f' {cell_class} 0\n')
    
    cv2.imwrite(os.path.join(path_res, 'mask'+img_name[-7:]), draw)
    file.close()

with Parallel(n_jobs=NUM_CORES, prefer='threads') as parallel:
    _ = parallel(delayed(_adjust_segmentation)(frame, path) \
                 for frame,path in tqdm(enumerate(path_images), total=len(path_images)))
