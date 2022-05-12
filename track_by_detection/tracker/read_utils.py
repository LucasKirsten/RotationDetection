# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 23:37:27 2022

@author: kirstenl
"""

import numpy as np
import pandas as pd
from numba import njit
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed
NUM_CORES = multiprocessing.cpu_count()

from .configs import *
from .classes import *

#%% functions to read detections

def _read(path_dets, frame_imgs, threshold, mit):
    
    detections = pd.read_csv(path_dets, header=None, sep=' ')
    detections = detections.sort_values(by=0)
    detections = detections.loc[detections.iloc[:,0].isin(frame_imgs)]
    detections = detections.loc[detections.iloc[:,1]>threshold]
    detections = [Detection(*det,mit) for _,det in detections.iterrows()]
    
    return detections

def read_detections(path_normals, path_mitoses, frame_imgs):
    
    # open normal detections
    print('Reading normal detections...')
    detections = _read(path_normals, frame_imgs, NORMAL_SCORE_TH, 0)

    # open mitoses detections
    print('Reading mitoses detections...')
    detections.extend(_read(path_mitoses, frame_imgs, MIT_SCORE_TH, 1))

    # sort detections by name
    return detections #sorted(detections, key=lambda x:x.frame)
    
#%% functions to read labels

@njit
def _is_continuos(frames_id):
    for i in range(len(frames_id)-1):
        if frames_id[i]+1!=frames_id[i+1]:
            return 0
    return 1

def read_annotations(path_gt):
    df = pd.read_csv(path_gt)
    
    # group by cell track and remove non continous ones
    groups = [x for _, x in df.groupby(['cell'])]
    groups = [g for g in groups if _is_continuos(np.array(g)[:,1])]
    
    def _get_tracklets(i,group):
        detections = [Detection(r['frame'],1,r['cx'],r['cy'],idx=i,convert=False) \
                      for _,r in group.iterrows()]
        return Tracklet(detections, start=detections[0].frame-1)
        
    pbar = tqdm(enumerate(groups), total=len(groups))
    pbar.set_description('Reading annotations')
    with Parallel(n_jobs=NUM_CORES, prefer='threads') as parallel:
        tracklets = parallel(delayed(_get_tracklets)(i,group) for i,group in pbar)
        
    return tracklets