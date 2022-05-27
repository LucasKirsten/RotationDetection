# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 23:37:27 2022

@author: kirstenl
"""

import os
import cv2
import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm
from glob import glob
import multiprocessing
from joblib import Parallel, delayed
NUM_CORES = multiprocessing.cpu_count()

from .configs import *
from .classes import *

#%% functions to read detections

def _read(path_dets, frame_imgs, threshold, mit):
    # auxiliar function to read the detections
    
    detections = pd.read_csv(path_dets, header=None, sep=' ')
    detections = detections.sort_values(by=0)
    detections = detections.loc[detections.iloc[:,0].isin(frame_imgs)]
    detections = detections.loc[detections.iloc[:,1]>threshold]
    detections = [Detection(*det,mit=mit) for _,det in detections.iterrows()]
    
    return detections

def read_detections(path_normals, path_mitoses, frame_imgs):
    
    # open normal detections
    print('Reading normal detections...')
    detections = _read(path_normals, frame_imgs, NORMAL_SCORE_TH, mit=0)

    # open mitoses detections
    print('Reading mitoses detections...')
    detections.extend(_read(path_mitoses, frame_imgs, MIT_SCORE_TH, mit=1))

    # sort detections by name
    return detections #sorted(detections, key=lambda x:x.frame)
    
#%% functions to read labels

@njit(parallel=True)
def _is_continuos(frames_id):
    # verify if the frames detections are continuos
    for i in range(len(frames_id)-1):
        if frames_id[i]+1!=frames_id[i+1]:
            return 0
    return 1

def read_annotations_csv(path_gt):
    df = pd.read_csv(path_gt)
    
    assert ('cx' in df) and ('cy' in df) and ('frame' in df), \
           'The annotation should contain at least columns frame,cx and cy!'
    
    # fill missing columns
    if not ('w' in df):
        df['w'] = [0]*len(df['cx'])
    if not ('h' in df):
        df['h'] = [0]*len(df['cx'])
    
    # group by cell track and remove non continous ones
    groups = [x for _, x in df.groupby(['cell'])]
    groups = [g for g in groups if _is_continuos(np.array(g)[:,1])]
    
    def _get_tracklets(i,group):
        # auxiliar function to define tracklets from detections
        detections = [Detection(r['frame'],1,r['cx'],r['cy'],r['w'],r['h'],\
                                idx=i,convert=False) \
                      for _,r in group.iterrows()]
        return Tracklet(detections, start=detections[0].frame-1)
        
    pbar = tqdm(enumerate(groups), total=len(groups))
    pbar.set_description('Reading annotations')
    with Parallel(n_jobs=NUM_CORES, prefer='threads') as parallel:
        tracklets = parallel(delayed(_get_tracklets)(i,group) \
                               for i,group in pbar)
        
    return tracklets

def read_annotations_tif(path_gt:str, ext:str='.tif'):
    
    path_gt = glob(os.path.join(path_gt, '*'+ext))
    path_gt = sorted(path_gt)
    
    tracklets = {}
    
    pbar = tqdm(enumerate(path_gt), total=len(path_gt))
    def _get_tracklets(i,path):
        nonlocal tracklets
        
        frame = os.path.split(path)[-1].split('.')[-0]
        seg = cv2.imread(path, -1)
        
        for val in np.unique(seg):
            if val==0:
                continue
            
            y,x = np.where(seg==val)
            points = np.array(list(zip(x,y)))
            (cx,cy), (w,h), ang = cv2.minAreaRect(points)
            
            detection = Detection(frame,1,cx,cy,w,h,ang)           
            
            if val not in tracklets:
                tracklets[val] = Tracklet(detection, i)
            else:
                tracklets[val].append(detection)
                
    pbar.set_description('Reading annotations')
    with Parallel(n_jobs=NUM_CORES, prefer='threads') as parallel:
        _ = parallel(delayed(_get_tracklets)(i,path) \
                               for i,path in pbar)
                
    return list(tracklets.values())







