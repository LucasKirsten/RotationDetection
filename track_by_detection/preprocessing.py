# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 14:23:50 2022

@author: kirstenl
"""

import numpy as np
from numba import jit
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed
NUM_CORES = multiprocessing.cpu_count()

from configs import *
from classes import *
from func_utils import helinger_dist

#%% NMS algorithm using ProbIoU

def _compute_NMS(frames):
    
    def __nms(i):
        frame = frames[i]
        frame = Frame(sorted(frame, key=lambda x:x.score, reverse=True))
        frame = frame.get_values()
        
        k = 0
        boxes, joined, scores, mits, num = [],[],[],[],[]
        for i,det1 in enumerate(frame):
            
            if i in joined:
                continue
            
            cx1,cy1,_,_,_,a1,b1,c1 = det1[1:-1]
            boxes.append(det1[1:-1]*det1[0])
            scores.append(det1[0])
            mits.append(det1[-1])
            num.append(1)
            
            for j,det2 in enumerate(frame[i+1:]):
                
                cx2,cy2,_,_,_,a2,b2,c2 = det2[1:-1]
                hd = helinger_dist(cx1,cy1,a1,b1,c1, cx2,cy2,a2,b2,c2)
                
                if (1-hd)>NMS_TH:
                    joined.append(j+i+1)
                    boxes[k]  += det2[1:-1]*det2[0]
                    scores[k] += det2[0]
                    mits[k] += det2[-1]
                    num[k]  += 1
                    
            k += 1
        
        scores = np.array(scores)[...,np.newaxis]
        boxes  = np.array(boxes)/scores
        scores = scores[...,0]/np.array(num)
        mits = np.array([m/n>0.5 for n,m in zip(num,mits)])
        
        yield np.concatenate([scores[...,np.newaxis],boxes,mits[...,np.newaxis]], axis=-1)
        
    return __nms

def apply_NMS(frames):
    if not isinstance(frames, list):
        frames = [frames]
    
    pbar = tqdm(enumerate(frames), total=len(frames))
    pbar.set_description('Applying NMS to frames')
    
    generator = _compute_NMS(frames)
    nms_frames = []
    def __compute_boxes(i,frame):
        boxes = next(generator(i))
        frame_name = frame[0].frame
        nms_frames.append(Frame([Detection(frame_name, *b) for b in boxes]))
        
    with Parallel(n_jobs=NUM_CORES, prefer='threads') as parallel:
        _ = parallel(delayed(__compute_boxes)(i,frame) for i,frame in pbar)
        
    return nms_frames

