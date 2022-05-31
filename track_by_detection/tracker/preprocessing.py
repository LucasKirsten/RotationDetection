# -*- coding: utf-8 -*-
"""
Preprocessing functions.

@author: Lucas N. Kirsten (lnkirsten@inf.ufrgs.br)
"""

import numpy as np
from numba import jit
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed
NUM_CORES = multiprocessing.cpu_count()

from .configs import *
from .classes import *
from .func_utils import helinger_dist

#%% NMS algorithm using ProbIoU

def _compute_NMS(frames):
    
    for frame in frames:
        if len(frame)<1:
            yield []
        frame_name = frame.name
        
        # iterate over boxes to verify which ones to join
        boxes2join, joined = [],set()
        for i,d1 in enumerate(frame):
            
            # if detection was already joined with other
            if i in joined:
                continue
            
            # add det1 to the list boxes
            boxes2join.append([d1])
            
            # iterate over the other detections
            for j,d2 in enumerate(frame[i+1:]):
                
                # if they belong to different classes, continue
                if d1.mit!=d2.mit:
                    continue
                
                # compute helinger distance between boxes
                hd = helinger_dist(d1.cx,d1.cy,d1.a,d1.b,d1.c,\
                                   d2.cx,d2.cy,d2.a,d2.b,d2.c)
                
                if (1-hd)>NMS_TH:
                    # add box to be joined
                    joined.add(j+i+1)
                    boxes2join[-1].append(d2)
        
        # iterate over boxes that have to be joined
        final_boxes = Frame(name=frame_name)
        for boxes in boxes2join:
            
            if len(boxes)==1:
                # add single detection to frame
                final_boxes.append(boxes[0])
                continue
            
            # initialize values
            mean = np.array([[0.],[0.]])
            corr = np.array([[0.,0.],[0.,0.]])
            sum_score, max_score = 0,0
            for d in boxes:
                # compute mean and corr
                m = np.array([[d.cx],[d.cy]])
                mean += d.score * m
                corr += d.score * (np.array([[d.a,d.c],[d.c,d.b]]) + np.matmul(m,m.T))
                sum_score += d.score
                
                # compute score (max score over detections)
                max_score = max(max_score, d.score)
            
            # divide arrays by the sum score
            mean /= (sum_score+1e-6)
            corr /= (sum_score+1e-6)
            
            # suvtract the correlation by the final mean value
            corr -= np.matmul(mean,mean.T)
            
            # get the calculated values to the final box
            cx, cy = mean[0][0], mean[1][0]
            a, b, c = corr[0][0], corr[0][1], corr[1][1]
            
            # add detection to frame
            final_boxes.append(Detection(frame_name,max_score,cx,cy,\
                                         a=a,b=b,c=c,mit=d.mit))
        
        yield final_boxes

def apply_NMS(frames:list) -> list:
    '''
    Apply Gaussian NMS algorithm to a list of frames.

    Parameters
    ----------
    frames : list
        List of frames with detections.

    Returns
    -------
    list
        NMS frames detections.

    '''
    
    pbar = tqdm(_compute_NMS(frames), total=len(frames))
    pbar.set_description('Applying NMS to frames')
    with Parallel(n_jobs=NUM_CORES, prefer='threads') as parallel:
        nms_frames = parallel(delayed(lambda x:x)(x) for x in pbar)
        
    return nms_frames

