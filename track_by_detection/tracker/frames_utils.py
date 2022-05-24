# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 18:33:00 2022

@author: kirstenl
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import product
from tqdm import tqdm
from numba import njit
from joblib import Parallel, delayed
import multiprocessing
NUM_CORES = multiprocessing.cpu_count()

from .configs import *
from .func_utils import *
from .classes import *

#%% get the frames from detections

def get_frames(detections, frame_names):
    
    # merge detections into frames
    frames = {name:Frame(name=name) for name in frame_names}
    pbar = tqdm(detections)
    pbar.set_description('Loading frames: ')
    for det in pbar:
        frames[det.frame].append(det)
        
    # sort the frames by name
    frames = [frames[name] for name in sorted(list(frames.keys()))]
        
    return frames

#%% iterate over frames to apply hungarian algorithm and get tracklets

@njit(parallel=True, cache=True)
def _build_costs(frm0, frm1):
    costs = np.zeros((len(frm0), len(frm1)))
    for j in range(costs.shape[0]):
        for k in range(costs.shape[1]):
            cx0,cy0,w0,h0,ang0,a0,b0,c0 = frm0[j][1:-1] # remove score and mit
            cx1,cy1,w1,h1,ang1,a1,b1,c1 = frm1[k][1:-1]
            hd = helinger_dist(cx0,cy0,a0,b0,c0, \
                               cx1,cy1,a1,b1,c1)
            iou = intersection_over_union(cx0,cy0,w0,h0,
                                          cx1,cy1,w1,h1)
            costs[j,k] = hd if iou>0 else 1
    return costs

def get_tracklets(frames):
    # add a indexing value for the first frame detections
    for n,det in enumerate(frames[0]):
        det.idx = n
    
    # add a -1 index value for all other frames detections
    for fr in frames[1:]:
        for det in fr:
            det.idx = -1
    
    # initialize trackelts
    tracklets = [Tracklet(det,0) for det in frames[0]]
    
    ids = set(range(len(tracklets))) # set of indexes
    pbar = tqdm(range(len(frames)-1))
    pbar.set_description('Getting tracklets: ')
    for i in pbar:
        
        # take consecutive frames
        frm0 = frames[i]
        frm1 = frames[i+1]
        
        # hungarian algorithm
        costs = _build_costs(frm0.get_values(), frm1.get_values())
        row_ind, col_ind = linear_sum_assignment(costs)
        
        # map the detected objects to its pairs
        for row,col in zip(row_ind, col_ind):
            if costs[row][col]<1:
                frm1[col].idx = int(frm0[row].idx)
                tracklets[frm1[col].idx].append(frm1[col])
            else:
                frm1[col].idx = int(max(ids) + 1)
                ids.add(float(max(ids) + 1))
                tracklets.append(Tracklet([frm1[col]], i+1))
        
        # verify if any detection remained with idx==-1
        for det in frm1:
            if int(det.idx) == -1:
                det.idx = int(max(ids) + 1)
                ids.add(float(max(ids) + 1))
                tracklets.append(Tracklet([det], i+1))
    
    # filter tracklets based on the number of detections and score
    if TRACK_SCORE_TH>0 or TRACK_SIZE_TH>0:
        tracklets = [tr for tr in tracklets \
                     if (len(tr)>TRACK_SIZE_TH and tr.score()>TRACK_SCORE_TH)]
    
    # sort tracklets based on the first frame they appear
    tracklets = sorted(tracklets, key=lambda x:x.start)
                
    return tracklets











