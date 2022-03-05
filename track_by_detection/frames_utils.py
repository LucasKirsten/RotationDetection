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

from configs import *
from func_utils import *
from classes import *

#%% get the frames from detections

def get_frames(detections):
    
    # split detections into frames
    frames, frame_detections = [],Frame()
    def _split_detections(i):
        nonlocal frames, frame_detections
        if detections[i].frame==detections[i+1].frame:
            frame_detections.append(detections[i])
        else:
            frames.append(frame_detections)
            frame_detections = Frame()
    
    pbar = tqdm(range(len(detections)-1))
    pbar.set_description('Loading frames: ')
    with Parallel(n_jobs=NUM_CORES, prefer="threads") as parallel:
        _ = parallel(delayed(_split_detections)(i) for i in pbar)
            
    # add a indexing value for the first frame detections
    for n,fr in enumerate(frames[0]):
        fr.idx = n
        
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
    # initialize trackelts
    tracklets = [Tracklet(det,0) for det in frames[0]]
    
    ids = set(range(len(frames[0]))) # set of indexes
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
                frames[i+1][col].idx = frames[i][row].idx
        
        # add a new id for new detections
        for k in range(len(frm1)):
            fr_id = float(frames[i+1][k].idx)
            if fr_id==-1:
                frames[i+1][k].idx = max(ids) + 1
                ids.add(float(frames[i+1][k].idx))
                
        # add objects to tracklets
        for det in frames[i+1]:
            det_id = int(float(det.idx))
            if det_id>=len(tracklets):
                tracklets.append(Tracklet(det, i+1))
            else:
                tracklets[det_id].append(det)
    
    # filter tracklets based on the number of detections and score
    if TRACK_SCORE_TH>0 or TRACK_SIZE_TH>0:
        tracklets = [tr for tr in tracklets \
                     if not (len(tr)<=TRACK_SIZE_TH and tr.score()<TRACK_SCORE_TH)]
    
    # sort tracklets based on the first frame they appear
    tracklets = sorted(tracklets, key=lambda x:x.start)
                
    return tracklets

#%% adjust the tracklets to the frames

def _adjust_tracklets(tracklets, hyphotesis):
    
    pbar = tqdm(hyphotesis)
    pbar.set_description('Iterating over hyphoteses: ')
    keep_tracks = [] #tracklets to be kept
    for hyp in pbar:
        
        # verify the hyphotesis name and its values
        mode,idxs = hyp.split('_')
        if ('init' in mode) or ('term' in mode):
            keep_tracks.append(int(idxs))
        
        if 'transl' in mode:
            idx1,idx2 = idxs.split(',')
            tracklets[int(idx1)].join(tracklets[int(idx2)])
            keep_tracks.append(int(idx2))
            
        elif 'fp' in mode:
            keep_tracks.append(int(idxs))
            
    tracklets = [tracklets[i] for i,k in enumerate(set(keep_tracks)) if k]
            
    return tracklets











