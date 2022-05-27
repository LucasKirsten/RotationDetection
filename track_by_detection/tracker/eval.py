# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:51:02 2022

@author: kirstenl
"""

import numpy as np
import motmetrics as mm
from numba import njit

from .classes import Frame
from .func_utils import helinger_dist, intersection_over_union

@njit(parallel=True, cache=True)
def _build_hd_matrix(frm0, frm1):
    costs = np.zeros((len(frm0), len(frm1)))
    for j in range(costs.shape[0]):
        for k in range(costs.shape[1]):
            cx0,cy0,w0,h0,ang0,a0,b0,c0 = frm0[j][1:-1] # remove score and mit
            cx1,cy1,w1,h1,ang1,a1,b1,c1 = frm1[k][1:-1]
            hd = helinger_dist(cx0,cy0,a0,b0,c0, \
                               cx1,cy1,a1,b1,c1)
            iou = intersection_over_union(cx0,cy0,w0,h0,
                                          cx1,cy1,w1,h1)
            costs[j,k] = hd if iou>0 else np.nan
    return costs

def evaluate(true_tracklets, pred_tracklets, num_frames, dist_method:str='hd'):
    
    assert dist_method in ('center', 'iou', 'hd'), \
        'Distance method should be either center, iou or hd!'
    
    # convert tracklets from predictions to frames
    pred_frames = [Frame() for _ in range(num_frames+1)]
    for track in pred_tracklets:
        fr_idx = track.start
        for d_idx,det in enumerate(track):
            pred_frames[fr_idx+d_idx].append(det)
    
    # convert tracklets from annotations to frames
    true_frames = [Frame() for _ in range(num_frames+1)]
    for track in true_tracklets:
        fr_idx = track.start
        for d_idx,det in enumerate(track):
            true_frames[fr_idx+d_idx].append(det)
    
    # define accumulator
    acc = mm.MOTAccumulator(auto_id=True)
    
    # add all detections to accumulator
    for pred,true in zip(pred_frames, true_frames):
        
        idx_pred = pred.get_idxs()
        idx_true = true.get_idxs()
        
        # compute distance between predicted and true detections
        if dist_method=='center':
            c_pred = pred.get_centers()
            c_true = true.get_centers()
            dists = mm.distances.norm2squared_matrix(c_true, c_pred)
            
        elif dist_method=='iou':
            c_pred = pred.get_iou_values()[:-1]
            c_true = true.get_iou_values()[:-1]
            dists = mm.distances.iou_matrix(c_true, c_pred)
            
        elif dist_method=='hd':
            c_pred = pred.get_values()
            c_true = true.get_values()
            dists = _build_hd_matrix(c_true, c_pred)
        
        acc.update(idx_true, idx_pred, dists)
        
    # make evaluation
    mh = mm.metrics.create()
    summary = mh.compute_many(
        [acc, acc.events.loc[0:1]],
        metrics=mm.metrics.motchallenge_metrics,
        names=['full', 'part'], generate_overall=True)
    
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    
    return strsummary