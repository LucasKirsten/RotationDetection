# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:51:02 2022

@author: kirstenl
"""

import motmetrics as mm

from .classes import Frame

def evaluate(true_tracklets, pred_tracklets, num_frames):
    
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
        c_pred = pred.get_centers()
        c_true = true.get_centers()
        
        idx_pred = pred.get_idxs()
        idx_true = true.get_idxs()
        
        dists = mm.distances.norm2squared_matrix(c_true, c_pred)
        
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