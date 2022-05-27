# -*- coding: utf-8 -*-
"""
Created on Fri May 20 20:52:18 2022

@author: kirstenl
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
NUM_CORES = multiprocessing.cpu_count()

def draw_detections(frames, path_imgs, img_format='.png', plot=True):
    
    pbar = tqdm(frames)
    pbar.set_description('Reading frames')
    def _draw_frame(frm):
        img_name = frm.name
        img_name = os.path.join(path_imgs, img_name+img_format)
        img = cv2.imread(img_name)[...,::-1]
        draw = np.copy(img)
        
        for det in frm:
            cx,cy,w,h,a,mit = det.cx,det.cy,det.w,det.h,det.a,det.mit
            box = cv2.boxPoints(((cx,cy),(w,h),a))
            box = np.int0(box)
            
            color = (0,0,255) if int(mit)==0 else (0,255,0)
            draw = cv2.drawContours(draw, [box], -1, color, 2)
            
        return draw
        
    with Parallel(n_jobs=NUM_CORES, prefer='threads') as parallel:
        frame_imgs = parallel(delayed(_draw_frame)(frm) for frm in pbar)
        
    if plot:
        for img in frame_imgs:
            cv2.imshow('', img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return frame_imgs

def draw_tracklets(tracklets, frames, path_imgs, img_format='.png', plot=True):
    
    frame_imgs = draw_detections(frames, path_imgs, img_format, plot=False)
    
    total_detections = len(tracklets)
    colors = np.linspace(10,240,total_detections+1,dtype='uint8')[1:]
    np.random.shuffle(colors)

    pbar = tqdm(enumerate(tracklets), total=len(tracklets))
    pbar.set_description('Drawing tracklets')
    def _draw_track(ti, track):
        nonlocal frame_imgs, colors
        
        start = track.start
        det_id = track[0].idx
        
        for di,det in enumerate(track):
            if start+di>=len(frames):
                continue
     
            cx,cy,w,h,a = det.cx,det.cy,det.w,det.h,det.a
            box = cv2.boxPoints(((cx,cy),(w,h),a))
            box = np.int0(box)
            
            color = (int(colors[ti]), int(255-colors[ti]), 255)
            frame_imgs[start+di] = cv2.drawContours(frame_imgs[start+di], [box], -1, color, 2)
            frame_imgs[start+di] = cv2.putText(frame_imgs[start+di], str(det_id), (int(cx),int(cy)), \
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
            #frame_imgs[start+di] = cv2.circle(frame_imgs[start+di], (int(cx),int(cy)), 2, color, 2)
    
    with Parallel(n_jobs=NUM_CORES, prefer='threads') as parallel:
        _ = parallel(delayed(_draw_track)(ti,track) for ti,track in pbar)

    if plot:
        for img in frame_imgs:
            cv2.imshow('', img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return frame_imgs