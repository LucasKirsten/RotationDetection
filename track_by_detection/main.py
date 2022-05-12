# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 22:11:31 2022

@author: kirstenl
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

from tracker import *

#%% read all detections
init = time()

path_imgs = f'./frames/{DATASET}/frames'
path_dets = f'./frames/{DATASET}/{DETECTOR}'
path_gt   = f'./frames/{DATASET}/migration.csv'

# get sorted frames by name
frame_imgs = [file.split('.')[0] for file in os.listdir(path_imgs)]
frame_imgs = sorted(frame_imgs)#[:1001]

# get detections
detections = read_detections(f'{path_dets}/det_normal_cell.txt', \
                             f'{path_dets}/det_mitoses.txt', frame_imgs)
    
annotations = read_annotations(path_gt)

#%% split detections into frames
frames = get_frames(detections)
Nf = len(frames)
del detections

#%% apply NMS on frames detections
frames = apply_NMS(frames)


#%% get trackelts
tracklets = get_tracklets(frames)

DEBUG = 0
if DEBUG:
    ids = len(tracklets)
    colors = np.linspace(10,200,len(tracklets)+1,dtype='uint8')[1:]
    
    frame_images = []
    for frm in frames:
        img_name = frm[0][0]
        img_name = os.path.join(path_imgs, img_name+'.jpg')
        img = cv2.imread(img_name)
        draw = np.copy(img)
        
        for det in frm:
            cx,cy,w,h,a = map(lambda x:float(x), det[2:-2])
            #if float(det[-1])==-1:
            #    continue
            obj_id = det[-1]
            if obj_id.isnumeric():
                obj_id = int(float(obj_id))
                cor = (int(colors[obj_id]), int(colors[int(len(tracklets)-obj_id-1)]), 255)
            else:
                cor = (0,0,0)
            
            box = cv2.boxPoints(((cx,cy),(w,h),a))
            box = np.int0(box)
            
            draw = cv2.drawContours(draw, [box], -1, cor, 2)
            draw = cv2.putText(draw, str(obj_id), (int(cx),int(cy)), \
                               cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 1, cv2.LINE_AA)
        
        frame_images.append(draw)
        
        cv2.imshow('tracklets', draw)
        cv2.waitKey(200)
    cv2.destroyAllWindows()

#%% solve tracklets
final_tracklets = solve_tracklets(tracklets, Nf)

print('Elapsed time: ', time()-init)

#%% evaluate predictions

print(evaluate(annotations, final_tracklets, len(frames)))

#%% draw detections from CNN

frame_imgs = []
pbar = tqdm(frames)
pbar.set_description('Reading frames')
for frm in pbar:
    img_name = frm[0].frame
    img_name = os.path.join(path_imgs, img_name+'.png')
    img = cv2.imread(img_name)[...,::-1]
    draw = np.copy(img)
    
    for det in frm:
        cx,cy,w,h,a,mit = det.cx,det.cy,det.w,det.h,det.a,det.mit
        box = cv2.boxPoints(((cx,cy),(w,h),a))
        box = np.int0(box)
        
        color = (0,0,255) if int(mit)==0 else (0,255,0)
        draw = cv2.drawContours(draw, [box], -1, color, 2)
        
    frame_imgs.append(draw)

#%% draw trackings

total_detections = len(final_tracklets)
colors = np.linspace(10,240,total_detections+1,dtype='uint8')[1:]

pbar = tqdm(enumerate(final_tracklets), total=len(final_tracklets))
pbar.set_description('Drawing tracklets')
for ti,track in pbar:
    
    start = track.start
    det_id = track[0].idx
    
    for di,det in enumerate(track):
        if start+di>=len(frames):
            continue
 
        cx,cy,w,h,a,mit = det.cx,det.cy,det.w,det.h,det.a,det.mit
        box = cv2.boxPoints(((cx,cy),(w,h),a))
        box = np.int0(box)
        
        color = (int(colors[ti]), int(255-colors[ti]), 255)
        #frame_imgs[start+di] = cv2.drawContours(frame_imgs[start+di], [box], -1, color, 2)
        #frame_imgs[start+di] = cv2.putText(frame_imgs[start+di], str(det_id), (int(cx),int(cy)), \
        #                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
        frame_imgs[start+di] = cv2.circle(frame_imgs[start+di], (int(cx),int(cy)), 2, color, 2)

DEBUG = True
if DEBUG:
    for img in frame_imgs:
        cv2.imshow('', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    
print('Elapsed time with drawing: ', time()-init)

#%%
    
h,w,c = frame_imgs[0].shape
out = cv2.VideoWriter(f'./{DATASET}_{DETECTOR}_{len(frames)}.avi',\
                      cv2.VideoWriter_fourcc(*'XVID'), 10.0, (w,h))
for img in frame_imgs:
    out.write(img)
out.release()


'''
TODO:
    - Remove bad frames
    - Solve multiple detections on same cell
    - Solve hyphotesis matrix in batches of N tracklets/frames
    - Add movement feature pixel wise for each detection (X)
    - Fill gaps between joined tracklets
    - Add NN feature for the detections
'''

#%%









