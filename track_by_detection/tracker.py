# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 22:11:31 2022

@author: kirstenl
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from frames_utils import *
from bise_etal import *
from classes import *

DEBUG = False

dataset  = 'Dados CytoSMART' # Dados CytoSMART OR osfstorage_dataset
detector = 'r3det' # r2cnn OR r3det

#%% read all detections

path_imgs = f'./frames/{dataset}/frames'
path_dets = f'./frames/{dataset}/{detector}'


# get only some frames
frame_imgs = [file.split('.')[0] for file in os.listdir(path_imgs)]
frame_imgs = sorted(frame_imgs)[:101]

# open normal detections
normal_detections = read_detections(f'{path_dets}/det_normal_cell.txt', frame_imgs)

# open mitoses detections
mitoses_detections = read_detections(f'{path_dets}/det_mitoses.txt', frame_imgs)

# merge detections
detections = [Detection(*det,0) for det in normal_detections]
detections.extend([Detection(*det,1) for det in mitoses_detections])

# sort detections by name
detections = sorted(detections, key=lambda x:x.frame)

#%% split detections into frames
frames = get_frames(detections)

if DEBUG:
    for frm in frames[:10]:
        img_name = frm[0][0]
        img_name = os.path.join(path_imgs, img_name+'.jpg')
        img = cv2.imread(img_name)[...,::-1]
        draw = np.copy(img)
        
        for det in frm:
            score,cx,cy,w,h,a,mit = map(lambda x:float(x), det[1:-1])
            box = cv2.boxPoints(((cx,cy),(w,h),a))
            box = np.int0(box)
            
            color = (0,0,255) if int(mit)==0 else (0,255,0)
            draw = cv2.drawContours(draw, [box], -1, color, 2)
        
        cv2.imshow('frames and detections',draw)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

#%% get trackelts
tracklets = get_tracklets(frames)

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
final_tracklets = solve_tracklets(tracklets, frames)

#%% draw detections from CNN

frame_imgs = []
for frm in frames:
    img_name = frm[0].frame
    img_name = os.path.join(path_imgs, img_name+'.jpg')
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

total_detections = len(tracklets)
colors = np.linspace(10,240,total_detections+1,dtype='uint8')[1:]

for ti,track in enumerate(tracklets):
    
    start = track.start
    det_id = track[0].idx
    
    for di,det in enumerate(track):
        if start+di>=len(frames):
            continue
 
        cx,cy,w,h,a,mit = det.cx,det.cy,det.w,det.h,det.a,det.mit
        box = cv2.boxPoints(((cx,cy),(w,h),a))
        box = np.int0(box)
        
        color = (int(colors[ti]), int(255-colors[ti]), 255)
        frame_imgs[start+di] = cv2.drawContours(frame_imgs[start+di], [box], -1, color, 2)
        frame_imgs[start+di] = cv2.putText(frame_imgs[start+di], str(det_id), (int(cx),int(cy)), \
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)

if DEBUG:
    for img in frame_imgs:
        cv2.imshow('', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

#%%
    
h,w,c = frame_imgs[0].shape
out = cv2.VideoWriter(f'./{dataset}_{detector}.avi',cv2.VideoWriter_fourcc(*'XVID'), 10.0, (w,h))
for img in frame_imgs:
    out.write(img)
out.release()


'''
TODO:
    - Test R3Det
    - Remove bad frames
    - Solve multiple detections on same cell
    - Solve hyphotesis matrix in batches of N tracklets/frames
    - Add movement feature pixel wise for each detection (X)
    - Fill gaps between joined tracklets
    - Add NN feature for the detections
'''

