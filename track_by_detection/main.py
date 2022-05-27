# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 22:11:31 2022

@author: kirstenl
"""

import os
import cv2
from time import time

from tracker import *

#%% read all detections
init = time()

path_imgs = f'./frames/{DATASET}/frames'
path_dets = f'./frames/{DATASET}/{DETECTOR}'
path_gt   = f'./frames/{DATASET}/migration.csv'

# get sorted frames by name
frame_imgs = [file.split('.')[0] for file in os.listdir(path_imgs)]
frame_imgs = sorted(frame_imgs)

# get detections
detections = read_detections(f'{path_dets}/det_normal_cell.txt', \
                             f'{path_dets}/det_mitoses.txt', frame_imgs)
    

annotations = read_annotations(path_gt) if os.path.exists(path_gt) else None

#%% split detections into frames
frames = get_frames(detections, frame_imgs)
Nf = len(frames)
#del detections

#%% apply NMS on frames detections
nms_frames = apply_NMS(frames)

#%% get trackelts
tracklets = get_tracklets(nms_frames)

#%% solve tracklets
final_tracklets = solve_tracklets(tracklets, Nf, max_iterations=100)

print('Elapsed time: ', time()-init)

#%% evaluate predictions

if annotations:
    print(evaluate(annotations, final_tracklets, Nf))

#%% draw trackings
draw_tracklets(tracklets, nms_frames, path_imgs, img_format='.png', plot=DEBUG)

print('Elapsed time with drawing: ', time()-init)

#%%
    
# h,w,c = frame_imgs[0].shape
# out = cv2.VideoWriter(f'./{DATASET}_{DETECTOR}_{len(frames)}.avi',\
#                       cv2.VideoWriter_fourcc(*'XVID'), 10.0, (w,h))
# for img in frame_imgs:
#     out.write(img)
# out.release()

'''
TODO:
    - Remove bad frames
    - Solve hyphotesis matrix in batches of N tracklets/frames
'''

#%%
