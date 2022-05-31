# -*- coding: utf-8 -*-
"""
Example of tracking by detection

@author: Lucas N. Kirsten (lnkirsten@inf.ufrgs.br)
"""

import os
import cv2
from time import time

from tracker import *

#%% read all detections
init = time()

# get sorted frames by name
frame_imgs = [file.split('.')[0] for file in os.listdir(path_imgs)]

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

#%% draw trackings
frame_imgs = draw_tracklets(final_tracklets, nms_frames, path_imgs, \
                            img_format='.png', plot=True, save_video=True)

print('Elapsed time with drawing: ', time()-init)

#%% evaluate predictions

if annotations:
    print(evaluate(annotations, final_tracklets, Nf, 'center'))

'''
TODO:
    - Remove bad frames
    - Solve hyphotesis matrix in batches of N tracklets/frames
'''

#%%
