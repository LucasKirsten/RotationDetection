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

DEBUG = False

#%% helinger distance
EPS = 1e-3

def helinger_dist(x1,y1,a1,b1,c1, x2,y2,a2,b2,c2):
    
    B1 = (a1+a2)*(y1-y2)**2 + (b1+b2)*(x1-x2)**2
    B1 += 2*(c1+c2)*(x2-x1)*(y1-y2)
    B1 /= (a1+a2)*(b1+b2)-(c1+c2)**2+EPS
    B1 *= 1/4
    
    B2 = (a1+a2)*(b1+b2)-(c1+c2)**2
    B2 /= 4*np.sqrt((a1*b1-c1**2)*(a2*b2-c2**2))+EPS
    B2 = 1/2*np.log(B2)
    
    Bd = B1+B2
    Bc = np.exp(-Bd)
    
    Hd = np.sqrt(1-Bc+EPS)
    
    return Hd

def get_probiou_values(array):
    cx,cy,w,h,angle = array
    angle *= np.pi/180
    
    # get ProbIoU values
    al = w**2./12.
    bl = h**2./12.
    
    a = al*np.cos(angle)**2+bl*np.sin(angle)**2
    b = al*np.sin(angle)**2+bl*np.cos(angle)**2
    c = 1/2*(al-bl)*np.sin(2*angle)
    return cx,cy,a,b,c

#%% read all detections

# get only some frames
frame_imgs = [file.split('.')[0] for file in os.listdir('frames/Dados CytoSMART/cytosmart arvores luana')]
frame_imgs = sorted(frame_imgs)[:101]

# open normal detections
with open('frames/Dados CytoSMART/r2cnn/det_normal_cell.txt', 'r') as file:
    det_normal = file.read()
    det_normal = det_normal.split('\n')
    normal_detections = []
    for det in det_normal:
        if det.split(' ')[0] not in frame_imgs:
            continue
        normal_detections.append(det.split(' '))

# open mitoses detections
with open('frames/Dados CytoSMART/r2cnn/det_mitoses.txt', 'r') as file:
    det_mitoses = file.read()
    det_mitoses = det_mitoses.split('\n')
    mitoses_detections = []
    for det in det_mitoses:
        if det.split(' ')[0] not in frame_imgs:
            continue
        mitoses_detections.append(det.split(' '))
    
# merge detections
detections = np.concatenate([normal_detections, mitoses_detections], axis=0)

# sort detections by name
detections = sorted(detections, key=lambda x:x[0])

#%% split detections into frames

frames, frame_detections = [],[]
for i in range(len(detections)-1):
    if detections[i][0]==detections[i+1][0]:
        frame_detections.append(detections[i])
    else:
        frames.append(frame_detections)
        frame_detections = []

#%% display some frames

if DEBUG:
    path_imgs = 'frames/Dados CytoSMART/cytosmart arvores luana'
    
    for frm in frames[:10]:
        img_name = frm[0][0]
        img_name = os.path.join(path_imgs, img_name+'.jpg')
        img = cv2.imread(img_name)[...,::-1]
        draw = np.copy(img)
        
        boxes = []
        for det in frm:
            score,cx,cy,w,h,a = map(lambda x:float(x), det[1:])
            box = cv2.boxPoints(((cx,cy),(w,h),a))
            box = np.int0(box)
            boxes.append(box)
            
        draw = cv2.drawContours(draw, boxes, -1, (0,0,255), 2)
        plt.figure()
        plt.imshow(draw)

#%% hungarian algorithm over consecutive frames

# add a indexing value for each frame
for i in range(len(frames)):
    if i==0:
        frames[i] = [np.concatenate([fr, [n]]) for n,fr in enumerate(frames[i])]
    else:
        frames[i] = [np.concatenate([fr, [-1]]) for fr in frames[i]]
    frames[i] = np.array(frames[i])

# iterate over frames to apply hungarian algorithm
ids = set(range(len(frames[0]))) # set of indexes
for i in range(len(frames)-1):
    
    # take consecutive frames
    frm0 = frames[i]
    frm1 = frames[i+1]
    
    costs = np.zeros((len(frm0), len(frm1)))
    
    # calculate the cost matrix
    for j in range(costs.shape[0]):
        for k in range(costs.shape[1]):
            v0 = np.float32(frm0[j][2:-1])
            v1 = np.float32(frm1[k][2:-1])
            piou0 = get_probiou_values(v0)
            piou1 = get_probiou_values(v1)
            costs[j,k] = helinger_dist(*piou0, *piou1)
            
    # hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(costs)
    
    # map the detected objects to its pairs
    for row,col in zip(row_ind, col_ind):
        frames[i+1][col][-1] = frames[i][row][-1]
    
    # add a new id for new detections
    for k in range(len(frm1)):
        fr_id = float(frames[i+1][k][-1])
        if fr_id==-1:
            frames[i+1][k][-1] = max(ids) + 1
            ids.add(float(frames[i+1][k][-1]))

#%%

DEBUG = False

if DEBUG:
    path_imgs = 'frames/Dados CytoSMART/cytosmart arvores luana'
    
    colors = np.linspace(10,200,len(ids)+1,dtype='uint8')[1:]
    
    frame_images = []
    for frm in frames:
        img_name = frm[0][0]
        img_name = os.path.join(path_imgs, img_name+'.jpg')
        img = cv2.imread(img_name)
        draw = np.copy(img)
        
        for det in frm:
            score,cx,cy,w,h,a = map(lambda x:float(x), det[1:-1])
            #if float(det[-1])==-1:
            #    continue
            obj_id = det[-1]
            if obj_id.isnumeric():
                obj_id = int(float(obj_id))
                cor = (int(colors[obj_id]), int(colors[int(len(ids)-obj_id-1)]), 255)
            else:
                cor = (0,0,0)
            
            box = cv2.boxPoints(((cx,cy),(w,h),a))
            box = np.int0(box)
            
            draw = cv2.drawContours(draw, [box], -1, cor, 2)
            draw = cv2.putText(draw, str(obj_id), (int(cx),int(cy)), \
                               cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 1, cv2.LINE_AA)
        
        frame_images.append(draw)
        
        cv2.imshow('', draw)
        cv2.waitKey(200)
    cv2.destroyAllWindows()
    
    # h,w,c = draw.shape
    # out = cv2.VideoWriter('./cell_frames.avi',cv2.VideoWriter_fourcc(*'XVID'), 5.0, (w,h))
    # for img in frame_images:
    #     out.write(img)
    # out.release()


#%%











