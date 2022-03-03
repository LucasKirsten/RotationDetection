# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:29:09 2021

@author: kirstenl
"""

import os
import cv2
import numpy as np
from itertools import combinations

OBJECTS  = 20
FRAMES   = 20
MAX_SIZE = 100
MIN_SIZE = 30
WIN_SIZE = 512
VARIATION_RATIO = 0.2
VERBOSE = True

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
    
    return 1.-Hd

def get_probiou_values(array):
    cx,cy,w,h,angle,_ = array
    
    # get ProbIoU values
    al = w**2./12.
    bl = h**2./12.
    
    a = al*np.cos(angle)**2+bl*np.sin(angle)**2
    b = al*np.sin(angle)**2+bl*np.cos(angle)**2
    c = 1/2*(al-bl)*np.sin(2*angle)
    return cx,cy,a,b,c

#%% Initialize random objects

if VERBOSE: print('Initializing objects...')

objects = []
for label in range(OBJECTS):
    cx = np.random.randint(MAX_SIZE, WIN_SIZE-MAX_SIZE)
    cy = np.random.randint(MAX_SIZE, WIN_SIZE-MAX_SIZE)
    w = np.random.randint(MIN_SIZE, MAX_SIZE)
    h = np.random.randint(MIN_SIZE, MAX_SIZE)
    angle = np.random.randint(1,90)
    
    objects.append([cx,cy,w,h,angle,label])
objects = np.float32(objects)

# color for the objects
colors = np.linspace(10,200,OBJECTS+1,dtype='uint8')[1:]
    
#%% Create the frames from the objects

if VERBOSE: print('Creating random frames...')

frames = []
for fr in range(FRAMES):
    frame = np.ones((WIN_SIZE,WIN_SIZE,3), dtype='uint8')*255
    objects[:,:2]  *= 1+(np.random.random(size=objects[:,:2].shape)-0.5)*VARIATION_RATIO*2
    objects[:,:2]  = np.clip(objects[:,:2], 0, WIN_SIZE)
    objects[:,2:4] *= 1+(np.random.random(size=objects[:,2:4].shape)-0.5)*VARIATION_RATIO*2
    objects[:,-2]  *= 1+(np.random.random(size=objects[:,-2].shape)-0.5)*VARIATION_RATIO*2
    objects[:,-2] = -objects[:,-2]%90
    
    frames.append(np.array(objects))
    
    if VERBOSE:
        for i in range(len(objects)):
            cor = (int(colors[i]), int(colors[OBJECTS-i-1]), 255)
            cx,cy,w,h,angle,lb = objects[i]
            cnt = ((cx,cy),(w,h),angle)
            
            box = cv2.boxPoints(cnt)
            box = np.int0(box)
            cv2.drawContours(frame, [box], -1, cor, 1)
            cv2.putText(frame, str(int(lb)), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1, cv2.LINE_AA)
        
        cv2.imshow('', frame)
        cv2.waitKey(150)
cv2.destroyAllWindows()

#%% Evaluate tracker with Helinger distance between two sequential frames
# montar algoritmo hungaro
# prever falsos positivos e negativos
# morte celular
# mitose celular

if VERBOSE: print('Evaluating tracking...')

all_values,acc = [],[]
for i in range(len(frames)-1):
    obj1 = frames[i]
    obj2 = frames[i+1]
    np.random.shuffle(obj2)
    for o1 in obj1:
        values = []
        for o2 in obj2:
            l1 = helinger_dist(*get_probiou_values(o1),*get_probiou_values(o2))
            values.append(l1)
            all_values.append(l1)
        
        acc.extend([1 if o1[-1]==obj2[np.argmax(values)][-1] else 0])

print('Tracking accuracy: ', str(np.sum(acc)/len(acc)))

