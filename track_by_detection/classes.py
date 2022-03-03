# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:26:49 2022

@author: kirstenl
"""

import numpy as np
from func_utils import get_piou

class Detection():
    def __init__(self, frame,score,cx,cy,w,h,ang,mit):
        self.frame = frame
        self.score = score
        self.w = w
        self.h = h
        self.ang = ang
        self.mit = mit
        self.idx = -1
        self.cx,self.cy,self.a,self.b,self.c = \
            get_piou(cx,cy,self.w,self.h,self.ang)
        
class Tracklet():
    def __init__(self, detections, start):
        if type(detections)==list:
            self.detections = detections
        else:
            self.detections = [detections]
        self.start = start
        self.size = len(self.detections)
        self.end = start + self.size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, i):
        return self.detections[i]
    
    def __add(self, k):
        self.size += k
        self.end += k
    
    def append(self, x):
        self.detections.append(x)
        self.__add(1)
        
    def extend(self, x):
        self.detections.extend(x)
        self.__add(len(x) if type(x)==list else 1)
    
    # join two tracklets (e.g., for translation hyphotesis)
    def join(self, tracklet):
        
        # if there are gaps between tracklets fill them
        if self.end - tracklet.start>0:
            print('filling gaps...')
            dx = self.end-tracklet.start
            d0 = self[-1]
            df = tracklet[0]
            for i in range(self.end-tracklet.start):
                
                parms = {'cx':None,'cy':None,'w':None,'h':None,'ang':None}
                for p in parms.keys():
                    parms[p] = d0.__dict__[p] + (df.__dict__[p]-d0.__dict__[p])/dx*(i+1)
                parms.update({'frame':None,'score':None,'mit':0})
                
                self.append(Detection(**parms))
            
        for det in tracklet:
            self.append(det)
        
class Frame(list):
    def get_values(self):
        return np.array([[d.cx,d.cy,d.w,d.h,d.ang,d.a,d.b,d.c] for d in self])
    
    
    
    
    
    
    
    