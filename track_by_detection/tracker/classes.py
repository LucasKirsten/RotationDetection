# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:26:49 2022

@author: kirstenl
"""

import numpy as np
from .func_utils import get_piou, get_from_piou

class Detection():
    def __init__(self,frame,score,cx,cy,\
                 w=None,h=None,ang=None,a=None,b=None,c=None,mit=0,\
                 idx=-1, convert=True):
        self.frame = frame
        self.score = float(score)
        self.mit = int(mit)
        self.idx = idx
        
        self.cx = float(cx)
        self.cy = float(cy)
        
        if convert:
            if (a is None) or (b is None) or (c is None):
                self.w, self.h, self.ang = float(w), float(h), float(ang)
                self.cx,self.cy,self.a,self.b,self.c = \
                    get_piou(self.cx,self.cy,self.w,self.h,self.ang)
                    
            elif (w is None) or (h is None) or (ang is None):
                self.a, self.b, self.c = float(a), float(b), float(c)
                self.cx,self.cy,self.w,self.h,self.ang = \
                    get_from_piou(self.cx,self.cy,self.a,self.b,self.c)
            
            else:
                self.w,self.h,self.ang = float(w), float(h), float(ang)
                self.a,self.b,self.c = float(a),float(b),float(c)
        
class Tracklet():
    def __init__(self, detections, start):
        if type(detections)==list:
            self.detections = detections
            self.sum_score = np.sum([d.score for d in detections])
        else:
            self.detections = [detections]
            self.sum_score = detections.score
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
        self.sum_score += x.score
        self.__add(1)
        
    def score(self):
        return self.sum_score/self.size
        
    # def extend(self, x):
    #     self.detections.extend(x)
    #     self.__add(len(x) if type(x)==list else 1)
    
    # join two tracklets (e.g., for translation hyphotesis)
    def join(self, tracklet):
        
        # if there are gaps between tracklets fill them
        if self.end - tracklet.start>0:
            dx = self.end-tracklet.start
            d0 = self[-1]
            df = tracklet[0]
            for i in range(self.end-tracklet.start):
                
                parms = {'cx':None,'cy':None,'w':None,'h':None,'ang':None}
                for p in parms.keys():
                    parms[p] = d0.__dict__[p] + (df.__dict__[p]-d0.__dict__[p])/dx*(i+1)
                parms.update({'frame':None,'score':0,'mit':0})
                
                self.append(Detection(**parms))
            
        for det in tracklet:
            self.append(det)
        
class Frame(list):
    def get_values(self):
        return np.array([[d.score,d.cx,d.cy,d.w,d.h,d.ang,d.a,d.b,d.c,d.mit] for d in self])
    
    def get_centers(self):
        return np.array([[d.cx,d.cy] for d in self])
    
    def get_idxs(self):
        return np.array([d.idx for d in self])
    
    
    
    
    