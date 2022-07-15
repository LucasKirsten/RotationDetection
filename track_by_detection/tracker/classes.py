# -*- coding: utf-8 -*-
"""
Auxiliar Classes for the tracking algorithm

@author: Lucas N. Kirsten (lnkirsten@inf.ufrgs.br)
"""

import cv2
import numpy as np
from shapely.geometry import Polygon

from .func_utils import get_hd, get_from_hd

class Detection():
    def __init__(self,frame:str,score:float,cx:float,cy:float,\
                 w:float=None,h:float=None,ang:float=None,\
                 a:float=None,b:float=None,c:float=None,mit:int=0,\
                 idx:int=-1, convert:bool=True):
        '''
        Define a detection.

        Parameters
        ----------
        frame : str
            Name of the detection frame.
        score : float
            Detection score.
        cx : float
            x center position.
        cy : float
            y center position.
        w : float, optional
            Width size. The default is None.
        h : float, optional
            Height size. The default is None.
        ang : float, optional
            Angle value. The default is None.
        a : float, optional
            a value from the covariance matrix. The default is None.
        b : float, optional
            b value from the covariance matrix. The default is None.
        c : float, optional
            c value from the covariance matrix. The default is None.
        mit : int, optional
            If detection is mitose (mit=1) or not (mit=0). The default is 0.
        idx : int, optional
            Index value of detection (detection with same idx belongs to same tracklet). The default is -1.
        convert : bool, optional
            If to convert values of one representation to another (e.g., standard (cx,cy,w,h,ang) to (cx,cy,a,b,c)). The default is True.
        '''
        
        self.frame = frame
        self.score = float(score)
        self.mit = int(mit)
        self.idx = idx
        
        self.cx = float(cx)
        self.cy = float(cy)
        self.area = None
        
        if convert:
            if (a is None) or (b is None) or (c is None):
                self.w, self.h, self.ang = float(w), float(h), float(ang)
                self.cx,self.cy,self.a,self.b,self.c = \
                    get_hd(self.cx,self.cy,self.w,self.h,self.ang)
                    
            elif (w is None) or (h is None) or (ang is None):
                self.a, self.b, self.c = float(a), float(b), float(c)
                self.cx,self.cy,self.w,self.h,self.ang = \
                    get_from_hd(self.cx,self.cy,self.a,self.b,self.c)
            
            else:
                self.w,self.h,self.ang = float(w), float(h), float(ang)
                self.a,self.b,self.c = float(a),float(b),float(c)
        
        else:
            self.w,self.h,self.ang,self.a,self.b,self.c = 0,0,0,0,0,0
                
        if self.w is not None and self.h is not None:
            self.area = self.w*self.h
            box = cv2.boxPoints(((self.cx,self.cy),(self.w,self.h),self.ang))
            box = np.int0(box)
            self.box = Polygon(box)
                
    def __str__(self):
        return str(self.__dict__)
    
    def iou(self, det) -> float:
        '''
        Get the intersection value percentage between two detections.

        Parameters
        ----------
        det : Detection
            Detection to compute the intersection with.

        Returns
        -------
        float
            Intersection value between the two detections.

        '''
        return self.box.intersection(det.box).area / self.box.union(det.box).area
        
class Tracklet():
    def __init__(self, detections, start:int):
        '''
        Define a Tracklet.

        Parameters
        ----------
        detections : list or Detection
            List of detections, or single detection to compose Tracklet.
        start : int
            Start value relative to the frames.
        '''
        
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
        '''
        Join one tracklet to another.

        Parameters
        ----------
        tracklet : Tracklet
            The tracklet to be joined with.
        '''
        
        assert tracklet.start>self.end, 'Trying to join non consecutive tracklets!'
        
        # if there are gaps between tracklets, linearly fill them
        if tracklet.start - self.end > 1:
            dx = tracklet.start - self.end - 1
            d0 = self[-1]
            df = tracklet[0]
            for i in range(dx):
                
                parms = {'cx':None,'cy':None,'w':None,'h':None,'ang':None}
                for p in parms.keys():
                    parms[p] = d0.__dict__[p] + (i+1)*(df.__dict__[p]-d0.__dict__[p])/dx
                parms.update({'frame':None,'score':0,'mit':0})
                
                self.append(Detection(**parms))
            
        for det in tracklet:
            self.append(det)
        
class Frame(list):
    def __init__(self, values:list=[], name:str=None):
        '''
        Define a Frame.

        Parameters
        ----------
        values : list, optional
            A list of values to the initially define the Frame list. The default is [].
        name : str, optional
            Frame image name. The default is None.
        '''
        super(Frame, self).__init__(values)
        self.name = name
    
    def get_values(self):
        '''
        Returns a numpy array containing all score,cx,cy,w,h,ang,a,b,c,d,mit values of detections in the Tracklet.
        '''
        return np.array([[d.score,d.cx,d.cy,d.w,d.h,d.ang,d.a,d.b,d.c,d.mit]\
                         for d in self])
    
    def get_centers(self):
        '''
        Returns a numpy array containing all cx,cy values of detections in the Tracklet.
        '''
        return np.array([[d.cx,d.cy] for d in self])
    
    def get_iou_values(self):
        '''
        Returns a numpy array containing all cx,cy,w,h,ang values of detections in the Tracklet.
        '''
        return np.array([[d.cx,d.cy,d.w,d.h,d.ang] for d in self])
    
    def get_hd_values(self):
        '''
        Returns a numpy array containing all cx,cy,a,b,c values of detections in the Tracklet.
        '''
        return np.array([[d.cx,d.cy,d.a,d.b,d.c] for d in self])
    
    def get_idxs(self):
        '''
        Returns a numpy array containing all idx values of detections in the Tracklet.
        '''
        return np.array([d.idx for d in self])
    
    
    
    
    