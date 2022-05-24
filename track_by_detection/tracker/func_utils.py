# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 18:43:43 2022

@author: kirstenl
"""

import numpy as np
from numpy import linalg as LA
import pandas as pd
from numba import njit
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed
NUM_CORES = multiprocessing.cpu_count()

from .configs import *

#%% intersection over union

@njit
def intersection_over_union(cxA,cyA,wA,hA, cxB,cyB,wB,hB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(cxA-wA/2, cxB-wB/2)
	yA = max(cyA-hA/2, cyB-hB/2)
	xB = min(cxA+wA/2, cxB+wB/2)
	yB = min(cyA+hA/2, cyB+hB/2)

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth rectangles
	boxAArea = (wA + 1) * (hA + 1)
	boxBArea = (wB + 1) * (hB + 1)
	iou = interArea / float(boxAArea + boxBArea - interArea)
    
	return iou

#%% helinger distance
EPS = 1e-3

@njit
def helinger_dist(x1,y1,a1,b1,c1, x2,y2,a2,b2,c2, shape_weight=1.):
    
    B1 = (a1+a2)*(y1-y2)**2. + (b1+b2)*(x1-x2)**2.
    B1 += 2.*(c1+c2)*(x2-x1)*(y1-y2)
    B1 /= (a1+a2)*(b1+b2)-(c1+c2)**2+EPS
    B1 *= 1./4.
    
    B2 = (a1+a2)*(b1+b2)-(c1+c2)**2.
    B2 /= 4.*np.sqrt((a1*b1-c1**2.)*(a2*b2-c2**2.))+EPS
    B2 = 1./2.*np.log(B2)
    
    Bd = B1+shape_weight*B2
    Bc = np.exp(-Bd)
    
    Hd = np.sqrt(1.-Bc+EPS)
    
    if Hd>1:
        return 1
    elif Hd<0 or np.isnan(Hd):
        return 0
    return Hd

@njit
def get_piou(cx,cy,w,h,angle):
    # get ProbIoU values
    angle *= np.pi/180.
    
    al = w**2./12.
    bl = h**2./12.
    
    a = al*np.cos(angle)**2.+bl*np.sin(angle)**2.
    b = al*np.sin(angle)**2.+bl*np.cos(angle)**2.
    c = 1./2.*(al-bl)*np.sin(2.*angle)
    return cx,cy,a,b,c

@njit
def get_from_piou(cx,cy,a,b,c):
    
    corr = np.array([[a,c],[c,b]])
    val,vec = LA.eig(corr)
    w = np.sqrt(np.abs(val[0])*12.)
    h = np.sqrt(np.abs(val[1])*12.)
    ang = vec[1][0]*180./np.pi
    
    return cx,cy,w,h,ang

@njit
def center_distances(x1,y1, x2,y2):
    return np.sqrt(np.square(x1-x2)+np.square(y1-y2))

#%% functions to calculate probabilities

def PFP(Xk, alpha, score):
    return (1-alpha)**len(Xk)

def PTP(Xk, alpha, score):
    return 1 - PFP(Xk, alpha, score)

def Pini(Xk):
    dt0 = Xk.start
    return np.exp(-dt0/INIT_FACTOR)

def Pterm(Xk, total_frames):
    dt0 = total_frames - Xk.end
    return np.exp(-dt0/INIT_FACTOR)

def Plink(Xj, Xi, cnt_dist):
    featij = cnt_dist
    featij *= Xj.end-Xi.start+1
    
    return np.exp(-np.abs(featij)/LINK_FACTOR)

def Pmit(Xj, Xi, cnt_dist, d_mit):
    featij = cnt_dist
    featij *= d_mit*abs(Xj.start-Xi.start+1)
    
    return np.exp(-np.abs(featij)/MIT_FACTOR)




    
    