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
from scipy.special import softmax

DEBUG = False

#%% helinger distance
EPS = 1e-3

def helinger_dist(x1,y1,a1,b1,c1, x2,y2,a2,b2,c2, shape_weight=1):
    
    B1 = (a1+a2)*(y1-y2)**2 + (b1+b2)*(x1-x2)**2
    B1 += 2*(c1+c2)*(x2-x1)*(y1-y2)
    B1 /= (a1+a2)*(b1+b2)-(c1+c2)**2+EPS
    B1 *= 1/4
    
    B2 = (a1+a2)*(b1+b2)-(c1+c2)**2
    B2 /= 4*np.sqrt((a1*b1-c1**2)*(a2*b2-c2**2))+EPS
    B2 = 1/2*np.log(B2)
    
    Bd = B1+shape_weight*B2
    Bc = np.exp(-Bd)
    
    Hd = np.sqrt(1-Bc+EPS)
    
    return np.clip(Hd, 0, 1)

def get_probiou_values(cx,cy,w,h,angle):
    angle *= np.pi/180
    
    # get ProbIoU values
    al = w**2./12.
    bl = h**2./12.
    
    a = al*np.cos(angle)**2+bl*np.sin(angle)**2
    b = al*np.sin(angle)**2+bl*np.cos(angle)**2
    c = 1/2*(al-bl)*np.sin(2*angle)
    return cx,cy,a,b,c

def center_distances(x1,y1, x2,y2):
    return np.sqrt(np.square(x1-x2)+np.square(y1-y2))

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
detections = []
for det in normal_detections:
    detections.append(det+[0])
for det in mitoses_detections:
    detections.append(det+[1])

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

DEBUG = False

if DEBUG:
    path_imgs = 'frames/Dados CytoSMART/cytosmart arvores luana'
    
    for frm in frames[:10]:
        img_name = frm[0][0]
        img_name = os.path.join(path_imgs, img_name+'.jpg')
        img = cv2.imread(img_name)[...,::-1]
        draw = np.copy(img)
        
        for det in frm:
            score,cx,cy,w,h,a,mit = map(lambda x:float(x), det[1:])
            box = cv2.boxPoints(((cx,cy),(w,h),a))
            box = np.int0(box)
            
            color = (0,0,255) if int(mit)==0 else (0,255,0)
            draw = cv2.drawContours(draw, [box], -1, color, 2)
        plt.figure()
        plt.imshow(draw)

#%% add a indexing value for each frame

for i in range(len(frames)):
    if i==0:
        frames[i] = [np.concatenate([fr, [n]]) for n,fr in enumerate(frames[i])]
    else:
        frames[i] = [np.concatenate([fr, [-1]]) for fr in frames[i]]
    frames[i] = np.array(frames[i])

#%% iterate over frames to apply hungarian algorithm

# initialize trackelts
tracklets = [[0, det] for det in frames[0]]

# number of detected mitoses
num_mitoses = 0

ids = set(range(len(frames[0]))) # set of indexes
for i in range(len(frames)-1):
    
    # take consecutive frames
    frm0 = frames[i]
    frm1 = frames[i+1]
    
    costs = np.zeros((len(frm0), len(frm1)))
    
    # calculate the cost matrix
    for j in range(costs.shape[0]):
        for k in range(costs.shape[1]):
            v0 = np.float32(frm0[j][2:-2])
            v1 = np.float32(frm1[k][2:-2])
            piou0 = get_probiou_values(*v0)
            piou1 = get_probiou_values(*v1)
            costs[j,k] = helinger_dist(*piou0, *piou1)
            
    # hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(costs)
    
    # map the detected objects to its pairs
    for row,col in zip(row_ind, col_ind):
        # look for mitoses spliting
        if int(frm0[row][-2])==1 and int(frm1[col][-2])==0:
            frames[i+1][col][-1] = -1
            num_mitoses += 1
        else:
            frames[i+1][col][-1] = frames[i][row][-1]
    
    # add a new id for new detections
    for k in range(len(frm1)):
        fr_id = float(frames[i+1][k][-1])
        if fr_id==-1:
            frames[i+1][k][-1] = max(ids) + 1
            ids.add(float(frames[i+1][k][-1]))
            
    # add objects to tracklets
    for det in frames[i+1]:
        det_id = int(float(det[-1]))
        if det_id>=len(tracklets):
            tracklets.append([i+1, det])
        else:
            tracklets[det_id].append(det)

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


#%% functions to calculate probabilities

def PFP(Xk, alpha=0.5):
    return alpha**(len(Xk)-1)

def PTP(Xk, alpha=0.2):
    return 1 - PFP(Xk, alpha)

def Pini(Xk, thetat=10, lambda1=5):
    dt0 = Xk[0]
    
    if dt0<thetat:
        return np.exp(-dt0/lambda1)
    else:
        return 1e-6

def Plink(Xj, Xi,\
          g=lambda x,y:center_distances(*x, *y), lambda3=300):
    featij = g(map(lambda x:float(x), Xj[1][2:-5]), map(lambda x:float(x), Xi[-1][2:-5]))
    featij *= abs(Xj[0]-Xi[0]+1)
    
    return np.exp(-np.abs(featij)/lambda3)

def Pdiv(Xp, Xc1, Xc2,\
         g=lambda x,y:center_distances(*x, *y), lambda3=300):
    featpc1 = g(map(lambda x:float(x), Xp[-1][2:-5]), map(lambda x:float(x), Xc1[1][2:-5]))
    featpc1 *= abs(Xp[0]-Xc1[0]+1)
    featpc2 = g(map(lambda x:float(x), Xp[-1][2:-5]), map(lambda x:float(x), Xc2[1][2:-5]))
    featpc2 *= abs(Xp[0]-Xc2[0]+1)
    
    return np.exp(-(np.abs(featpc1)+np.abs(featpc2))/(2*lambda3))

#%% populate C and pho matrixes for hypothesis

Nx = len(tracklets)
hyp,C,pho = [],[],[]
for k,track in enumerate(tracklets):
    
    # determine alpha value based on normal and mitoses AP50
    alpha = 0.2267 if int(track[-1][-2])==0 else 0.3035
    
    # initialization hypothesis
    if track[0]<10:
        Ch = np.zeros(2*Nx+len(frames))
        Ch[[i==Nx+k for i in range(2*Nx+len(frames))]] = 1
        ph = Pini(track)*PTP(track, alpha)
    
        hyp.append(f'init_{k}')
        C.append(Ch)
        pho.append(ph)
    
    # false positive hypothesis
    Ch = np.zeros(2*Nx+len(frames))
    Ch[[i==k or i==Nx+k for i in range(2*Nx+len(frames))]] = 1
    ph = PFP(track, alpha)
    
    hyp.append(f'fp_{k}')
    C.append(Ch)
    pho.append(ph)
    
    has_mitoses = any([int(det[-2])==1 for det in track[1:]])
    term_hyp_prob = 0
    k2 = k
    for track2 in tracklets[k+1:]:
        k2 += 1
        
        if track2[0]>track[0]+10 or track[0]==track2[0]:
            continue
        
        # translation hypothesis
        Ch = np.zeros(2*Nx+len(frames))
        Ch[[i==k or i==Nx+k2 for i in range(2*Nx+len(frames))]] = 1
        Ch[2*Nx+track2[0]] = 1
        ph = Plink(track, track2)
        
        hyp.append(f'transl_{k},{k2}')
        C.append(Ch)
        pho.append(ph)
        
        term_hyp_prob = max(term_hyp_prob, ph)
        
        # dividing hypothesis
        if has_mitoses:
            if track2[0]>track[0]+3:
                continue
            
            k3 = k2
            for track3 in tracklets[k2+1:]:
                k3 += 1
                
                if track3[0]>track2[0]+3:
                    continue
                
                Ch = np.zeros(2*Nx+len(frames))
                Ch[[i==k or i==Nx+k2 or i==Nx+k3 for i in range(2*Nx+len(frames))]] = 1
                Ch[2*Nx+track2[0]] = 1
                ph = Pdiv(track, track2, track3)
                
                hyp.append(f'mit_{k},{k2},{k3}')
                C.append(Ch)
                pho.append(ph)
                
                term_hyp_prob = max(term_hyp_prob, ph)
                
    # termination hypothesis
    Ch = np.zeros(2*Nx+len(frames))
    Ch[[i==k for i in range(2*Nx+len(frames))]] = 1
    ph = 1 - term_hyp_prob
    
    hyp.append(f'term_{k}')
    C.append(Ch)
    pho.append(ph)

C = np.array(C)
pho = np.array(pho)

#%% solve integer optimization problem

import cvxpy

x = cvxpy.Variable((len(pho),1), boolean=True)

idx_init = ['init' in h for h in hyp]
idx_term = ['term' in h for h in hyp]

constrains = [(C.T @ x) <= 1]
total_prob = cvxpy.sum(pho.T @ x)

knapsack_problem = cvxpy.Problem(cvxpy.Maximize(total_prob), constrains)
knapsack_problem.solve(solver=cvxpy.GLPK_MI)

x = x.value

#%% evalute the hypothesis

final_tracklets = {i:[] for i in range(len(tracklets))}
for i,h in enumerate(hyp):
    if int(x[i])==0:
        continue
    
    print(h)
    
    mode,vals = h.split('_')
    if 'init' in mode:
        final_tracklets[int(vals)].append(tracklets[int(vals)])
    
    elif 'transl' in mode:
        v1,v2 = vals.split(',')
        #if len(final_tracklets[int(v1)])<1:
        #    continue
        final_tracklets[int(v1)].append(tracklets[int(v2)])
        
    elif 'mit' in mode:
        v1,v2,v3 = vals.split(',')
        final_tracklets[int(v2)].append(tracklets[int(v2)])
        final_tracklets[int(v3)].append(tracklets[int(v3)])
        
    #break

#%% draw trackings

path_imgs = 'frames/Dados CytoSMART/cytosmart arvores luana'

# draw detections from CNNs
frame_imgs = []
for frm in frames:
    img_name = frm[0][0]
    img_name = os.path.join(path_imgs, img_name+'.jpg')
    img = cv2.imread(img_name)[...,::-1]
    draw = np.copy(img)
    
    for det in frm:
        cx,cy,w,h,a,mit = map(lambda x:float(x), det[2:-1])
        box = cv2.boxPoints(((cx,cy),(w,h),a))
        box = np.int0(box)
        
        color = (0,0,255) if int(mit)==0 else (0,255,0)
        draw = cv2.drawContours(draw, [box], -1, color, 2)
        
    frame_imgs.append(draw)

#%%
total_detections = len([1 for track in final_tracklets.values() if len(track)>0])

colors = np.linspace(10,240,total_detections+1,dtype='uint8')[1:]
det_id = 0
for nti, track in enumerate(final_tracklets.values()):
    if len(track)<1:
        continue
    
    frame_id,di = None,None
    for ti, tracklet in enumerate(track):
        # add gap detections
        if ti>0:
            gap_frame = tracklet[0]
            for gdi in range(gap_frame-frame_id):
                
                frame_imgs[frame_id+gdi] = cv2.drawContours(frame_imgs[frame_id+gdi], [box], -1, cor, 2)
                frame_imgs[frame_id+gdi] = cv2.putText(frame_imgs[frame_id+gdi], str(det_id), (int(cx),int(cy)), \
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 1, cv2.LINE_AA)
                
        frame_id, detections = tracklet[0], tracklet[1:]
        cor = (int(colors[det_id]), int(colors[int(total_detections-det_id-1)]), 255)
        
        for di, det in enumerate(detections):
            cx,cy,w,h,a = map(lambda x:float(x), det[2:-2])
            
            box = cv2.boxPoints(((cx,cy),(w,h),a))
            box = np.int0(box)
            
            frame_imgs[frame_id+di] = cv2.drawContours(frame_imgs[frame_id+di], [box], -1, cor, 2)
            frame_imgs[frame_id+di] = cv2.putText(frame_imgs[frame_id+di], str(det_id), (int(cx),int(cy)), \
                               cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 1, cv2.LINE_AA)
    
    det_id += 1
    
for img in frame_imgs:
    cv2.imshow('', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
    
h,w,c = frame_imgs[0].shape
out = cv2.VideoWriter('./cell_frames.avi',cv2.VideoWriter_fourcc(*'XVID'), 2.0, (w,h))
for img in frame_imgs:
    out.write(img)
out.release()

            
            
    
    






