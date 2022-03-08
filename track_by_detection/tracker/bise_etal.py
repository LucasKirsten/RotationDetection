# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 18:49:44 2022

@author: kirstenl
"""

import numpy as np
import cvxpy
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
NUM_CORES = multiprocessing.cpu_count()

from .configs import *
from .func_utils import *
from .frames_utils import _adjust_tracklets

#%% make hyphotesis
def _make_hypotheses(tracklets, Nf):
    
    Nx = len(tracklets)
    SIZE = 2*Nx+Nf
    
    #for k,track in enumerate(tracklets):
    def __get_hyphoteses(k):
        
        track = tracklets[k]
    
        hyp,C,pho = [],[],[]
            
        # determine alpha value based on normal and mitoses precision
        alpha = ALPHA_NORMAL if int(track[-1].mit)==0 else ALPHA_MITOSE
        
        # initialization hypothesis
        if track.start<INIT_TH:
            Ch = np.zeros(SIZE, dtype='bool')
            Ch[[i==Nx+k for i in range(SIZE)]] = 1
            ph = Pini(track)*PTP(track, alpha, track.score())
        
            hyp.append(f'init_{k}')
            C.append(Ch)
            pho.append(ph)
        
        # false positive hypothesis
        Ch = np.zeros(SIZE, dtype='bool')
        Ch[[i==k or i==Nx+k for i in range(SIZE)]] = 1
        ph = PFP(track, alpha, track.score())
        
        hyp.append(f'fp_{k}')
        C.append(Ch)
        pho.append(ph)
        
        # translation and mitoses hypothesis
        has_mitoses  = any([int(det.mit)==1 for det in track])
        dist_mitoses = track.start + np.array([i for i,det in enumerate(track) if int(det.mit)==1])
        term_hyp_prob = 0
        k2 = k
        for track2 in tracklets[k+1:]:
            k2 += 1
            
            if track2.start-track.start>TRANSP_TH:
                break
            
            if track.start==track2.start:
                continue
            
            # calculate center distances
            cnt_dist = center_distances(track[0].cx,track[0].cy, track2[-1].cx,track2[-1].cy)
            if cnt_dist>CENTER_TH:
                continue
            
            # translation hypothesis
            Ch = np.zeros(SIZE, dtype='bool')
            Ch[[i==k or i==Nx+k2 for i in range(SIZE)]] = 1
            Ch[2*Nx+track2.start] = 1
            ph = Plink(track, track2, cnt_dist)
            
            hyp.append(f'transl_{k},{k2}')
            C.append(Ch)
            pho.append(ph)
            
            term_hyp_prob = max(term_hyp_prob, ph)
            
            # mitoses hypothesis
            if has_mitoses:
                
                # mitoses distance in frames
                d_mit = min(abs(dist_mitoses-track2.start))
                if d_mit>MIT_TH:
                    continue
                
                # calculate center distances
                cnt_dist = center_distances(track[0].cx,track[0].cy, track2[-1].cx,track2[-1].cy)
                if cnt_dist>CENTER_MIT_TH:
                    continue
                    
                Ch = np.zeros(SIZE, dtype='bool')
                Ch[[i==k or i==Nx+k2 for i in range(SIZE)]] = 1
                Ch[2*Nx+track2.start] = 1
                ph = Pmit(track, track2, cnt_dist, d_mit)
                
                hyp.append(f'mit_{k},{k2}')
                C.append(Ch)
                pho.append(ph)
                
                term_hyp_prob = max(term_hyp_prob, ph)
                    
        # termination hypothesis
        Ch = np.zeros(SIZE, dtype='bool')
        Ch[[i==k for i in range(SIZE)]] = 1
        ph = 1 - term_hyp_prob
        
        hyp.append(f'term_{k}')
        C.append(Ch)
        pho.append(ph)
        
        yield hyp,C,pho
        
    return __get_hyphoteses

#%% populate C and pho matrixes for hypothesis
def _get_C_pho_matrixes(tracklets, Nf):
    
    Nx = len(tracklets)
    pbar = tqdm(range(Nx))
    pbar.set_description('Bulding hyphotesis matrix: ')
    
    generator = _make_hypotheses(tracklets, Nf)
    hyp,C,pho = [],[],[]
    def _populate_matrixes(i):
        h,c,p = next(generator(i))
        hyp.extend(h)
        C.extend(c)
        pho.extend(p)
        
    with Parallel(n_jobs=NUM_CORES, prefer='threads') as parallel:
        _ = parallel(delayed(_populate_matrixes)(i) for i in pbar)
    
    return np.array(C), np.float32(pho), hyp

#%% solve integer optimization problem

def solve_tracklets(tracklets, Nf):
    
    # get hypothesis matrixes
    C, pho, hyp = _get_C_pho_matrixes(tracklets, Nf)
    
    x = cvxpy.Variable((len(pho),1), boolean=True)
    
    # define and solve integer optimization
    constrains = [(cvxpy.transpose(C) @ x) <= 1]
    total_prob = cvxpy.sum(cvxpy.transpose(pho) @ x)
    
    print('Solving integer optimization...')
    knapsack_problem = cvxpy.Problem(cvxpy.Maximize(total_prob), constrains)
    knapsack_problem.solve(solver=cvxpy.GLPK_MI)
    
    # get the true hypothesis
    x = np.squeeze(x.value).astype('int')
    hyp = np.array(hyp)[x==1]
    
    print('Adjusting final tracklets...')
    final_tracklets = _adjust_tracklets(tracklets, hyp)
            
    return final_tracklets







