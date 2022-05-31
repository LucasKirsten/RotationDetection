# -*- coding: utf-8 -*-
"""
Code for solving the Bise et al algorithm
"RELIABLE CELL TRACKING BY GLOBAL DATA ASSOCIATION"

@author: Lucas N. Kirsten (lnkirsten@inf.ufrgs.br)
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import cvxpy
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed
import multiprocessing
NUM_CORES = multiprocessing.cpu_count()

from .configs import *
from .func_utils import *

#%% make hyphotesis
def _make_hypotheses(tracklets, Nf):
    
    Nx = len(tracklets)
    SIZE = 2*Nx+Nf
    
    # define generator for hyphoteses
    def __get_hyphoteses(k):
        
        track = tracklets[k]
        
        # intialize variables
        hyp,C,pho = [],[],[]
            
        # determine alpha value based on normal and mitoses precision
        num_mit = np.sum([int(det.mit) for det in track])
        num_normal = len(track) - num_mit
        alpha = (num_normal*ALPHA_NORMAL + num_mit*ALPHA_MITOSE)/(len(track))
        
        # initialization hypothesis
        if track.start<INIT_TH:
            Ch = np.zeros(SIZE, dtype='bool')
            Ch[[i==Nx+k for i in range(SIZE)]] = 1
            ph = Pini(track)*PTP(track, alpha, track.score())
        
            hyp.append(f'init_{k}')
            C.append(Ch)
            pho.append(ph)
        
        # false positive hypothesis
        if len(track)<FP_TH:
            Ch = np.zeros(SIZE, dtype='bool')
            Ch[[i==k or i==Nx+k for i in range(SIZE)]] = 1
            ph = PFP(track, alpha, track.score())
            
            hyp.append(f'fp_{k}')
            C.append(Ch)
            pho.append(ph)
        
        # translation and mitoses hypothesis
        if track.end>=Nf:
            yield hyp,C,pho
        
        # compute the position of mitose detections in tracklet
        dist_mitoses = np.array([i for i,det in enumerate(track) if int(det.mit)==1])
        term_hyp_prob = 0
        k2 = k
        for track2 in tracklets[k+1:]:
            k2 += 1
            
            if track2.start-track.end>TRANSP_TH:
                continue
            
            if track.start==track2.start or \
                track.end+len(track2)>Nf or \
                track.end>=track2.start:
                continue
            
            # calculate center distances
            cnt_dist = center_distances(track[-1].cx,track[-1].cy, \
                                        track2[0].cx,track2[0].cy)
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
            if len(dist_mitoses)>0:
                
                for d_mit in dist_mitoses:
                    # mitoses distance in frames to the track2
                    if abs(d_mit+track.start-track2.start)>MIT_TH:
                        continue
                    
                    # calculate center distances between tracklets
                    cnt_dist = center_distances(track[d_mit].cx,track[d_mit].cy, \
                                                track2[0].cx,track2[0].cy)
                    if cnt_dist>CENTER_MIT_TH:
                        continue
                        
                    Ch = np.zeros(SIZE, dtype='bool')
                    Ch[[i==k or i==Nx+k2 for i in range(SIZE)]] = 1
                    Ch[2*Nx+track2.start] = 1
                    ph = Pmit(cnt_dist, d_mit)
                    
                    hyp.append(f'mit_{k},{k2}')
                    C.append(Ch)
                    pho.append(ph)
                    
                    term_hyp_prob = max(term_hyp_prob, ph)
                    
        # termination hypothesis
        Ch = np.zeros(SIZE, dtype='bool')
        Ch[[i==k for i in range(SIZE)]] = 1
        ph = max(1 - term_hyp_prob, Pterm(track, Nf))
        
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

#%% adjust the tracklets to the frames

@njit(parallel=True)
def _build_costs(vals0, vals1):
    # Hungarian algorithm to join tracklets
    
    costs = np.zeros((len(vals0), len(vals1)))
    for j in range(costs.shape[0]):
        for k in range(costs.shape[1]):
            s0,e0,cx0,cy0 = vals0[j]
            s1,e1,cx1,cy1 = vals1[k]
            cdist = center_distances(cx0,cy0, cx1,cy1)
            if e0<s1 and cdist < CENTER_TH:
                costs[j,k] = cdist * (e1-s0+1)
            else:
                costs[j,k] = 1
    return costs

def _adjust_tracklets(tracklets, hyphotesis, merge_term=False):
    
    # sort hyphotesis based on the tracklet position
    hyphotesis = sorted(hyphotesis, key=lambda x:int(x.split('_')[-1].split(',')[0]))
    
    # solve initial tracklets and store the other hyphotesis
    final_tracklets = {}
    transl,mit,term = [],[],[]
    for hyp in hyphotesis:
        # verify the hyphotesis name and its values
        mode,idxs = hyp.split('_')
        if 'init' in mode:
            final_tracklets[int(idxs)] = tracklets[int(idxs)]
        elif 'transl' in mode:
            idx1,idx2 = idxs.split(',')
            transl.append([int(idx1),int(idx2)])
        elif 'mit' in mode:
            idx1,idx2 = idxs.split(',')
            final_tracklets[int(idx2)] = tracklets[int(idx2)]
            mit.append([int(idx1),int(idx2)])
        elif 'term' in mode:
            term.append(int(idxs))
            
    # add mitoses tracklets
    for idx1,idx2 in mit:
        final_tracklets[idx2] = tracklets[idx2]
            
    # add translation tracklets
    for idx1,idx2 in transl:
        if idx1 in final_tracklets:
            final_tracklets[idx1].join(tracklets[idx2])
            final_tracklets[idx2] = final_tracklets.pop(idx1)
        elif not merge_term:
            final_tracklets[idx2] = tracklets[idx2]
        elif merge_term:
            term.append(idx2)
            
    # get only termination tracklets out of the final tracklets
    term = [t for t in term \
               if not ((t in transl) or (t in mit) or (t in final_tracklets))]
            
    # adjust termination tracklets
    if not merge_term:
        # add termination tracklet to final tracklets
        for idx in term:
            final_tracklets[idx] = tracklets[idx]
    
    elif merge_term and len(term)>0:
        # aux function to get tracklets for idx
        t = lambda idx:tracklets[idx]
        
        # get the necessary values of tracklets (start, end, center x,y)
        term_vals = [[t(idx).start, t(idx).end, t(idx)[-1].cx, t(idx)[-1].cy] \
                       for idx in term]
        
        final_idx  = [int(idx) for idx in final_tracklets]
        final_vals = [[t(idx).start, t(idx).end, t(idx)[-1].cx, t(idx)[-1].cy] \
                        for idx in final_idx]
        
        # join the termination tracklets to the other tracklets based on
        # minimal assignment with the hungarian algorithm
        costs = _build_costs(np.float32(final_vals), np.float32(term_vals))
        row_ind, col_ind = linear_sum_assignment(costs)
        
        for row,col in zip(row_ind, col_ind):
            if costs[row][col]<1:
                final_tracklets[final_idx[row]].join(tracklets[term[col]])
                final_tracklets[term[col]] = final_tracklets.pop(final_idx[row])
            else:
                final_tracklets[term[col]] = tracklets[term[col]]
                
        # add not joined termination hyphotesis
        col_ind = [i for i in range(len(term_vals)) if not i in col_ind]
        for col in col_ind:
            final_tracklets[term[col]] = tracklets[term[col]]
            
    return list(final_tracklets.values())

#%% solve integer optimization problem

def solve_tracklets(tracklets:list, Nf:int, squeeze_factor:float=0.8, max_iterations:int=100) -> list:
    '''
    
    Solve the tracklets using the Bise et al algorithm.

    Parameters
    ----------
    tracklets : list
        List of Tracklets.
    Nf : int
        Total number of frames.
    squeeze_factor : float, optional
        Value to squeeze the INIT_TH, FP_TH after each iteration. The default is 0.8.
    max_iterations : int, optional
        Maximum number of iterations. The default is 100.

    Returns
    -------
    list
        List of solved Tracklets.

    '''
    
    global INIT_TH, FP_TH, TRANSP_TH, CENTER_TH, MIT_TH, CENTER_MIT_TH
    
    last_track_size = len(tracklets)
    for it in range(max_iterations):
        print(f'Iteration {it+1}/{max_iterations}:')
        
        # get hypothesis matrixes
        C, pho, hyp = _get_C_pho_matrixes(np.copy(tracklets), Nf)
            
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
        adj_tracklets = _adjust_tracklets(np.copy(tracklets), hyp)
        
        # verify if to stop iterations
        current_track_size = len(adj_tracklets)
        if current_track_size==last_track_size:
            tracklets = _adjust_tracklets(np.copy(tracklets), hyp, merge_term=True)
            print('Early stop.')
            break
        # update variables for next iterations
        last_track_size = current_track_size
        tracklets = np.copy(adj_tracklets)
        print()
        
        # squeeze the threshold values for the next iteration
        INIT_TH *= squeeze_factor
        FP_TH *= squeeze_factor
        # TRANSP_TH *= squeeze_factor
        # CENTER_TH *= squeeze_factor
        # MIT_TH *= squeeze_factor
        # CENTER_MIT_TH *= squeeze_factor
            
    return list(tracklets)







