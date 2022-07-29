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
from .classes import Tracklet

#%% make hyphotesis
def _make_hypotheses(tracklets, Nf, mitoses):
    
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
        
        # determine the mitoses detections distances
        dist_mitoses = np.array([i+1 for i,det in enumerate(track[1:]) \
                                 if int(det.mit)==1])
        
        k2 = k
        for track2 in tracklets[k+1:]:
            k2 += 1
            
            # translation hyphotesis
            # if tracklets begins togheter or is not possible to join
            if track.end<track2.start or track.end+len(track2)<=Nf:
            
                # calculate center distances
                cnt_dist = center_distances(track[-1].cx, track[-1].cy,
                                            track2[0].cx, track2[0].cy)
                
                # verify hyphotesis
                if track2.start-track.end<TRANSP_TH and \
                    cnt_dist<CENTER_TH and track.end<track2.start:
                    
                    # translation hypothesis
                    Ch = np.zeros(SIZE, dtype='bool')
                    Ch[[i==k or i==Nx+k2 for i in range(SIZE)]] = 1
                    ph = Plink(track, track2, cnt_dist)
                    
                    hyp.append(f'transl_{track.idx},{track2.idx}')
                    C.append(Ch)
                    pho.append(ph)
                
            # iterate over the mitoses events
            for d_mit in dist_mitoses:
                    
                # if tracklets begins togheter or is above gap threshold
                if track.start+d_mit>track2.start or track2.start-track.start-d_mit>MIT_TH:
                    continue
                
                # calculate center distances
                cnt_dist = center_distances(track[d_mit].cx, track[d_mit].cy,
                                            track2[0].cx, track2[0].cy)
                if cnt_dist>CENTER_MIT_TH:
                    continue
                
                # mitoses hypothesis
                Ch = np.zeros(SIZE, dtype='bool')
                Ch[[i==k or i==Nx+k2 for i in range(SIZE)]] = 1
                ph_mit = Pmit(cnt_dist, d_mit)
                
                hyp.append(f'mit_{track.idx}-{d_mit},{track2.idx}')
                C.append(Ch)
                pho.append(ph_mit)
                
                # partial false positive
                Ch = np.zeros(SIZE, dtype='bool')
                Ch[[i==k or i==Nx+k2 for i in range(SIZE)]] = 1
                ph_fp = PFP(track, ALPHA_NORMAL, start=d_mit)
                ph_fp *= PTP(track2, ALPHA_NORMAL)
                
                hyp.append(f'fp_{track.idx}-{d_mit},{track2.idx}')
                C.append(Ch)
                pho.append(ph_fp)
                    
            # if the final detection is a mitoses
            if int(track[-1].mit)==1:
                k3 = k2
                for track3 in tracklets[k2+1:]:
                    
                    # if tracklets begins togheter or is above gap threshold
                    if track.end>=track2.start or track.end>=track3.start:
                        continue
                    
                    if track2.start-track.end>MIT_TH or track3.start-track.end>MIT_TH:
                        continue
                    
                    cnt_dist2 = center_distances(track[-1].cx, track[-1].cy,
                                                 track2[0].cx, track2[0].cy)
                    if cnt_dist2>CENTER_MIT_TH:
                        continue
                    cnt_dist3 = center_distances(track[-1].cx, track[-1].cy,
                                                 track3[0].cx, track3[0].cy)
                    if cnt_dist3>CENTER_MIT_TH:
                        continue
                    cnt_dist23 = center_distances(track2[0].cx, track2[0].cy,
                                                  track3[0].cx, track3[0].cy)
                    if cnt_dist23>CENTER_MIT_TH:
                        continue
                    
                    # mitoses hypothesis
                    Ch = np.zeros(SIZE, dtype='bool')
                    Ch[[i==k or i==Nx+k2 or i==Nx+k3 for i in range(SIZE)]] = 1
                    d_mit = (track2.start-track.end + track3.start-track.end)/2-1
                    cnt_dist = cnt_dist23*(cnt_dist2+cnt_dist3)/2
                    ph_mit = Pmit(cnt_dist, d_mit)
                    
                    hyp.append(f'mit_{track.idx},{track2.idx},{track3.idx}')
                    C.append(Ch)
                    pho.append(ph_mit)
                    
        # completeness hypothesis
        Ch = np.zeros(SIZE, dtype='bool')
        Ch[[i==k or i==Nx+k for i in range(SIZE)]] = 1
        ph = Pini(track)
        if len(pho)>0:
            ph = PTP(track, alpha)*(ph+np.mean(pho))/2
        
        hyp.append(f'compl_{track.idx}')
        C.append(Ch)
        pho.append(ph)
        
        # false positive hypothesis
        if track.score()<TRACK_SCORE_TH:
                
            Ch = np.zeros(SIZE, dtype='bool')
            Ch[[i==k or i==Nx+k for i in range(SIZE)]] = 1
            ph = PFP(track, alpha)
            
            hyp.append(f'fp_{track.idx}')
            C.append(Ch)
            pho.append(ph)
            
        idx = np.argmax(pho)
        print('hyp:', hyp[idx], 'pho: ', pho[idx], end=' * \n')
        return hyp,C,pho
        
    return __get_hyphoteses

#%% adjust the tracklets to the frames

def _adjust_tracklets(tracklets, hyphotesis, add_parent=False):
    
    # sort hyphotesis based on the tracklet position
    hyphotesis = sorted(hyphotesis,\
                        key=lambda x:int(x.split('_')[-1].split(',')[0].split('-')[0]))
    print()
    print(hyphotesis)
    
    # make list of tracklets into a dict of their indexes
    tracklets = {int(track.idx):track for track in tracklets}
    
    # adjust tracklets for each hyphotesis
    for hyp in hyphotesis:
        
        # verify the hyphotesis name and its values
        mode,idxs = hyp.split('_')
        
        if 'fp' in mode:
            indexes = idxs.split(',')
            if len(indexes)>1:
                idx1,idx2 = idxs.split(',')
                idx1,d_split = idx1.split('-')
                
                track = tracklets.pop(int(idx1))
                track = track.split(int(d_split))
                track.join(tracklets[int(idx2)])
                track.set_idx(int(idx2))
                tracklets[int(idx2)] = track
                
            else:
                tracklets.pop(int(idxs))
                
        elif 'transl' in mode:
            idx1,idx2 = idxs.split(',')
            track = tracklets.pop(int(idx1))
            track.join(tracklets[int(idx2)])
            track.set_idx(int(idx2))
            tracklets[int(idx2)] = track
            
        elif 'mit' in mode:
            indexes = idxs.split(',')
            if len(indexes)==2:
                idx1,idx2 = idxs.split(',')
                idx1,d_mit = idx1.split('-')
                idmax = np.max(list(tracklets.keys()))+1
                trackp = tracklets.pop(int(idx1))
                trackc1 = tracklets.pop(int(idx2))
                
                trackp, trackc2 = trackp.split_mitoses(int(d_mit))
                if add_parent:
                    trackc2.parent = trackp
                    trackc1.parent = trackp
                
                trackp.set_idx(int(idx1))
                trackc1.set_idx(int(idx2))
                trackc2.set_idx(int(idmax))
                
                tracklets[trackp.idx] = trackp
                tracklets[trackc1.idx] = trackc1
                tracklets[trackc2.idx] = trackc2
                                
            else:
                idx1,idx2,idx3 = idxs.split(',')
                if add_parent:
                    tracklets[int(idx2)].parent = tracklets[int(idx1)]
                    tracklets[int(idx3)].parent = tracklets[int(idx1)]
            
    return list(tracklets.values())

#%% solve integer optimization problem

def _solve_optimization(C, pho, hyp):
    # solve integer optimization problem
    
    x = cvxpy.Variable((len(pho),1), boolean=True)
    
    # define and solve integer optimization
    constrains = [(cvxpy.transpose(C) @ x) <= 1]
    total_prob = cvxpy.sum(cvxpy.transpose(pho) @ x)
    
    if DEBUG: print('Solving integer optimization...')
    knapsack_problem = cvxpy.Problem(cvxpy.Maximize(total_prob), constrains)
    knapsack_problem.solve(solver=cvxpy.GLPK_MI)
    
    # get the true hypothesis
    x = np.squeeze(x.value).astype('int')
    hyp = np.array(hyp)[x==1]
    
    return hyp

#%% populate C and pho matrixes for hypothesis
def _get_C_pho_matrixes(tracklets, Nf, mitoses=False):
    
    Nx = len(tracklets)
    pbar = range(Nx)
    if DEBUG:
        pbar = tqdm(pbar)
        pbar.set_description('Bulding hyphotesis matrix: ')
    
    generator = _make_hypotheses(tracklets, Nf, mitoses)
    hyp,C,pho = [],[],[]
    def _populate_matrixes(i):
        h,c,p = generator(i)
        hyp.extend(h)
        C.extend(c)
        pho.extend(p)
        
    with Parallel(n_jobs=1, prefer='threads') as parallel:
        _ = parallel(delayed(_populate_matrixes)(i) for i in pbar)
    
    return np.array(C), np.float32(pho), hyp

#%%

def solve_tracklets(tracklets:list, Nf:int, max_iterations:int=100) -> list:
    '''
    
    Solve the tracklets using the Bise et al algorithm.

    Parameters
    ----------
    tracklets : list
        List of Tracklets.
    Nf : int
        Total number of frames.
    max_iterations : int, optional
        Maximum number of iterations. The default is 100.

    Returns
    -------
    list
        List of solved Tracklets.

    '''
    
    # solve iteratively
    # last_track_size = len(tracklets)
    # for it in range(max_iterations):
    #     if DEBUG: print(f'Iteration {it+1}/{max_iterations}:')
        
    #     # get hypothesis matrixes
    #     C, pho, hyp = _get_C_pho_matrixes(np.copy(tracklets), Nf)
            
    #     # solve integer optimization
    #     hyp = _solve_optimization(C, pho, hyp)
        
    #     if DEBUG: print('Adjusting final tracklets...')
    #     adj_tracklets = _adjust_tracklets(np.copy(tracklets), hyp)
        
    #     # verify if to stop iterations
    #     current_track_size = len(adj_tracklets)
    #     if current_track_size==last_track_size:
    #         tracklets = np.copy(adj_tracklets)
    #         if DEBUG: print('Early stop.')
    #         break
    #     # update variables for next iterations
    #     last_track_size = current_track_size
    #     tracklets = np.copy(adj_tracklets)
    #     if DEBUG: print()
        
    # get hypothesis matrixes
    if DEBUG: print('Solving for mitoses...')
    C, pho, hyp = _get_C_pho_matrixes(np.copy(tracklets), Nf, mitoses=True)
    print()
    print(list(zip(hyp,pho)))
    hyp = _solve_optimization(C, pho, hyp)
    tracklets = _adjust_tracklets(np.copy(tracklets), hyp, add_parent=True)
        
    # # set idx values
    # tracklets = [t for t in tracklets if len(t)>0]
    # tracklets = sorted(tracklets, key=lambda x:x.start)
    # for i,track in enumerate(tracklets):
    #     track.set_idx(i+1)
        
    return tracklets






