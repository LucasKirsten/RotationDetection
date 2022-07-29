# -*- coding: utf-8 -*-
"""
Define global configuration values

@author: Lucas N. Kirsten (lnkirsten@inf.ufrgs.br)
"""

#%% values to be set by the user

DEBUG = True

DATASET  = 'Fluo-N2DL-HeLa'
LINEAGE  = '01'
FRAME_SHAPE = (700,1100)
DETECTOR = 'r2cnn'

path_imgs = f'./frames/{DATASET}/{LINEAGE}/images'
path_dets = f'./frames/{DATASET}/{LINEAGE}/{DETECTOR}'
path_gt   = f'./frames/{DATASET}/{LINEAGE}_GT/TRA'

# value to filter individual detections based on minimun score
# for normal detections
NORMAL_SCORE_TH = 0.5
# for mitoses detection
MIT_SCORE_TH = 0.5

# values to filter tracklets based on the number of detections and score
# minimun score to validate a tracklet as valid (score for tracklets are the
# mean score value for all detections in it)
TRACK_SCORE_TH = 0.5
# minimun size of the tracklet to have at least SCORE_TH to be valid
TRACK_SIZE_TH  = 5

# threshold to use in order to join detections on NMS algorithm
NMS_TH = 0.5

#%% values to compute the bise et.al algorithm

# thresholds for the tracker

# maximal number of detections to a frame be a possible false positive
# higher values means that longer tracklets can be consider false positives
FP_TH = 5

# distance between frames to consider transposition
# higher values means that higher gaps allows to join detections
TRANSP_TH = 10

# distance between frames to consider mitoses
# higher values means that allows higher gaps to consider a mitoses event
MIT_TH = 3

# distance between cell centers in pixels to consider joining
# higher values means that cells far appart can be joined in tracklets
CENTER_TH = 0.05*(FRAME_SHAPE[0]**2+FRAME_SHAPE[1]**2)**(1/2)

# distance between cell centers in pixels for mitoses
CENTER_MIT_TH = 0.05*(FRAME_SHAPE[0]**2+FRAME_SHAPE[1]**2)**(1/2)

# values to adjust the probabilities distributions
# higher values means larger probabilites
INIT_FACTOR = 30
LINK_FACTOR = 1000
MIT_FACTOR  = 1000

#%% values that were calculated previously

# alpha values for the networks (based on their P50 values)
# add values for each detector (normal, mitose)

ALPHAS = {
    'glioblastoma': {'':{
        'dcl':      (0.7181,0.6335),
        'csl':      (0.7242,0.5398),
        'rsdet':    (0.7433,0.6360),
        'retinanet':(0.7525,0.5647),
        'r3det':    (0.7515,0.6675),
        'r3detdcl': (0.7550,0.6263),
        'r2cnn':    (0.7733,0.6965)
        }},
    
    'Fluo-N2DH-GOWT1': {
        '01':{
            'r2cnn': (0.8993,0.8993)},
        '02':{
            'r2cnn': (0.8644,0.8644)},
        },
        
     'Fluo-N2DL-HeLa': {
        '01':{
            'r2cnn': (0.912,0.912)},
        '02':{
            'r2cnn': (0.880,0.880)},
        },
    
    'PhC-C2DH-U373': {
        '01':{
            'r2cnn': (0.7673,0.7673)},
        '02':{
            'r2cnn': (0.6867,0.6867)}
        },
    
    'PhC-C2DL-PSC': {
        '01':{
            ()},
        '02':{
            'r2cnn': (0.865,0.865)}
        },
    'Fluo-N2DH-SIM+':{
        '01':{
            'r2cnn':(0.,0.)},
        '02':{
            'r2cnn':(0.895,0.895)}}
    }

# map values according to the alpha values above
ALPHA_NORMAL = ALPHAS[DATASET][LINEAGE][DETECTOR][0]
ALPHA_MITOSE = ALPHAS[DATASET][LINEAGE][DETECTOR][1]
