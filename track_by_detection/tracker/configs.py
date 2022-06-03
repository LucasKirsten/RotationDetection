# -*- coding: utf-8 -*-
"""
Define global configuration values

@author: Lucas N. Kirsten (lnkirsten@inf.ufrgs.br)
"""
#%% values to be set by the user

DEBUG = False

DATASET  = 'migration_samlai'
DETECTOR = 'r2cnn'

path_imgs = f'./frames/{DATASET}/frames'
path_dets = f'./frames/{DATASET}/{DETECTOR}'
path_gt   = f'./frames/{DATASET}/migration.csv'

# value to filter individual detections based on minimun score
# for normal detections
NORMAL_SCORE_TH = 0.
# for mitoses detection
MIT_SCORE_TH = 0.

# values to filter tracklets based on the number of detections and score
# minimun score to validate a tracklet as valid (score for tracklets are the
# mean score value for all detections)
TRACK_SCORE_TH = 0
# minimun size of the tracklet to have at least SCORE_TH to be valid
TRACK_SIZE_TH  = 0

# threshold to use in order to join detections on NMS algorithm
NMS_TH = 0.5

#%% values to compute the bise et.al algorithm

# thresholds for the tracker

# frames to be consider the initial state
# higher values means that more cells can be consider as the starters
INIT_TH = 10

# maximal number of detections to a frame be a possible false positive
# higher values means that longer tracklets can be consider false positives
FP_TH = 5

# distance between frames to consider transposition
# higher values means that higher gaps allows to join detections
TRANSP_TH = 10

# distance between frames to consider mitoses
# higher values means that allows higher gaps to consider a mitoses event
MIT_TH = 10

# distance between cell centers in pixels to consider joining
# higher values means that cells far appart can be joined in tracklets
CENTER_TH = 30

# distance between cell centers in pixels for mitoses
CENTER_MIT_TH = 25

# values to adjust the probabilities distributions
# higher values means larger probabilites
INIT_FACTOR = 20
LINK_FACTOR = 500
MIT_FACTOR  = 500

#%% values that were calculated previously

# alpha values for the networks (based on their P50 values)
# add values for each detector (normal, mitose)
ALPHAS = {
    'dcl':      (0.7181,0.6335),
    'csl':      (0.7242,0.5398),
    'rsdet':    (0.7433,0.6360),
    'retinanet':(0.7525,0.5647),
    'r3det':    (0.7515,0.6675),
    'r3detdcl': (0.7550,0.6263),
    'r2cnn':    (0.7733,0.6965)
}

# map values according to the alpha values above
ALPHA_NORMAL = ALPHAS[DETECTOR][0]
ALPHA_MITOSE = ALPHAS[DETECTOR][1]




