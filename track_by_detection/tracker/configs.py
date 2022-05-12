# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 20:14:23 2022

@author: kirstenl
"""
#%% values to be set by the user

DEBUG = False

DATASET  = 'migration_samlai'
DETECTOR = 'r2cnn' # r2cnn OR r3det

# value to filter individual detections based on minimun score
# for normal detections
NORMAL_SCORE_TH = 0.3
# for mitoses detection
MIT_SCORE_TH = 0.1

# values to filter tracklets based on the number of detections and score
# minimun score to validate a tracklet as valid (score for tracklets are the
# mean score value for all detections)
TRACK_SCORE_TH = 0.3
# minimun size of the tracklet to have at least SCORE_TH to be valid
TRACK_SIZE_TH  = 10

# threshold to use in order to join detections on NSM algorithm
NMS_TH = 0.5

#%% values to compute the bise et.al algorithm

# thresholds for the tracker

# frames to be consider the initial state
# higher values means that more cells can be consider as the starters
INIT_TH = 10

# distance between frames to consider transposition
# higher values means that higher gaps allows to join detections
TRANSP_TH = 20

# distance between cell centers in pixels
# higher values means that cells far appart can be joined in tracklets
CENTER_TH = 100

# distance between frames to consider mitoses
# higher values means that higher gaps allows to consider a mitoses event
MIT_TH = 20

# distance between cell centers in pixels for mitoses
CENTER_MIT_TH = 50

# values to adjust the probabilities distributions
# higher values means larger probabilites
INIT_FACTOR = 5
LINK_FACTOR = 300
MIT_FACTOR  = 300

#%% values that were calculated previously

# alpha values for the networks (based on their P50 values)
# add values for each detector
ALPHAS = {
    'r2cnn':(0.3251,0.4721),
    'r3det':(0.7207,0.8806)
}

ALPHA_NORMAL = ALPHAS[DETECTOR][0]
ALPHA_MITOSE = ALPHAS[DETECTOR][1]




