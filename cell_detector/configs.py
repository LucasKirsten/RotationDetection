import os
import numpy as np

# gpu setting
GPU_GROUP = "0"
NUM_GPUS = len(GPU_GROUP.split(','))

# log print
SHOW_TRAIN_INFO_INTE = 20
SMRY_ITER = 200
ADD_BOX_IN_TENSORBOARD = True

# learning policy
BATCH_SIZE = 4
EPSILON = 1e-5
MOMENTUM = 0.9
LR = 5e-4
WEIGHT_DECAY = 1e-4
DECAY_EPOCH = [12, 16, 20]
WARM_EPOCH = 1.0 / 4.0

MUTILPY_BIAS_GRADIENT = 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = 10.0  # if None, will not clip

# schedule
GPU_GROUP = '0'
NUM_GPU = len(GPU_GROUP.strip().split(','))
SAVE_WEIGHTS_INTE = 10000
MAX_EPOCH = 20
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = (SAVE_WEIGHTS_INTE * MAX_EPOCH) // BATCH_SIZE
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'UFRGS_CELL'
IMG_SHORT_SIDE_LEN = 512
IMG_MAX_LENGTH = 512
IMG_SHAPE = (512,512,3)
PATH_TFRECORDS = '/datasets/tfrecord/UFRGS_CELL_train'
CLASSES = 2
PIXEL_MEAN = [108.92795, 108.92795, 108.92795]
PIXEL_STD = [7.9574094, 7.9574094, 7.9574094]

# data augmentation
IMG_ROTATE = True
RGB2GRAY = True
VERTICAL_FLIP = True
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = False

# model
#pretrain_zoo = PretrainModelZoo()
#PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
#TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')
REGRESSIONS = 5

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0
ANGLE_WEIGHT = 0.5

# DCL
OMEGA = 180 / 256.
ANGLE_MODE = 1  # {0: BCL, 1: GCL}

VERSION = 'DLCv3_UFRGS_CELL_smooth_l1_loss'