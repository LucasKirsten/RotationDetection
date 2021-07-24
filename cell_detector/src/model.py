import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL

import configs as cfgs
from losses import *
        
def Model():
    input_img = KL.Input((None,None,3), name='input')
    backbone = tf.keras.applications.ResNet50(include_top=False, input_tensor=input_img)
    
    feats = backbone(input_img)
    feats = KL.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', name='upsample_feats') (feats)
    feats = KL.BatchNormalization(name='norm_feats') (feats)
    feats = KL.Activation('relu', name='relu_feats') (feats)
    # skip connection
    skip = KL.Conv2D(64, (1,1), padding='same', name='skip') (backbone.get_layer('conv4_block6_out').output)
    feats = KL.Add(name='skip_feats') ([feats, skip])
    
    classes = KL.Conv2D(cfgs.CLASS_NUM, (1,1), padding='same', activation='sigmoid', name='classification') (feats)
    regress = KL.Conv2D(cfgs.REGRESSIONS, (1,1), padding='same', activation='sigmoid', name='regression') (feats)
    
    model = tf.keras.Model(input_img, [classes, regress], name='cell_detector')
    model.compile(tf.keras.optimizers.Adam(), \
                  loss={'classification':focal_loss(), 'regression':smoothl1()})
    
    return model
    