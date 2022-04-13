# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import os
import sys
import tensorflow as tf
import time
import cv2
import pickle
import numpy as np
import argparse
from tqdm import tqdm

from importlib import import_module
model_name = sys.argv[-1]

if 'probiou' in model_name:
    cfgs = import_module(f'libs.configs.UFRGS_CELL.{model_name}_step2')
else:
    cfgs = import_module(f'libs.configs.UFRGS_CELL.{model_name}')

if model_name=='csl':
    DetectionNetwork = import_module(f'libs.models.detectors.{model_name}.build_whole_network').DetectionNetworkCSL
elif model_name=='dcl':
    DetectionNetwork = import_module(f'libs.models.detectors.{model_name}.build_whole_network').DetectionNetworkDCL
elif model_name=='r3det':
    DetectionNetwork = import_module(f'libs.models.detectors.{model_name}.build_whole_network').DetectionNetworkR3Det
elif model_name=='retinanet':
    DetectionNetwork = import_module(f'libs.models.detectors.{model_name}.build_whole_network').DetectionNetworkRetinaNet
elif model_name=='rsdet':
    DetectionNetwork = import_module(f'libs.models.detectors.{model_name}.build_whole_network_5p').DetectionNetworkRSDet
elif model_name=='refine_retinanet':
    DetectionNetwork = import_module(f'libs.models.detectors.{model_name}.build_whole_network').DetectionNetworkRefineRetinaNet
elif model_name=='r3det_dcl':
    DetectionNetwork = import_module(f'libs.models.detectors.{model_name}.build_whole_network').DetectionNetworkR3DetDCL
elif model_name=='r2cnn':
    DetectionNetwork = import_module(f'libs.models.detectors.{model_name}.build_whole_network').DetectionNetworkR2CNN
elif model_name=='retinanet_probiou' or model_name=='retinanet_gwd':
    DetectionNetwork = import_module(f'libs.models.detectors.gwd.build_whole_network').DetectionNetworkGWD
elif model_name=='r3det_gwd' or model_name=='r3det_probiou':
    DetectionNetwork = import_module(f'libs.models.detectors.r3det_gwd.build_whole_network').DetectionNetworkR3DetGWD
    

from libs.val_libs.voc_eval_r import EVAL
from tools.test_ufrgscell_base import TestUFRGSCELL

class TestUFRGSCELLGWD(TestUFRGSCELL):

    def eval(self):
        dcl = DetectionNetwork(cfgs=self.cfgs, is_training=False)

        all_boxes_r = self.eval_with_plac(img_dir=self.args.img_dir, det_net=dcl,
                                          image_ext=self.args.image_ext)
        all_boxes_1class = []
        for boxes in all_boxes_r:
            all_boxes_1class.append(np.array([[1,*b[1:]] for b in boxes]))

        imgs = os.listdir(self.args.img_dir)
        
        real_test_imgname_list = [i.split(self.args.image_ext)[0] for i in imgs]

        print(10 * "**")
        print('rotation eval:')
        evaler = EVAL(self.cfgs)
        evaler.voc_evaluate_detections(all_boxes=all_boxes_1class, test_imgid_list=real_test_imgname_list,
                                       test_annotation_path=self.args.test_annotation_path)


if __name__ == '__main__':
    
    cfgs.DATASET_NAME = 'UFRGS_CELL_1class'
    tester = TestUFRGSCELLGWD(cfgs)
    tester.eval()
