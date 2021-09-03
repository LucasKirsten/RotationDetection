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

cfgs = import_module(f'libs.configs.UFRGS_CELL.{model_name}')
if model_name=='csl':
    DetectionNetwork = import_module(f'libs.models.detectors.{model_name}.build_whole_network.DetectionNetworkCSL')
elif model_name=='dcl':
    DetectionNetwork = import_module(f'libs.models.detectors.{model_name}.build_whole_network.DetectionNetworkDCL')
elif model_name=='r3det':
    DetectionNetwork = import_module(f'libs.models.detectors.{model_name}.build_whole_network.DetectionNetworkR3Det')
elif model_name=='retinanet':
    DetectionNetwork = import_module(f'libs.models.detectors.{model_name}.build_whole_network.DetectionNetworkRetinaNet')

from libs.val_libs.voc_eval_r import EVAL
from tools.test_ufrgscell_base import TestUFRGSCELL

class TestUFRGSCELLGWD(TestUFRGSCELL):

    def eval(self):
        dcl = DetectionNetwork(cfgs=self.cfgs, is_training=False)

        all_boxes_r = self.eval_with_plac(img_dir=self.args.img_dir, det_net=dcl,
                                          image_ext=self.args.image_ext)

        imgs = os.listdir(self.args.img_dir)
        
        real_test_imgname_list = [i.split(self.args.image_ext)[0] for i in imgs]

        print(10 * "**")
        print('rotation eval:')
        evaler = EVAL(self.cfgs)
        evaler.write_voc_results_file(all_boxes=all_boxes_r,
                                      test_imgid_list=real_test_imgname_list,
                                      det_save_dir=f'./{model_name}_results')


if __name__ == '__main__':

    tester = TestUFRGSCELLGWD(cfgs)
    tester.eval()
