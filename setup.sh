#!/bin/bash

cd ./libs/utils/cython_utils
python setup.py build_ext --inplace

cd ./libs/utils/
python setup.py build_ext --inplace