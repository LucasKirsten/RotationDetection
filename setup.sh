#!/bin/bash

cd ./libs/utils/cython_utils
rm *.so
rm *.c
rm *.cpp
python setup.py build_ext --inplace

cd ./libs/utils/
rm *.so
rm *.c
rm *.cpp
python setup.py build_ext --inplace