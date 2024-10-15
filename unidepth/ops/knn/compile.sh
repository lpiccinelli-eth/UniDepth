#!/usr/bin/env bash

export TORCH_CUDA_ARCH_LIST="6.1 7.0 7.5 8.0 8.6 8.7 8.9 9.0+PTX" 
# export FORCE_CUDA=1 #if you do not actually have cuda, workaround
python setup.py build install