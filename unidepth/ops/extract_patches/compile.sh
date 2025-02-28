#!/usr/bin/env bash

if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then
    export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX"
fi

python setup.py build install
