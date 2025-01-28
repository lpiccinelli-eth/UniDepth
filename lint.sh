#!/bin/bash

python3 -m black unidepth
python3 -m black scripts
python3 -m isort unidepth
python3 -m isort scripts
