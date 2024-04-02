#!/bin/bash
NAME=${1}
VENV_DIR=${2}

python -m venv ${VENV_DIR}/${NAME}

source ${VENV_DIR}/${NAME}/bin/activate

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install -e .
pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu118
