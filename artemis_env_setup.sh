#!/bin/bash
module load python/3.8.2 
virtualenv --system-site-packages ~/venv

source ~/venv/bin/activate
pip install timm
pip install einops