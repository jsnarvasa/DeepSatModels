#!/bin/bash

# This script is used to setup the GCP environment for machine learning debugging

# Setup conda environment
conda update -n base -c conda-forge conda --yes
conda create -n tsvit python=3.9 --yes
conda activate tsvit

# Downloading PASTIS24 dataset
pip install gdown
gdown 1Av9hou8DviCJsEB9a_XU9SyqTuNxVpdE
unzip -q PASTIS24.zip
rm -rf PASTIS24.zip

# Install required libraries
pip3 install torch torchvision torchaudio
pip install timm
pip install einops
pip install tensorboard
pip install pandas
pip install scikit-image
pip install matplotlib
pip install PyYAML==5.4.1
pip install scikit-learn