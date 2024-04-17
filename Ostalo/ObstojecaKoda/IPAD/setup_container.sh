#!/usr/bin/env bash

# Python, wget, nano
apt-get update
apt-get install -y software-properties-common wget nano
add-apt-repository ppa:deadsnakes/ppa
apt-get install -y python3.7 python3.7-tk python3.7-distutils python3-apt
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
update-alternatives --config python3
apt-get install -y python3-pip libsm6 libxrender1 libfontconfig1
python3.7 -m pip install --upgrade pip

# Required python packages
python3.7 -m pip install joblib tqdm matplotlib numpy pillow scikit-learn scipy visdom jsonpatch torchsummary opencv-python==4.1.1.26
