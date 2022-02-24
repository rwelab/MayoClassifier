**Mayo Classifer**


This repository provides the code for creating classification models for clinical text based on mayo scores.
For a more detailed reference please refer to our paper "Accurate, Robust, and Scalable Machine Abstraction of Mayo Endoscopic Subscores from Colonoscopy Reports". 
This project is done by Real World Evidence-Lab.

Download

The autoML package that is cutomised for the current study is from AutoGluon

AutoGluon enables easy-to-use and easy-to-extend AutoML with a focus on automated stack ensembling, deep learning, and real-world applications spanning text, image, and tabular data

More details can be found at https://auto.gluon.ai/stable/index.html


Installation

1. GPU Version on Linux system with PIP

python3 -m pip install -U pip
python3 -m pip install -U setuptools wheel

# Here we assume CUDA 10.1 is installed.  You should change the number
# according to your own CUDA version (e.g. mxnet_cu100 for CUDA 10.0).
python3 -m pip install -U "mxnet_cu101<2.0.0"
python3 -m pip install autogluon


2. CPU Version on Linux system with PIP

python3 -m pip install -U pip
python3 -m pip install -U setuptools wheel
python3 -m pip install -U "mxnet<2.0.0"
python3 -m pip install autogluon


3. GPU Version on Linux system from Source

python3 -m pip install -U pip
python3 -m pip install -U setuptools wheel

# Here we assume CUDA 10.1 is installed.  You should change the number
# according to your own CUDA version (e.g. mxnet_cu102 for CUDA 10.2).
python3 -m pip install -U "mxnet_cu101<2.0.0"
git clone https://github.com/awslabs/autogluon
cd autogluon && ./full_install.sh
