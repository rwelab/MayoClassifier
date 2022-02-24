**Mayo Classifier**


This repository provides the code for creating classification models for clinical text based on mayo scores. <br />
For a more detailed reference please refer to our paper "Accurate, Robust, and Scalable Machine Abstraction of Mayo Endoscopic Subscores from Colonoscopy Reports". <br /> 
This project is done by Real World Evidence-Lab.

**Files**

Classifier.py creates and evaluvates automl models
ShortScript.py creates and evaluvates automl models with least number of line of codes

**Download**

The autoML package that is cutomised for the current study is from AutoGluon <br />

AutoGluon enables easy-to-use and easy-to-extend AutoML with a focus on automated stack ensembling, deep learning, and real-world applications spanning text, image, and tabular data <br />

More details can be found at https://auto.gluon.ai/stable/index.html


**Installation**

1. GPU Version on Linux system with PIP

_python3 -m pip install -U pip <br />
python3 -m pip install -U setuptools wheel <br />_

*Here we assume CUDA 10.1 is installed.  You should change the number <br />
*according to your own CUDA version (e.g. mxnet_cu100 for CUDA 10.0). <br /> <br />
_python3 -m pip install -U "mxnet_cu101<2.0.0" <br />
python3 -m pip install autogluon <br />_

2. CPU Version on Linux system with PIP

_python3 -m pip install -U pip <br />
python3 -m pip install -U setuptools wheel <br />
python3 -m pip install -U "mxnet<2.0.0" <br />
python3 -m pip install autogluon <br />_


3. GPU Version on Linux system from Source

_python3 -m pip install -U pip <br />
python3 -m pip install -U setuptools wheel <br />_

*Here we assume CUDA 10.1 is installed.  You should change the number <br />
*according to your own CUDA version (e.g. mxnet_cu102 for CUDA 10.2). <br /> <br />
_python3 -m pip install -U "mxnet_cu101<2.0.0" <br />
git clone https://github.com/awslabs/autogluon <br />
cd autogluon && ./full_install.sh <br />_
