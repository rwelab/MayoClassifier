# MayoClassifier

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Description

Welcome to the Mayo Classifier repository! 

This repository provides the code for creating classification models for clinical text based on mayo scores. 
For a more detailed reference please refer to our paper "Accurate, Robust, and Scalable Machine Abstraction of Mayo Endoscopic Subscores from Colonoscopy Reports". 
This project is done by Real World Evidence-Lab.

 **Introduction**: 

Question: Is natural language processing (NLP)a viable alternative to manuallyabstractingdisease activity from notes?
Findings: Across a range of metrics, automated machine learning (autoML) was the best method for training text classifiers. AutoML classifiers accurately assigned Ulcerative colitis Mayo subscores to colonoscopy reports (97%), recognized when to abstain, generalized well tootherhealth systems, required limited effort for annotation and programming, demonstratedfairness,and had a smallcarbon footprint.
Meaning:NLP methods likeautoML appear to be sufficiently mature technologies for clinicaltext classification,and thus are poised to enable manydownstream endeavors using electronic health records data.

Importance: Electronic health records (EHR) data are growing in importance as a source of evidence on real-world treatment effects. However, EHRs typically do not directly capture quantitative measures of disease activity from clinicians, limiting their utility for research and quality improvement. Although disease activity scores are often manually abstractable from clinical notes, this process is expensive and subject to variability. Natural language processing (NLP)isa scalable alternative but has historically been subject to multiple limitations including insufficient accuracy, data hunger, technical complexity, poor generalizability, algorithmic unfairness, and an outsized carbon footprint.

**Features**: 
Objective: Develop and comparecomputational methods for classifying colonoscopyreports according to their Ulcerative Colitis Mayo endoscopic subscores and recognizing when to abstain.
Design:Other observational study –NLP algorithm development and validation
Setting: Academic medical center (UCSF) and safety-net hospital(ZSFG) in California
Participants: Patients with Ulcerative colitis
Exposures:Colonoscopy
Main Outcomes and Measures:The primary outcome was accuracy in identifying reports suitable forMayo subscoring (binary) and assigning a Mayo subscore where relevant (ordinal). Secondary outcomesincludedlearning efficiency from training data, generalizability, computational costs, fairness, and sustainability.
Results:Using automated machine learning (autoML) we trained a pair of classifiersthat were 98% accurate at determining which reports to score and 97% accurate at assigning the correct Mayo endoscopic subscore. Binary classification algorithms trainedo n UCSF data achieved 96% accuracy on hold-out test data from ZSFG. 
Conclusions and Relevance:We identified autoML as an efficient method for training algorithms to perform clinical text classification accurately. The autoML training procedure demonstrates many favorable properties, including generalizability, limited effort needed for data annotation and algorithm training, fairness, and sustainability. These results support the feasibility of using unstructured EHR data to generate real-world evidence and drive continuous improvements in learning health systems.

##Data and annotation
The input for this algorithm is a clinical note that has been extracted using queries of an EHR database. These notes had been subjected to machine redaction of protected health information prior to the development of our algorithm.
```
The notes were annotated based on:
Mayo Score	Keywords
'0	 ‘normal’, ‘quiescent’, ‘scar’'
1	‘erythema’, ‘decreased vascular pattern’, ‘granularity’, ‘aphthous ulcer’, ‘aphthae’, ‘mild’ 
2	‘friability’, marked or extensive ‘erythema’, ‘loss of vascularity’, ‘absent vascularity’, ‘erosions’, ‘moderate’ 
3	spontaneous ‘bleeding’, ‘ulcer’, ‘ulcerated’, ‘severe’ 
```
 **Installation**: 
Check, requirements.txt file

## Installation

Here are the steps to install and set up the project locally:

1. Clone the repository: `https://github.com/rwelab/MayoClassifier.git`
2. Navigate to the project directory: `cd MayoClassifier`
3. Install dependencies: `pip install` or `yarn install` (depending on the package manager you use)

##Files

Classifier.py creates and evaluvates automl models 
ShortScript.py creates and evaluvates automl models with least number of line of codes 
StandardModels.py creates n-gram based classification models and provides performance evaluvation

## Usage

```bash
# Run the script
python ShortScript.py #run the model with fewer line of codes

python StandardModels.py #generate the baseline models for comapre the results

python Classifier.py #In depth codes for automated machine learning models generation

```
##Sample data
```
The classifer expectes an annoated data in the format of a csv with 2 columns \n
Label, note text\n
Label can be 0,1,2 or 3 based on disease sivierity annotated by a clinician
note text is the clinical note descrbing the mayo score.\n

Eg: 1, the patient was seen in the clinia and reported decreased vascular pattern and aphthous ulcer in the mold form......
```

##Download and Install AutoGluon 

The autoML package that is cutomised for the current study is from AutoGluon 

AutoGluon enables easy-to-use and easy-to-extend AutoML with a focus on automated stack ensembling, deep learning, and real-world applications spanning text, image, and tabular data 

More details can be found at https://auto.gluon.ai/stable/index.html

Installation

GPU Version on Linux system with PIP
python3 -m pip install -U pip 
python3 -m pip install -U setuptools wheel 

*Here we assume CUDA 10.1 is installed. You should change the number 
*according to your own CUDA version (e.g. mxnet_cu100 for CUDA 10.0). 

python3 -m pip install -U "mxnet_cu101<2.0.0" 
python3 -m pip install autogluon 

CPU Version on Linux system with PIP
python3 -m pip install -U pip 
python3 -m pip install -U setuptools wheel 
python3 -m pip install -U "mxnet<2.0.0" 
python3 -m pip install autogluon 

GPU Version on Linux system from Source
python3 -m pip install -U pip 
python3 -m pip install -U setuptools wheel 

*Here we assume CUDA 10.1 is installed. You should change the number 
*according to your own CUDA version (e.g. mxnet_cu102 for CUDA 10.2). 

python3 -m pip install -U "mxnet_cu101<2.0.0" 
git clone https://github.com/awslabs/autogluon 
cd autogluon && ./full_install.sh 

## Contributing

We welcome contributions from the community! If you want to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix: `git checkout -b feature/your-feature-name`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature/your-feature-name`.
5. Submit a pull request to the `main` branch.

Please ensure that your code follows the project's coding style and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

