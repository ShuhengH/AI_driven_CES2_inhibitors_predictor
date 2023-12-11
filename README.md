# AI_driven_CES2_inhibitors_predictor
## Introduction

Human carboxylesterase 2 (hCES2A) is a prominent serine hydrolase and inhibiting hCES2A can effectively alleviate side effects of hCES2A-substrate drugs like the delayed diarrhea induced by the anticancer drug irinotecan. We developed a synergized workflow combining with the machine learning, molecular simulations and cross-species validation for the de novo inhibitors design of hCES2A. In this demo, machine learning (ML)-based predictors were built for molecular activities prediction. was used to build the predictors for hCES2A inhibitors. 

## Requirements
In order to get started you will need:
  
* Modern NVIDIA GPU, compute capability 3.5 of newer  
* Pytorch 0.4.1  
* Scikit-learn  
* RDKit  
* Numpy  
* pandas  
* pickle  
* RDKit  
* tqdm  
* Mordred

## Installation with Anaconda
If you installed your Python with Anacoda you can run the following commands to get started:

* Create new conda environment with Python 3.6 and activation  
    conda create --new CES2_pred python=3.6  
    conda activate GRU-RNN  
* Install conda dependencies  
    conda install tqdm  
    conda install -c rdkit -c mordred-descriptor mordred  
    conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch
## Demos
We uploaded several demos in a form of iPython notebooks:
* Machine-learning based predictor construction of CES2 inhibitors.ipynb -- Demo_ML-based_predictor  
* DFNN-based_predictor.ipynb -- DFFN-based predictor construction
