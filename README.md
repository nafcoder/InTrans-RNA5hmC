# InTrans-RNA5hmC

### Architecture of InceptionModule
![inception_module-1](https://github.com/user-attachments/assets/52ae3bc0-b6df-405f-9c11-eea71d8da770)

### Model architecture of InTrans-RNA5hmC
![InTrans-RNA5hmC-1](https://github.com/user-attachments/assets/4d34a704-e552-4168-b77f-13b05e00123b)

### Data availability
All training and independent datasets are given in the [dataset](Dataset) folder

### Environments
OS: Pop!_OS 22.04 LTS

Python version: Python 3.9.19


Used libraries: 
```
numpy==1.26.4
pandas==2.2.1
pytorch==2.4.1
xgboost==2.0.3
pickle5==0.0.11
scikit-learn==1.2.2
matplotlib==3.8.2
PyQt5==5.15.10
imblearn==0.0
skops==0.9.0
shap==0.45.1
IPython==8.18.1
tqdm==4.66.5
biopython==1.84
transformers==4.44.2
```

### Reproduce results
1. We have given the training and testing scripts in [Training](Training) and [Testing](Testing) folders respectively.

### Prediction
#### Prerequisites
To generate the RiNALMo embedding, please see [RiNALMo GitHub](https://github.com/lbcb-sci/RiNALMo).

#### Steps
1. Firstly, you need to fill up the [dataset.txt](Predict/dataset.txt) file. Follow the pattern shown below:

```
>seq_id1
Fasta
>seq_id2
Fasta
```

2. For predicting 5-Hydroxymethylcytosine modification from the RNA sequences, you need to run the [extract_word_embedding.py](Predict/extract_word_embedding.py) to generate Word embedding and then run [predict.py](Predict/predict.py) for prediction. Before prediction, the RiNALMo embedding should be generated and stored in [RiNALMo.npy](Predict/RiNALMo.npy) file.

### Reproduce previous paper metrics for 5-Hydroxymethylcytosine modification
In the [existing SOTA models](existing\ SOTA\ models) folder, scripts are provided for reproducing the results of the previous papers.
