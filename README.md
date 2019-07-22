MIMIC-III Time Series Models
=========================

## Motivation

"Deep Learning" has been a buzz word since its big success in ImageNet 2012 competition, and greatly pushing forward the research frontier in computer vision, speech recognition and NLP since then, but for a larger portion of real world applications that deal with graph data and tabular data, such as social networks, recommendation systems and so on, deep learning techniques haven't shown significant advantages. In this work, we focus on time series tasks, which on one hand are in tablur data format, while on the other hand can be applied with both traditional machine learning techniques and deep learning techniques. We will compare the accuracy and speed performance of different modeling approaches and explore model structures that best suitable for time series tasks in Healthcare domain. 


## Data preparation

We do not provide the MIMIC-III data itself. You must acquire the data yourself from https://mimic.physionet.org/. Specifically, download the CSVs. Otherwise, generally we make liberal use of the following packages:

- numpy
- pandas

To extract time series data from MIMIC-III, you need to refer  https://github.com/YerevaNN/mimic3-benchmarks. **Please be sure also to cite the original [MIMIC-III Benchmarks paper](https://www.nature.com/articles/s41597-019-0103-9) and the original [MIMIC-III paper](http://www.nature.com/articles/sdata201635).**


## Experiments

### In-hospital mortality prediction#

| Model | accuracy| AUC of ROC | AUC of PRC |min(+P, Se) | Processing Time on Test Set (seconds)||
-|---|---|---|---|---|---|
|Logistic Regression|0.895|0.815|0.430|[0.428](https://github.com/telefire/mimic3-time-series/blob/master/experiment_ihm_lr.ipynb?short_path=9d140c8#L48)| 0.006|
|XGBoost|0.894|0.822|0.479|[0.482](https://github.com/telefire/mimic3-time-series/blob/master/experiment_ihm_xgboost.ipynb?short_path=f9be529#L50)| 2.34|
|LSTM|0.898|0.858|0.488|[0.485](https://github.com/telefire/mimic3-time-series/blob/master/ihm_test.ipynb?short_path=276ccf7L138)| 29.19|

### Decompensation prediction

### Length of stay prediction
### Phenotype classification
### MultiTask



For training and testing models for these 4 time series tasks, please refer to [Command.md](Command.md) , there are also jupyter notebooks showing running logs.

### Questions

Please feel free to reach me (Yanke Hu) or Raj (Raj Subramanian) for any questions or just open a github issue.