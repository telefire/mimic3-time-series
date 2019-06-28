MIMIC-III Time Series Models
=========================

## Motivation

Exploring models that best suitable for time series tasks in clinical senarios.


## Data preparation

We do not provide the MIMIC-III data itself. You must acquire the data yourself from https://mimic.physionet.org/. Specifically, download the CSVs. Otherwise, generally we make liberal use of the following packages:

- numpy
- pandas

To extract time series data from MIMIC-III, you need to refer  https://github.com/YerevaNN/mimic3-benchmarks. **Please be sure also to cite the original [MIMIC-III Benchmarks paper](https://arxiv.org/abs/1703.07771) and the original [MIMIC-III paper](http://www.nature.com/articles/sdata201635).**


## Experiments

### In-hospital mortality prediction#

| Model | accuracy| AUC of ROC | AUC of PRC |min(+P, Se) | Processing Time on Test Set||
-|---|---|---|---|---|---|
|Logistic Regression|0.895|0.815|0.430|[0.428](https://github.com/telefire/mimic3-time-series/blob/master/experiment_ihm_lr.ipynb?short_path=9d140c8#L48)| to do|
|XGBoost|0.894|0.822|0.479|[0.482](https://github.com/telefire/mimic3-time-series/blob/master/experiment_ihm_xgboost.ipynb?short_path=f9be529#L50)| to do|
|LSTM|0.898|0.858|0.488|[0.485](https://github.com/telefire/mimic3-time-series/blob/master/ihm_test.ipynb?short_path=276ccf7L138)| to do|

### Decompensation prediction

### Length of stay prediction
### Phenotype classification
### MultiTask



For training and testing models for these 4 time series tasks, please refer to [Command.md](Command.md) , there are also jupyter notebooks showing running logs.

### Questions

Please feel free to reach me (Yanke Hu, yhu@humana.com) or Raj (Raj Subramanian, rsubramanian5@humana.com) for any questions or just open a github issue.