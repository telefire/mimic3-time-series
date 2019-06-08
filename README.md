MIMIC-III Time Series Models
=========================

## Motivation

Exploring models that best suitable for time series tasks in clinical senarios.


## Data preparation

We do not provide the MIMIC-III data itself. You must acquire the data yourself from https://mimic.physionet.org/. Specifically, download the CSVs. Otherwise, generally we make liberal use of the following packages:

- numpy
- pandas

To extract time series data from MIMIC-III, you need to refer  https://github.com/YerevaNN/mimic3-benchmarks. **Please be sure also to cite the original [MIMIC-III Benchmarks paper](https://arxiv.org/abs/1703.07771) and the original [MIMIC-III paper](http://www.nature.com/articles/sdata201635).**


## Requirments

### In-hospital mortality prediction
```
python in_hospital_mortality/main.py --network models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --output_dir in_hospital_mortality/
 
```

### Decompensation prediction


```
python decompensation/main.py --network models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --output_dir decompensation
```


### Length of stay prediction

### Phenotype classification

### Questions

Please feel free to reach me (Yanke Hu, yhu@humana.com) or Raj (Raj Subramanian, rsubramanian5@humana.com) for any questions or just open a github issue.