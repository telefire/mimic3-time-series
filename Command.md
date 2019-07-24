

## Requirments

If you are using Python with a Terminal, you can set python environment by typing this under mimic3-time-series folder
```
export PYTHONPATH=. 
```
If you are running with Jupyter Notebook, you can set python environment by typing this under mimic3-time-series folder
```
PYTHONPATH=. jupyter notebook --ip 0.0.0.0 --port 8003
```

### In-hospital mortality prediction#

##### LSTM
Train
```
python in_hospital_mortality/main.py --network models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --output_dir in_hospital_mortality/
 
```

Test
```
python in_hospital_mortality/main.py --network models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --batch_size 8 --load_state in_hospital_mortality/keras_states/k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch27.test0.27868239298546116.state
```

##### XGBoost
Train & Test
```
python in_hospital_mortality/xgboost/main.py  --output_dir in_hospital_mortality/xgboost/
```

##### Logistic Regression
Train & Test
```
python in_hospital_mortality/lr/main.py --l2 --C 0.001 --output_dir in_hospital_mortality/lr/
```


### Decompensation prediction

##### LSTM
Train

```
python decompensation/main.py --network models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --output_dir decompensation
```

##### XGBoost
Train & Test
```
python decompensation/xgboost/main.py  --output_dir decompensation/xgboost/
```

### Phenotype classification

##### LSTM
Train

```
python phenotyping/main.py --network models/lstm.py --dim 256 --timestep 1.0 --depth 1 --dropout 0.3 --mode train --batch_size 8 --output_dir phenotyping/
```

##### XGBoost
Train & Test
```
python phenotyping/xgboost/main.py  --output_dir phenotyping/xgboost/
```


### Length of stay prediction

##### LSTM
Train
```
python length_of_stay/main.py --network models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --output_dir length_of_stay
```

##### XGBoost
Train & Test
```
python length_of_stay/xgboost/main.py  --output_dir length_of_stay/xgboost/
```


### MultiTask

##### LSTM
Train
```
python multitask/main.py --network models/multitask_lstm.py --dim 512  --timestep 1.0  --dropout 0.3 --mode train --batch_size 16  --ihm_C 0.2 --decomp_C 1.0 --los_C 1.5 --pheno_C 1.0   --output_dir multitask/
```

##### XGBoost
Train & Test
```
python multitask/xgboost/main.py  --output_dir multitask/xgboost/
```