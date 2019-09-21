

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
python in_hospital_mortality/rnn/main.py --network models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --output_dir in_hospital_mortality/
 
```

Test
```
python in_hospital_mortality/rnn/main.py --network models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --batch_size 8 --load_state in_hospital_mortality/keras_states/k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch27.test0.27868239298546116.state
```

##### XGBoost
Train & Test
```
python in_hospital_mortality/xgboost/main.py  --output_dir in_hospital_mortality/xgboost/
```

##### XGBoost FBFE
Train & Test
```
python in_hospital_mortality/xgboost_raw/main.py  --timestep 1.0  --output_dir in_hospital_mortality/xgboost_FBFE/
```

##### LightGBM
Train & Test
```
python in_hospital_mortality/lightgbm/main.py  --output_dir in_hospital_mortality/lightgbm/
```

##### LightGBM FBFE
Train & Test
```
python in_hospital_mortality/lightgbm_FBFE/main.py  --timestep 1.0  --output_dir in_hospital_mortality/lightgbm_FBFE/
```
##### Logistic Regression
Train & Test
```
python in_hospital_mortality/lr/main.py --l2 --C 0.001 --output_dir in_hospital_mortality/lr/
```

### Phenotype classification

##### LSTM
Train

```
python phenotyping/rnn/main.py --network models/lstm.py --dim 256 --timestep 1.0 --depth 1 --dropout 0.3 --mode train --batch_size 8 --output_dir phenotyping/
```

##### XGBoost
Train & Test
```
python phenotyping/xgboost/main.py  --output_dir phenotyping/xgboost/
```

##### LightGBM
Train & Test
```
python phenotyping/lightgbm/main.py  --output_dir phenotyping/lightgbm/
```

##### XGBoost FBFE
Train & Test
```
python phenotyping/xgboost_FBFE/main.py --timestep 1.0  --output_dir phenotyping/xgboost_FBFE/
```
