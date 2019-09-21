from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import imp
import re

import preprocessing
from utils.readers import InHospitalMortalityReader

from utils.preprocessing import Discretizer, Normalizer
from utils import metrics
from utils import common_utils

import time
import json

import xgboost as xgb

from utils.metrics import print_metrics_binary
from in_hospital_mortality.preprocessing import save_results
from sklearn.preprocessing import Imputer, StandardScaler

parser = argparse.ArgumentParser()
#common_utils.add_common_arguments(parser)
parser.add_argument('--data', type=str, help='Path to the data of decompensation task',
                    default=os.path.join(os.path.dirname(__file__), '../../../data/in-hospital-mortality/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
parser.add_argument('--timestep', type=float, help='1.0 or 0.8 or 2.0', default='1.0')

args = parser.parse_args()
print(args)


train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                    listfile=os.path.join(args.data, 'train_listfile.csv'))
val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                  listfile=os.path.join(args.data, 'val_listfile.csv'))

test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                    listfile=os.path.join(args.data, 'test_listfile.csv'))

discretizer = Discretizer(timestep=args.timestep,
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')


discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize


normalizer_state = 'ihm_ts{}.input_str:previous.start_time:zero.normalizer'.format(args.timestep)
normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

# Read data
train_raw = preprocessing.load_data(train_reader, discretizer, normalizer)
val_raw = preprocessing.load_data(val_reader, discretizer, normalizer)
test_raw = preprocessing.load_data(test_reader, discretizer, normalizer)

# Prepare training

print("==> training")

print("train_x ", train_raw[0].shape)
print("train_y ", len(train_raw[1]))
print("val ", len(val_raw[1]))
print("test ", len(test_raw[1]))

print("result", sum(train_raw[1]) + sum(val_raw[1]) + sum(test_raw[1]))

train_raw_reshape = train_raw[0].reshape(17939, 48*76)
val_raw_reshape = val_raw[0].reshape(3222, 48*76)
test_raw_reshape = test_raw[0].reshape(3236, 48*76)

print('Imputing missing values ...')
imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0, verbose=0, copy=True)
imputer.fit(train_raw_reshape)
train_raw_reshape = np.array(imputer.transform(train_raw_reshape), dtype=np.float32)
val_raw_reshape = np.array(imputer.transform(val_raw_reshape), dtype=np.float32)
test_raw_reshape = np.array(imputer.transform(test_raw_reshape), dtype=np.float32)

print('Normalizing the data to have zero mean and unit variance ...')
scaler = StandardScaler()
scaler.fit(train_raw_reshape)
train_raw_reshape = scaler.transform(train_raw_reshape)
val_raw_reshape = scaler.transform(val_raw_reshape)
test_raw_reshape = scaler.transform(test_raw_reshape)


print("train reshape", train_raw_reshape.shape)
print("val reshape", val_raw_reshape.shape)
print("test reshape", test_raw_reshape.shape)

file_name = 'xgboost_raw.'

xgreg = xgb.XGBRegressor(colsample_bytree=0.4,
             gamma=0,                 
             learning_rate=0.07,
             max_depth=3,
             min_child_weight=1.5,
             n_estimators=10000,                                                                    
             reg_alpha=0.75,
             reg_lambda=0.45,
             subsample=0.6,
             seed=42)

eval_set = [(train_raw_reshape, train_raw[1]) , (val_raw_reshape, val_raw[1])]

xgreg.fit(train_raw_reshape, train_raw[1], eval_metric = 'auc', eval_set = eval_set, verbose = True, early_stopping_rounds = 40)


result_dir = os.path.join(args.output_dir, 'results')
common_utils.create_directory(result_dir)

with open(os.path.join(result_dir, 'train_{}.json'.format(file_name)), 'w') as res_file:
    ret = print_metrics_binary(train_raw[1], xgreg.predict(train_raw_reshape))
    ret = {k : float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

with open(os.path.join(result_dir, 'val_{}.json'.format(file_name)), 'w') as res_file:
    ret = print_metrics_binary(val_raw[1], xgreg.predict(val_raw_reshape))
    ret = {k: float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

time_start = time.time()
prediction = xgreg.predict(test_raw_reshape)
time_elapse = time.time() - time_start
print("Processing time on Test set :", time_elapse, " s")

with open(os.path.join(result_dir, 'test_{}.json'.format(file_name)), 'w') as res_file:
    ret = print_metrics_binary(test_raw[1], prediction)
    ret = {k: float(v) for k, v in ret.items()}
    json.dump(ret, res_file)


