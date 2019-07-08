from __future__ import absolute_import
from __future__ import print_function

from utils.readers import InHospitalMortalityReader
from utils import common_utils
from utils.metrics import print_metrics_binary
from in_hospital_mortality.preprocessing import save_results
from sklearn.preprocessing import Imputer, StandardScaler

import os
import numpy as np
import argparse
import json
import xgboost as xgb
import time


def read_and_extract_features(reader, period, features):
    print("number of get_number_of_examples" , reader.get_number_of_examples())
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    # ret = common_utils.read_chunk(reader, 100)
    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    return (X, ret['y'], ret['name'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=str, default='all', help='specifies which period extract features from',
                        choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'len', 'all_but_len'])
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default=os.path.join(os.path.dirname(__file__), '../../../data/in-hospital-mortality/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')
    args = parser.parse_args()
    print(args)

    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                             listfile=os.path.join(args.data, 'train_listfile.csv'),
                                             period_length=48.0)

    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                           listfile=os.path.join(args.data, 'val_listfile.csv'),
                                           period_length=48.0)

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=48.0)

    #print("shape->",train_reader.read_example(100)['X'].shape)

    print('Reading data and extracting features ...')
    (train_X, train_y, train_names) = read_and_extract_features(train_reader, args.period, args.features)
    (val_X, val_y, val_names) = read_and_extract_features(val_reader, args.period, args.features)
    (test_X, test_y, test_names) = read_and_extract_features(test_reader, args.period, args.features)
    print('  train data shape = {}'.format(train_X.shape))
    print('  validation data shape = {}'.format(val_X.shape))
    print('  test data shape = {}'.format(test_X.shape))

    #print("feature sample->", train_X[11])

    print('Imputing missing values ...')
    imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0, verbose=0, copy=True)
    imputer.fit(train_X)
    train_X = np.array(imputer.transform(train_X), dtype=np.float32)
    val_X = np.array(imputer.transform(val_X), dtype=np.float32)
    test_X = np.array(imputer.transform(test_X), dtype=np.float32)

    print('Normalizing the data to have zero mean and unit variance ...')
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)

    file_name = 'xgboost_{}.{}.'.format(args.period, args.features)

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

    eval_set = [(train_X, train_y) , (val_X, val_y)]

    xgreg.fit(train_X, train_y, eval_metric = 'rmse', eval_set = eval_set, verbose = True, early_stopping_rounds = 40)


    result_dir = os.path.join(args.output_dir, 'results')
    common_utils.create_directory(result_dir)

    with open(os.path.join(result_dir, 'train_{}.json'.format(file_name)), 'w') as res_file:
        ret = print_metrics_binary(train_y, xgreg.predict(train_X))
        ret = {k : float(v) for k, v in ret.items()}
        json.dump(ret, res_file)

    with open(os.path.join(result_dir, 'val_{}.json'.format(file_name)), 'w') as res_file:
        ret = print_metrics_binary(val_y, xgreg.predict(val_X))
        ret = {k: float(v) for k, v in ret.items()}
        json.dump(ret, res_file)

    time_start = time.time()
    prediction = xgreg.predict(test_X)
    time_elapse = time.time() - time_start
    print("Processing time on Test set :", time_elapse, " s")

    with open(os.path.join(result_dir, 'test_{}.json'.format(file_name)), 'w') as res_file:
        ret = print_metrics_binary(test_y, prediction)
        ret = {k: float(v) for k, v in ret.items()}
        json.dump(ret, res_file)

    save_results(test_names, prediction, test_y,
                 os.path.join(args.output_dir, 'predictions', file_name + '.csv'))


if __name__ == '__main__':
    main()
