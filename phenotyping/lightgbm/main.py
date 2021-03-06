from __future__ import absolute_import
from __future__ import print_function

from utils.readers import PhenotypingReader
from utils import common_utils
from utils.metrics import print_metrics_multilabel
from phenotyping.preprocessing import save_results
from sklearn.preprocessing import Imputer, StandardScaler

import os
import numpy as np
import argparse
import json
import xgboost as xgb
import lightgbm as lgb
import time


def read_and_extract_features(reader, period, features):
    print("number of get_number_of_examples" , reader.get_number_of_examples())
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    # ret = common_utils.read_chunk(reader, 100)
    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    return (X, ret['y'], ret['name'], ret['t'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=str, default='all', help='specifies which period extract features from',
                        choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'len', 'all_but_len'])
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default=os.path.join(os.path.dirname(__file__), '../../../data/phenotyping/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')
    args = parser.parse_args()
    print(args)

    
    train_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                                     listfile=os.path.join(args.data, 'train_listfile.csv'))
                                     

    val_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                                   listfile=os.path.join(args.data, 'val_listfile.csv'))

    test_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'test'),
                                    listfile=os.path.join(args.data, 'test_listfile.csv'))

    #print("shape->",train_reader.read_example(100)['X'].shape)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    print('Reading data and extracting features ...')
    (train_X, train_y, train_names, train_ts) = read_and_extract_features(train_reader, args.period, args.features)
    train_y = np.array(train_y)
    (val_X, val_y, val_names, val_ts) = read_and_extract_features(val_reader, args.period, args.features)
    val_y = np.array(val_y)
    (test_X, test_y, test_names, test_ts) = read_and_extract_features(test_reader, args.period, args.features)
    test_y = np.array(test_y)
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
    result_dir = os.path.join(args.output_dir, 'results')
    common_utils.create_directory(result_dir)

    n_tasks = 25

    train_activations = np.zeros(shape=train_y.shape, dtype=float)
    val_activations = np.zeros(shape=val_y.shape, dtype=float)
    test_activations = np.zeros(shape=test_y.shape, dtype=float)

    for task_id in range(n_tasks):
        print('Starting task {}'.format(task_id))

        xgreg = lgb.LGBMRegressor(objective='regression',num_leaves=11,learning_rate=0.07,n_estimators=10000)

        eval_set = [(train_X, train_y[:, task_id]) , (val_X, val_y[:, task_id])]

        xgreg.fit(train_X, train_y[:, task_id], eval_metric = 'auc', eval_set = eval_set, verbose = True, early_stopping_rounds = 10)

        train_preds = xgreg.predict(train_X)

        print("train_preds shape ", train_preds.shape)
        print("train_activations shape", train_activations.shape)

        train_activations[:, task_id] = train_preds

        val_preds = xgreg.predict(val_X)
        val_activations[:, task_id] = val_preds

        time_start = time.time()
        test_preds = xgreg.predict(test_X)
        time_elapse = time.time() - time_start
        print("Processing time on Test set :", time_elapse, " s for task ", task_id)
        test_activations[:, task_id] = test_preds

    with open(os.path.join(result_dir, 'train_{}.json'.format(file_name)), 'w') as f:
        ret = print_metrics_multilabel(train_y, train_activations)
        ret = {k: float(v) for k, v in ret.items() if k != 'auc_scores'}
        json.dump(ret, f)

    with open(os.path.join(result_dir, 'val_{}.json'.format(file_name)), 'w') as f:
        ret = print_metrics_multilabel(val_y, val_activations)
        ret = {k: float(v) for k, v in ret.items() if k != 'auc_scores'}
        json.dump(ret, f)

    with open(os.path.join(result_dir, 'test_{}.json'.format(file_name)), 'w') as f:
        ret = print_metrics_multilabel(test_y, test_activations)
        ret = {k: float(v) for k, v in ret.items() if k != 'auc_scores'}
        json.dump(ret, f)

    save_results(test_names, test_ts, test_activations, test_y,
                 os.path.join(args.output_dir, 'predictions', file_name + '.csv'))


if __name__ == '__main__':
    main()
