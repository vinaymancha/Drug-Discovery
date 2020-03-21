import pandas as pd
import numpy as np
from nutsflow import *
from nutsml import *
import sys

data_root = 'C:/Users/vinay/Desktop/Drug Discovery/Data/'
save_root = 'C:/Users/vinay/Desktop/Drug Discovery/preprocessed/'
FEATURE_SCALE = 'log'   # 'uniform'


dataset_names = ['CB1', '3A4', 'DPP4', 'HIVINT', 'HIVPROT', 'LOGD', 'METAB', 'NK1', 'OX1', 'OX2', 'PGP', 'PPB', 'RAT_F', 'TDI', 'THROMBIN']

stat_hold = list() # hold the mean and standard deviation for each data-set

for dataset_name in dataset_names:

    test_filename = data_root + dataset_name + '_test_disguised.csv'
    train_filename = data_root + dataset_name + '_training_disguised.csv'

    test_filename_save = save_root + dataset_name + '_test.csv'
    train_filename_save = save_root + dataset_name + '_training.csv'

    print('Preprocessing dataset ', dataset_name)

    train = pd.read_csv(train_filename)
    test = pd.read_csv(test_filename)

    print(len(train.columns.values))
    print(len(test.columns.values))

    train_inx_set = set(train.columns.values)
    test_inx_set = set(test.columns.values)
    
    # remove columns that are not common to both training and test sets
    train_inx = [inx for inx in train.columns.values if inx in set.intersection(train_inx_set, test_inx_set)]
    test_inx = [inx for inx in test.columns.values if inx in set.intersection(train_inx_set, test_inx_set)]

    train = train[train_inx]
    test = test[test_inx]

    print(train.shape)
    print(test.shape)

    # Normalize activations
    X = np.asarray(train.Act)
    x_mean = np.mean(X)
    x_std = np.std(X)

    stat_hold.append((dataset_name, x_mean, x_std))

    train.Act = (train.Act - x_mean) / x_std
    test.Act = (test.Act - x_mean) / x_std

    # rescale features
    if FEATURE_SCALE == 'log':
        train.iloc[:, 2:] = np.log(train.iloc[:, 2:] + 1)
        test.iloc[:, 2:] = np.log(test.iloc[:, 2:] + 1)

    elif FEATURE_SCALE == 'uniform':
        max_feature = train.max(axis=0)[2:]
        train.iloc[:, 2:] = train.iloc[:, 2:] / max_feature
        test.iloc[:, 2:] = test.iloc[:, 2:] / max_feature
    else:
        sys.exit("Check FEATURE_SCALE.")

    # save data to csv
    train.to_csv(train_filename_save, index=False)
    test.to_csv(test_filename_save, index=False)

    print('Done dataset ', dataset_name)
