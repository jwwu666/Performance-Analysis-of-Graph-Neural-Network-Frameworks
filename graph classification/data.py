import torch
import pickle
import torch.utils.data
import time
import os

import random
random.seed(42)

from sklearn.model_selection import StratifiedKFold, train_test_split
from dgl.data import LegacyTUDataset
import csv

def get_all_split_idx(dataset):
    """
        - Split total number of graphs into 3 (train, val and test) in 80:10:10
        - Stratified split proportionate to original distribution of data with respect to classes
        - Using sklearn to perform the split and then save the indexes
        - Preparing 10 such combinations of indexes split to be used in Graph NNs
        - As with KFold, each of the 10 fold have unique test set.
    """
    root_idx_dir = './data/TUs/dgl/'
    if not os.path.exists(root_idx_dir):
        os.makedirs(root_idx_dir)
    all_idx = {}

    # If there are no idx files, do the split and store the files
    if not (os.path.exists(root_idx_dir + dataset.name + '_train.index')):
        print("[!] Splitting the data into train/val/test ...")

        # Using 10-fold cross val to compare with benchmark papers
        k_splits = 10

        cross_val_fold = StratifiedKFold(n_splits=k_splits, shuffle=True)
        k_data_splits = []

        # this is a temporary index assignment, to be used below for val splitting
        for i in range(len(dataset.graph_lists)):
            dataset[i][0].a = lambda: None
            setattr(dataset[i][0].a, 'index', i)

        for indexes in cross_val_fold.split(dataset.graph_lists, dataset.graph_labels):
            remain_index, test_index = indexes[0], indexes[1]
            print(remain_index)
            remain_set = dataset.graph_lists[remain_index.tolist()]

            # Gets final 'train' and 'val'
            train, val, _, __ = train_test_split(remain_set,
                                                 range(len(remain_set)),
                                                 test_size=0.111,
                                                 stratify=remain_set.graph_labels)

            #train, val = format_dataset(train), format_dataset(val)
            test = dataset[test_index]

            # Extracting only idxs
            idx_train = [item[0].a.index for item in train]
            idx_val = [item[0].a.index for item in val]
            idx_test = [item[0].a.index for item in test]

            f_train_w = csv.writer(open(root_idx_dir + dataset.name + '_train.index', 'a+'))
            f_val_w = csv.writer(open(root_idx_dir + dataset.name + '_val.index', 'a+'))
            f_test_w = csv.writer(open(root_idx_dir + dataset.name + '_test.index', 'a+'))

            f_train_w.writerow(idx_train)
            f_val_w.writerow(idx_val)
            f_test_w.writerow(idx_test)

        print("[!] Splitting done!")

    # reading idx from the files
    for section in ['train', 'val', 'test']:
        with open(root_idx_dir + dataset.name + '_' + section + '.index', 'r') as f:
            reader = csv.reader(f)
            all_idx[section] = [list(map(int, idx)) for idx in reader]
    return all_idx

dataset = LegacyTUDataset('ENZYMES')

get_all_split_idx(dataset)