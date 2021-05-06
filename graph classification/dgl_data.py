import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
import dgl
from torch_geometric.data import DataLoader as pyg_loader
import time

# default collate function
def collate(samples):
    # The input `samples` is a list of pairs (graph, label).
    graphs, labels = map(list, zip(*samples))
    for g in graphs:
        # deal with node feats
        for feat in g.node_attr_schemes().keys():
            # TODO torch.Tensor is not recommended
            # torch.DoubleTensor and torch.tensor
            # will meet error in executor.py@runtime line 472, tensor.py@backend line 147
            # RuntimeError: expected type torch.cuda.DoubleTensor but got torch.cuda.FloatTensor
            g.ndata['feat'] = g.ndata['feat'].float()
        # no edge feats
    torch.cuda.synchronize()
    t = time.perf_counter()
    batched_graph = dgl.batch(graphs)
    torch.cuda.synchronize()
    print("batch:", time.perf_counter() - t)
    labels = torch.tensor(labels)
    return batched_graph, labels


class GraphDataLoader():
    def __init__(self,
                 frame,
                 dataset,
                 batch_size,
                 device,
                 collate_fn=collate,
                 seed=0,
                 shuffle=True,
                 split_name='fold10',
                 fold_idx=0,
                 split_ratio=0.7):

        self.shuffle = shuffle
        self.seed = seed
        self.kwargs = {'pin_memory': True} if 'cuda' in device.type else {}

        if frame == 'pyg':
            labels = dataset.data.y
        else:
            labels = [l for _, l in dataset]


        if split_name == 'fold10':
            train_idx, test_idx = self._split_fold10(
                labels, fold_idx, seed, shuffle)
        elif split_name == 'rand':
            train_idx, test_idx = self._split_rand(
                labels, split_ratio, seed, shuffle)
        else:
            raise NotImplementedError()

        if frame == 'dgl':
            #print(type(dataset))
            #print(train_idx,  test_idx)
            #remain_dataset = [dataset[idx] for idx in train_idx]
            #test_dataset = [dataset[idx] for idx in test_idx]
            #train_sampler = SubsetRandomSampler(train_idx)
            #valid_sampler = SubsetRandomSampler(test_idx)

            stratify = []
            for i in train_idx:
                stratify.append(labels[i])
            # print(labels.__getitem__(train_idx.tolist()))
            #stratify = labels.__getitem__(train_idx.tolist())
            x = np.arange(len(train_idx)).tolist()
            train, valid, _, __ = train_test_split(x,
                                                   range(len(train_idx)),
                                                   test_size=0.111,
                                                   random_state=1,
                                                   stratify=stratify)

            idx_train = [train_idx[idx] for idx in train]
            idx_valid = [train_idx[idx] for idx in valid]
            idx_test = test_idx

            # print(idx_test, idx_valid, idx_train)

            valid_sampler = SubsetRandomSampler(idx_valid)
            train_sampler = SubsetRandomSampler(idx_train)
            test_sampler = SubsetRandomSampler(idx_test)

            print(
                "train_set : valid_set : test_set= ",
                len(idx_train), len(idx_valid), len(idx_test))

            self.train_loader = DataLoader(
                dataset, sampler=train_sampler,
                batch_size=batch_size, collate_fn=collate, num_workers= 0, pin_memory=True)
            self.valid_loader = DataLoader(
                dataset, sampler=valid_sampler,
                batch_size=batch_size, collate_fn=collate, num_workers= 0, pin_memory=True)
            self.test_loader = DataLoader(
                dataset, sampler=test_sampler,
                batch_size=batch_size, collate_fn=collate, num_workers= 0, pin_memory=True)

        else:
            remain_dataset = dataset[train_idx.tolist()]
            #print(labels.__getitem__(train_idx.tolist()))
            stratify = labels.__getitem__(train_idx.tolist())
            x = np.arange(len(remain_dataset)).tolist()
            train, valid, _, __ = train_test_split(x,
                                                 range(len(remain_dataset)),
                                                 test_size=0.111,
                                                 random_state = 1,
                                                 stratify= stratify)


            train_dataset = remain_dataset[train]
            valid_dataset = remain_dataset[valid]
            test_dataset = dataset[test_idx.tolist()]

            #print(train_dataset, valid_dataset, test_dataset)

            print(
                "train_set : valid_set : test_set= ",
                len(train_dataset), len(valid_dataset), len(test_dataset))

            #print(test_idx, train_idx[train], train_idx[valid])

            #t3 = time.time()
            self.test_loader = pyg_loader(test_dataset, batch_size=batch_size)
            self.valid_loader = pyg_loader(valid_dataset, batch_size=batch_size)
            self.train_loader = pyg_loader(train_dataset, batch_size=batch_size)





    def train_valid_test_loader(self):
        return self.train_loader, self.valid_loader, self.test_loader

    def train_valid_loader(self):
        return self.train_loader, self.valid_loader, self.test_loader

    def _split_fold10(self, labels, fold_idx=0, seed=0, shuffle=True):
        ''' 10 flod '''
        assert 0 <= fold_idx and fold_idx < 10, print(
            "fold_idx must be from 0 to 9.")

        idx_list = []
        skf = StratifiedKFold(n_splits=10, shuffle=shuffle, random_state=seed)
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):    # split(x, y)
            idx_list.append(idx)
        train_idx, valid_idx = idx_list[fold_idx]

        #print( "train_set : test_set = ",len(train_idx), len(valid_idx))

        return train_idx, valid_idx

    def _split_rand(self, labels, split_ratio=0.7, seed=0, shuffle=True):
        num_entries = len(labels)
        indices = list(range(num_entries))
        np.random.seed(seed)
        np.random.shuffle(indices)
        split = int(math.floor(split_ratio * num_entries))
        train_idx, valid_idx = indices[:split], indices[split:]

        print(
            "train_set : test_set = %d : %d",
            len(train_idx), len(valid_idx))

        return train_idx, valid_idx