import os.path as osp
import numpy as np
import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = "7"

import torch.nn.functional as F
from dgl.data import TUDataset
from dgl.data import LegacyTUDataset
from dgl_data import GraphDataLoader
from train_TUs_dgl import train_dgl, test_dgl
from gcn import GCN_DGL
import dgl
import time
from dgl.data import QM7bDataset
import gc
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
import dgl
from torch_geometric.data import DataLoader as pyg_loader

batch_time = 1
# default collate function
def collate(samples):
    global  batch_time
    t = time.perf_counter()
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
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    batch_time = batch_time +time.perf_counter() - t
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

epoch_load_time = 0
epoch_batch_time = 0
epoch_forward_time = 0
epoch_backward_time = 0
epoch_update_time = 0

def accuracy(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    return acc


# def train_dgl(model_name, model, optimizer, device, data_loader):
#     global epoch_load_time, epoch_forward_time, epoch_backward_time, epoch_batch_time, epoch_update_time, batch_time
#     t10 = time.perf_counter()
#     model.train()
#     epoch_loss = 0
#     epoch_train_acc = 0
#     nb_data = 0
#     #t4 = time.perf_counter()
#     t2 = time.perf_counter()
#     batch_time = 0
#     for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
#         epoch_load_time = epoch_load_time + time.perf_counter() - t2
#         print("data load time:", time.perf_counter() - t2)
#         t7 = time.perf_counter()
#         batch_graphs = batch_graphs.to(device)
#         batch_x = batch_graphs.ndata['feat']#.to(device)  # num x feat
#         #batch_e = batch_graphs.edata['feat'].to(device)
#         batch_labels = batch_labels.to(device)
#         print("data to gpu:", time.perf_counter() - t7)
#         optimizer.zero_grad()
#         t9 = time.perf_counter()
#         batch_scores = model.forward(batch_graphs, batch_x)
#         #batch_scores = model.forward(batch_graphs, batch_x)
#         epoch_forward_time = epoch_forward_time + time.perf_counter() - t9
#         print("forward:", time.perf_counter() - t9)
#
#         t = time.perf_counter()
#         loss = F.nll_loss(batch_scores, batch_labels)
#         print("compute loss:", time.perf_counter() - t)
#
#         t3 = time.perf_counter()
#         loss.backward()
#         epoch_backward_time = epoch_backward_time + time.perf_counter() - t3
#         print("backward:", time.perf_counter() - t3)
#
#         tt = time.perf_counter()
#         optimizer.step()
#         epoch_update_time = epoch_update_time + time.perf_counter() - tt
#         print("update:", time.perf_counter() - tt)
#
#         t = time.perf_counter()
#         epoch_loss += loss.item()
#         print("detach:", time.perf_counter() - t)
#         t1 = time.perf_counter()
#         epoch_train_acc += accuracy(batch_scores, batch_labels)
#         print("accuracy:", time.perf_counter() - t1)
#         nb_data += batch_labels.size(0)
#         print("last:", time.perf_counter() - t)
#         epoch_batch_time = epoch_batch_time + time.perf_counter() - t7
#         print("a batch:", time.perf_counter() - t7)
#         t2 = time.perf_counter()
#     epoch_loss /= (iter + 1)
#     epoch_train_acc /= nb_data
#     print("train:", time.perf_counter() - t10)
#     return epoch_loss, epoch_train_acc, optimizer

def train_dgl(model_name, model, optimizer, device, data_loader):
    global epoch_load_time, epoch_forward_time, epoch_backward_time, epoch_batch_time,  epoch_update_time
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0

    torch.cuda.synchronize()
    t2 = time.perf_counter()
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        torch.cuda.synchronize()
        end = time.perf_counter() - t2
        epoch_load_time = epoch_load_time + end
        print("data load time:", end)

        t7 = time.perf_counter()

        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat']
        batch_labels = batch_labels.to(device)

        torch.cuda.synchronize()
        end = time.perf_counter() - t7

        print("data to gpu:", end)


        optimizer.zero_grad()
        #print("model pre:", time.time() - t7)

        torch.cuda.synchronize()
        t9 = time.perf_counter()
        batch_scores = model.forward(batch_graphs, batch_x)
        torch.cuda.synchronize()
        end = time.perf_counter() - t9
        epoch_forward_time = epoch_forward_time + end
        print("forward:", end)

        t = time.perf_counter()
        loss = F.nll_loss(batch_scores, batch_labels)
        torch.cuda.synchronize()
        end = time.perf_counter() - t
        print(" computer loss:", end)

        torch.cuda.synchronize()
        t3 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        end = time.perf_counter() - t3
        epoch_backward_time = epoch_backward_time + end
        print("backward:", end)


        tt = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        end = time.perf_counter() - tt
        epoch_update_time = epoch_update_time + end
        print("update:", end)

        epoch_loss += loss.item()
        #epoch_train_acc += accuracy(batch_scores, batch_labels)
        pred = batch_scores.max(dim=1)[1]
        epoch_train_acc += pred.eq(batch_labels).sum().item()
        nb_data += batch_labels.size(0)

        torch.cuda.synchronize()
        end = time.perf_counter() - t7
        epoch_batch_time = epoch_batch_time + end
        print("a batch:", end)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    return epoch_loss, epoch_train_acc, optimizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

t0 = time.time()

dataset = LegacyTUDataset(name='DD', force_reload = False, verbose= True)#Legacy

#dataset = QM7bDataset(force_reload = False, verbose= True)

print("load data time : {:.4f}".format(time.time() - t0))

dataset.graph_lists = [dgl.add_self_loop(g) for g in dataset.graph_lists]
#dataset.graph_lists = [dgl.add_self_loop(g) for g in dataset.graphs]


train_loader, valid_loader, test_loader = GraphDataLoader('dgl',
        dataset, batch_size=128, device=torch.device('cuda'),
        seed=0, shuffle=True,
        split_name='fold10').train_valid_loader()


num_features = dataset.graph_lists[0].ndata['feat'][0].shape[0]
n_classes = dataset.num_labels
print("n_feats, n_classes:", num_features, n_classes)

model = GCN_DGL(num_features, 128, n_classes, 0.0).to(device)

total_param = 0
print("MODEL DETAILS:\n")
for param in model.parameters():
        # print(param.data.size())
    total_param += np.prod(list(param.data.size()))
print('Total parameters:', total_param)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay = 0)
scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=0.5,
                                                             patience=25,
                                                             verbose=True)

dur = []
total_load = 0
total_fp = 0
total_bp = 0
total_up = 0
total_ot = 0
for epoch in range(1, 201):

    torch.cuda.synchronize()
    t1 = time.time()

    train_loss, epoch_train_acc, optimizer = train_dgl('gcn', model, optimizer, device, train_loader)
    # gc.collect()

    torch.cuda.synchronize()
    dur.append(time.time() - t1)

    print('load Time: {:.4f},forward Time: {:.4f}, backward Time: {:.4f}, update Time: {:.4f}, batch Time: {:.4f}'.format(epoch_load_time,
                                                                                                     epoch_forward_time,
                                                                                                     epoch_backward_time,
                                                                                                     epoch_update_time,
                                                                                                     epoch_batch_time))
    if epoch > 50 and epoch < 151:
        total_load = total_load + epoch_load_time
        total_fp = total_fp + epoch_forward_time
        total_bp = total_bp + epoch_backward_time
        total_up = total_up + epoch_update_time
        total_ot = epoch_batch_time - epoch_forward_time - epoch_backward_time - epoch_update_time + total_ot

    epoch_load_time, epoch_forward_time, epoch_backward_time, epoch_batch_time , epoch_update_time = 0, 0, 0, 0, 0

    test_loss, test_acc = test_dgl('gcn', model, device, test_loader)

    scheduler.step(test_loss)

    print('Epoch: {:03d}, Train Loss: {:.4f}, '
          'Train Acc: {:.4f}, Test Acc: {:.4f}, Time: {:.4f}'.format(epoch, train_loss,
                                                                     epoch_train_acc, test_acc, np.mean(dur)))

print('load: {:04f}, forward: {:.4f}, '
          'backward: {:.4f}, update: {:.4f}, other: {:.4f}'.format(total_load/100, total_fp/100, total_bp/100, total_up/100, total_ot/100))

print('Total Time: {:.4f}'.format(time.time() - t0))