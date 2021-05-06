"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import time
import ctypes

_cudart = ctypes.CDLL('libcudart.so')

def cu_prof_start():
  ret = _cudart.cudaProfilerStart()
  if ret != 0:
    raise Exception('cudaProfilerStart() returned %d' % ret)


def cu_prof_stop():
  ret = _cudart.cudaProfilerStop()
  if ret != 0:
    raise Exception('cudaProfilerStop() returned %d' % ret)

class GCN_DGL(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 activation,
                 dropout):
        super(GCN_DGL, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            t = time.time()
            h = layer(self.g, h)
            print("conv ", i, time.time() - t)
        return F.log_softmax(h, dim=1)

class GCN_PYG(torch.nn.Module):
    def __init__(self, in_feats, n_classes):
        super(GCN_PYG, self).__init__()
        '''
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True)
        '''
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_feats, 64))#, cached=True))
        self.layers.append(GCNConv(64, n_classes))#, cached=True))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, edge_index, edge_weight):
        #x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            t = time.time()
            x = layer(x, edge_index, edge_weight)
            print("conv ", i, time.time() - t)
            if i == 0:
                x = F.relu(x)
        return F.log_softmax(x, dim=1)
