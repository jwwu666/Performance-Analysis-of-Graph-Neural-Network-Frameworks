import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GMMConv as GMMDGLConv
from torch_geometric.nn import GMMConv as GMMPYGConv
import time

class MoNet_DGL(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 out_feats,
                 #n_layers,
                 dim,
                 n_kernels,):
        super(MoNet_DGL, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.pseudo_proj = nn.ModuleList()

        # Input layer
        self.layers.append(
            GMMDGLConv(in_feats, n_hidden, dim, n_kernels, aggregator_type = "mean"))
        self.pseudo_proj.append(
            nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
        '''
        # Hidden layer
        for _ in range(n_layers - 1):
            self.layers.append(GMMConv(n_hidden, n_hidden, dim, n_kernels))
            self.pseudo_proj.append(
                nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
        '''
        # Output layer
        self.layers.append(GMMDGLConv(n_hidden, out_feats, dim, n_kernels, aggregator_type = "mean"))
        self.pseudo_proj.append(
            nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
        self.dropout = nn.Dropout(0.5)

    def forward(self, feat, pseudo):
        h = feat
        for i in range(len(self.layers)):
            if i != 0:
                h = self.dropout(F.relu(h))
            t = time.time()
            h = self.layers[i](
                self.g, h, self.pseudo_proj[i](pseudo))
            print( i, time.time() -t)
        return F.log_softmax(h, dim=1)

class MoNet_PYG(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_feats,
                 #n_layers,
                 dim,
                 n_kernels,):
        super(MoNet_PYG, self).__init__()
        self.layers = nn.ModuleList()
        self.pseudo_proj = nn.ModuleList()

        # Input layer
        self.layers.append(
            GMMPYGConv(in_feats, n_hidden, dim, n_kernels))
        self.pseudo_proj.append(
            nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
        '''
        # Hidden layer
        for _ in range(n_layers - 1):
            self.layers.append(GMMConv(n_hidden, n_hidden, dim, n_kernels))
            self.pseudo_proj.append(
                nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
        '''
        # Output layer
        self.layers.append(GMMPYGConv(n_hidden, out_feats, dim, n_kernels))
        self.pseudo_proj.append(
            nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, edge_attr):
        h = x
        for i in range(len(self.layers)):
            if i != 0:
                h = self.dropout(F.elu(h))
            t = time.time()
            h = self.layers[i](h, edge_index, self.pseudo_proj[i](edge_attr))
            print(i, time.time() - t)
        return F.log_softmax(h, dim=1)