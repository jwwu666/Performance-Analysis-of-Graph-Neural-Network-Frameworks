
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from torch_geometric.nn import GCNConv,  global_mean_pool
import torch.nn.functional as F
from dgl.nn.pytorch.glob import AvgPooling
from graph.MLP import MLPReadout
import time

class GCN_DGL(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 dropout,):
        super(GCN_DGL, self).__init__()
        self.layers = nn.ModuleList()
        self.embedding_h = nn.Linear(in_feats, n_hidden)
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, allow_zero_in_degree = True))#, activation=activation))
        for _ in range(3):
            self.layers.append(GraphConv(n_hidden, n_hidden, allow_zero_in_degree = True))#, activation=activation))
        # output layer
        #self.layers.append(GraphConv(n_hidden, n_hidden))

        self.dropout = nn.Dropout(p=dropout)

        self.MLP = MLPReadout(n_hidden, n_classes)

        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_classes)

        self.bn = nn.BatchNorm1d(n_hidden)

        self.avgpooling = AvgPooling()

    def forward(self, g):
        t = time.time()
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
           # t1 = time.time()
            h = layer(g, h)
           # print("conv ", i, time.time() - t1)
            h = self.bn(h)
            h = F.relu(h)

            # h = self.dropout(h)
        #print(time.time() - t)

        h = self.avgpooling(g, h)
        #print("pool ", time.time() - t)


        # h = self.fc1(h)
        # h = F.relu(h)
        # h = self.fc2(h)
        h = self.MLP(h)
        return F.log_softmax(h, dim=0)

class GCN_PYG(torch.nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, dropout):
        super(GCN_PYG, self).__init__()
        self.layers = nn.ModuleList()
        self.embedding_h = nn.Linear(in_feats, n_hidden)

        # input layer
        self.layers.append(GCNConv(in_feats, n_hidden))
        for _ in range(3):
            self.layers.append(GCNConv(n_hidden, n_hidden))
        self.dropout = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_classes)

        self.MLP = MLPReadout(n_hidden, n_classes)
        self.bn = nn.BatchNorm1d(n_hidden)

    def forward(self, data):
        # print('Inside Model:  num graphs: {}, device: {}'.format(
        #     data.num_graphs, data.batch.device))

        x, edge_index, edge_weight = x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, layer in enumerate(self.layers):
            #t1 = time.time()
            x = layer(x, edge_index, edge_weight)
            #print(i, time.time() - t1)
            x = self.bn(x)
            x = F.relu(x)
            #x = self.dropout(x)

        x = global_mean_pool(x, data.batch)

        x = self.MLP(x)

        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)

        return F.log_softmax(x, dim=0)

