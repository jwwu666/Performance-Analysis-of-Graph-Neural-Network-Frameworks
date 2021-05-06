from torch_geometric.nn import GatedGraphConv as GatedPYGConv
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GatedGraphConv as GatedDGLConv
from torch_geometric.nn import GatedGraphConv as GatedPYGConv
import time

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

class Gated_DGL(nn.Module):

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 dropout):
        super(Gated_DGL, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        # self.activation = activation
        self.embedding_h = nn.Linear(in_feats, n_hidden)
        # input layer
        self.layers.append(GatedDGLConv(n_hidden, n_hidden, 1, 1))
        '''
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEDGLConv(n_hidden, n_hidden, aggregator_type))
        '''
        # output layer
        self.layers.append(GatedDGLConv(n_hidden, n_hidden, 1, 1))  # activation None

        self.readout = nn.Linear(n_hidden, n_classes)

        #self.MLP_layer = MLPReadout(n_hidden, n_classes)

    def forward(self, graph, h, etypes):
        #h = self.dropout(inputs)
        h = self.embedding_h(h)
        for l, layer in enumerate(self.layers):
            t = time.time()
            h = layer(graph, h, etypes)
            print(l, time.time() - t)
            h = F.elu(h)
            h = self.dropout(h)
            #if l != len(self.layers) - 1:
                #h = F.relu(h)
               # h = self.dropout(h)
        h = self.readout(h)
       # h = self.MLP_layer(h)
        return F.log_softmax(h, dim=1)

class Gated_PYG(nn.Module):

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 dropout):
        super(Gated_PYG, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        # self.activation = activation
        self.embedding_h = nn.Linear(in_feats, n_hidden)
        # input layer
        self.layers.append(GatedPYGConv(n_hidden, 1))
        # output layer
        self.layers.append(GatedPYGConv(n_hidden, 1))  # activation None

        self.batch_norms = nn.BatchNorm1d(n_hidden)

        self.readout = nn.Linear(n_hidden, n_classes)

        #self.MLP_layer = MLPReadout(n_hidden, n_classes)

    def forward(self, h, edge_index):
        #h = self.dropout(inputs)
        h = self.embedding_h(h)
        for l, layer in enumerate(self.layers):
            t = time.time()
            h = layer(h, edge_index)
            print(l, time.time() - t)
            h = F.elu(h)
            #h = self.batch_norms(h)
            h = self.dropout(h)

        h = self.readout(h)
        # h = self.MLP_layer(h)
        return F.log_softmax(h, dim=1)