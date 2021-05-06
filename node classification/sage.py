from torch_geometric.nn import SAGEConv as SAGEPYGConv
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv as SAGEDGLConv
import time
class SAGE_DGL(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 #n_layers,
                 #activation,
                 dropout,
                 aggregator_type):
        super(SAGE_DGL, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        #self.activation = activation

        # input layer
        self.layers.append(SAGEDGLConv(in_feats, n_hidden, aggregator_type))
        '''
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEDGLConv(n_hidden, n_hidden, aggregator_type))
        '''
        # output layer
        self.layers.append(SAGEDGLConv(n_hidden, n_classes, aggregator_type)) # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            t = time.time()
            h = layer(graph, h)
            print( l, time.time() -t)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return F.log_softmax(h, dim=1)

class SAGE_PYG(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 #n_layers,
                 #activation,
                 dropout):
        super(SAGE_PYG, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        #self.activation = activation

        # input layer
        self.layers.append(SAGEPYGConv(in_feats, n_hidden))
        '''
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEDGLConv(n_hidden, n_hidden, aggregator_type))
        '''
        # output layer
        self.layers.append(SAGEPYGConv(n_hidden, n_classes)) # activation None

    def forward(self, h, edge_index):
        h = self.dropout(h)
        for l, layer in enumerate(self.layers):
            t = time.time()
            h = layer(h, edge_index)
            print(l, time.time() - t)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return F.log_softmax(h, dim=1)