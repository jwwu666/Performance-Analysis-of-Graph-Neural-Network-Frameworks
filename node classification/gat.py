import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv as GAT_PYGConv
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import GATConv as GAT_DGLConv
import time

class GAT_PYG(torch.nn.Module):
    def __init__(self, in_feats, n_classes):
        super(GAT_PYG, self).__init__()
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GAT_PYGConv(in_feats, 8, heads=8, dropout=0))
        # On the Pubmed dataset, use heads=8 in conv2.
        self.gat_layers.append(GAT_PYGConv(8 * 8, n_classes, heads=1, concat=False, dropout=0))

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)

        t = time.time()
        x = self.gat_layers[0](x, edge_index)
        print("conv ", 0, time.time() - t)

        x = F.relu(x)

        x = F.dropout(x, p=0.6, training=self.training)
        t = time.time()
        x = self.gat_layers[-1](x, edge_index)
        print("conv ", 1, time.time() - t)

        return F.log_softmax(x, dim=1)

class GAT_DGL(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 num_hidden,
                 num_classes,
                 feat_drop):
        super(GAT_DGL, self).__init__()
        self.g = g
        #self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.feat_drop = feat_drop
        self.gat_layers.append(GAT_DGLConv(in_dim, num_hidden, 8))

        # output projection
        self.gat_layers.append(GAT_DGLConv(num_hidden * 8, num_classes, 1))

    def forward(self, inputs):
        h = inputs
        h = F.dropout(h, p = self.feat_drop, training=self.training)
        t = time.time()
        h = self.gat_layers[0](self.g, h).flatten(1)
        print("conv ", 0, time.time() - t)

        h = F.relu(h)
        h = F.dropout(h, p=self.feat_drop, training=self.training)
        # output projection

        t = time.time()
        h = self.gat_layers[-1](self.g, h).mean(1)
        print("conv ", 1, time.time() - t)

        return F.log_softmax(h, dim=1)