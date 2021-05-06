import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from torch_geometric.nn import GCNConv,  global_mean_pool
import dgl.function as fn
from torch.autograd import Function

class GCNLayer(Function):
    def _init_(self, in_dim, out_dim):
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.conv = GCNConv(in_dim, out_dim)

    def forward(self, x, edge_index, edge_weight):

        x = self.conv(x, edge_index, edge_weight)
        return x

