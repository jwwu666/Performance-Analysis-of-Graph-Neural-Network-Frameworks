import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv as GAT_PYGConv
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import GATConv as GAT_DGLConv
from torch_geometric.nn import global_mean_pool
from dgl.nn.pytorch.glob import AvgPooling
from MLP import MLPReadout
import time
def hook_relu(grad):
    print("elu :", time.perf_counter())

def hook_gcn(grad):
    print("gcn :", time.perf_counter())

def hook(grad):
    print(time.perf_counter())

def hook_pool(grad):
    print("pool :", time.perf_counter())
class GAT_PYG(torch.nn.Module):
    def __init__(self, in_feats, n_classes, n_hidden, n_heads, drop):
        super(GAT_PYG, self).__init__()

        self.layers = nn.ModuleList()

        self.embedding_h = nn.Linear(in_feats, n_hidden * n_heads)

        self.layers.append(GAT_PYGConv(in_feats, n_hidden, heads = n_heads))

        for _ in range(2):
            self.layers.append(GAT_PYGConv(n_hidden * n_heads, n_hidden, n_heads))

        self.layers.append(GAT_PYGConv(n_hidden * n_heads, n_hidden, heads= 1))

        self.MLP = MLPReadout(n_hidden * n_heads, n_classes)

        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_classes)

        self.bn1 = nn.BatchNorm1d(n_hidden * n_heads)
        self.bn2 = nn.BatchNorm1d(n_hidden * n_heads)

        self.dropout = nn.Dropout(p=drop)

    def forward(self, h, edge_index, edge_attr, batch):
        if self.training:
            for l, layer in enumerate(self.layers):
                t1 = time.perf_counter()
                h = layer(h, edge_index)
                print("conv", l, "forward time: ", time.perf_counter() - t1)
                h.register_hook(hook_gcn)
                h = F.relu(h)
                h.register_hook(hook_relu)
                # h = self.dropout(h)

            t2 = time.perf_counter()
            h = global_mean_pool(h, batch)
            print("pooling forward time: ", time.perf_counter() - t2)
            h.register_hook(hook_pool)

            t3 = time.perf_counter()
            h = self.fc1(h)
            print("fc1 forward time: ", time.perf_counter() - t3)
            h.register_hook(hook)

            h = F.elu(h)
            h.register_hook(hook_relu)

            t4 = time.perf_counter()
            h = self.fc2(h)
            print("fc2 forward time: ", time.perf_counter() - t4)
            h.register_hook(hook)
            # h = self.readout(h)
            h = F.log_softmax(h, dim=0)
            h.register_hook(hook)
        else:
            for l, layer in enumerate(self.layers):
                t1 = time.perf_counter()
                h = layer(h, edge_index)
                print("conv", l, "forward time: ", time.perf_counter() - t1)
                h = F.relu(h)
                # h = self.dropout(h)

            t2 = time.perf_counter()
            h = global_mean_pool(h, batch)
            print("pooling forward time: ", time.perf_counter() - t2)

            t3 = time.perf_counter()
            h = self.fc1(h)
            print("fc1 forward time: ", time.perf_counter() - t3)

            h = F.elu(h)

            t4 = time.perf_counter()
            h = self.fc2(h)
            print("fc2 forward time: ", time.perf_counter() - t4)
            # h = self.readout(h)
            h = F.log_softmax(h, dim=0)
        return h

class GAT_DGL(nn.Module):
    def __init__(self, in_feats, n_classes, n_hidden, n_heads, drop, bn):
        super(GAT_DGL, self).__init__()
        #self.g = g
        self.bn = bn
        self.layers = nn.ModuleList()
        self.embedding_h = nn.Linear(in_feats, n_hidden * n_heads)
        self.layers.append(GAT_DGLConv(in_feats, n_hidden, n_heads))
        for _ in range(2):
            self.layers.append(GAT_DGLConv(n_hidden * n_heads, n_hidden, n_heads))

        self.layers.append(GAT_DGLConv(n_hidden * n_heads, n_hidden, 1))

        self.MLP = MLPReadout(n_hidden, n_classes)
        self.avgpooling = AvgPooling()

        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_classes)

        self.dropout = nn.Dropout(p=drop)

        self.bn1 = nn.BatchNorm1d(n_hidden * n_heads)
        self.bn2 = nn.BatchNorm1d(n_hidden)

    def forward(self, g, h):
        if self.training:
            for l, layer in enumerate(self.layers):
                t1 = time.perf_counter()
                h = layer(g, h)
                print("conv", l, "forward time: ", time.perf_counter() - t1)
                h.register_hook(hook_gcn)
                h = h.view(-1, h.size(1) * h.size(2))
                h = F.relu(h)
                h.register_hook(hook_relu)
                #h = self.dropout(h)
            h = h.squeeze()
            t2 = time.perf_counter()
            h = self.avgpooling(g, h)
            print("pooling forward time: ", time.perf_counter() - t2)
            h.register_hook(hook_pool)

            t3 = time.perf_counter()
            h = self.fc1(h)
            print("fc1 forward time: ", time.perf_counter() - t3)
            h.register_hook(hook)

            h = F.elu(h)
            h.register_hook(hook_relu)

            t4 = time.perf_counter()
            h = self.fc2(h)
            print("fc2 forward time: ", time.perf_counter() - t4)
            h.register_hook(hook)
            #h = self.readout(h)
            h = F.log_softmax(h, dim=0)
            h.register_hook(hook)
        else:
            for l, layer in enumerate(self.layers):
                t1 = time.perf_counter()
                h = layer(g, h)
                print("conv", l, "forward time: ", time.perf_counter() - t1)
                h = h.view(-1, h.size(1) * h.size(2))
                h = F.relu(h)
                # h = self.dropout(h)
            h = h.squeeze()
            t2 = time.perf_counter()
            h = self.avgpooling(g, h)
            print("pooling forward time: ", time.perf_counter() - t2)

            t3 = time.perf_counter()
            h = self.fc1(h)
            print("fc1 forward time: ", time.perf_counter() - t3)

            h = F.elu(h)

            t4 = time.perf_counter()
            h = self.fc2(h)
            print("fc2 forward time: ", time.perf_counter() - t4)
            # h = self.readout(h)
            h = F.log_softmax(h, dim=0)
        return h
