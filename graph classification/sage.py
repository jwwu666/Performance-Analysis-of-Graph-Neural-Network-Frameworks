from torch_geometric.nn import SAGEConv as SAGEPYGConv
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv as SAGEDGLConv
from torch_geometric.nn import global_mean_pool
from dgl.nn.pytorch.glob import AvgPooling
import time
from MLP import MLPReadout
import torch
def hook_relu(grad):
    torch.cuda.synchronize()
    print("elu :", time.perf_counter())

def hook_gcn(grad):
    torch.cuda.synchronize()
    print("gcn :", time.perf_counter())

def hook(grad):
    torch.cuda.synchronize()
    print(time.perf_counter())

def hook_pool(grad):
    torch.cuda.synchronize()
    print("pool :", time.perf_counter())
class SAGE_DGL(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 #n_layers,
                 #activation,
                 dropout,
                 aggregator_type,
                 device):
        super(SAGE_DGL, self).__init__()
        self.device = device
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.avgpooling = AvgPooling()
        #self.activation = activation

        # input layer
        self.layers.append(SAGEDGLConv(in_feats, n_hidden, aggregator_type))

        # hidden layers
        for i in range(2):
            self.layers.append(SAGEDGLConv(n_hidden, n_hidden, aggregator_type))

        # output layer
        self.layers.append(SAGEDGLConv(n_hidden, n_hidden, aggregator_type)) # activation None

        self.readout = nn.Linear(n_hidden, n_classes)

        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_classes)

    def forward(self, g, h):
        #h = self.dropout(inputs)
        # if self.training:
        #     for l, layer in enumerate(self.layers):
        #         torch.cuda.synchronize()
        #         t1 = time.perf_counter()
        #         h = layer(g, h)
        #         torch.cuda.synchronize()
        #         print("conv", l, "forward time: ", time.perf_counter() - t1)
        #         h.register_hook(hook_gcn)
        #         h = F.relu(h)
        #         h.register_hook(hook_relu)
        #         #h = self.dropout(h)
        #     torch.cuda.synchronize()
        #     t2 = time.perf_counter()
        #     h = self.avgpooling(g, h)
        #     torch.cuda.synchronize()
        #     print("pooling forward time: ", time.perf_counter() - t2)
        #     h.register_hook(hook_pool)
        #
        #     torch.cuda.synchronize()
        #     t3 = time.perf_counter()
        #     h = self.fc1(h)
        #     torch.cuda.synchronize()
        #     print("fc1 forward time: ", time.perf_counter() - t3)
        #     h.register_hook(hook)
        #
        #     h = F.elu(h)
        #     h.register_hook(hook_relu)
        #
        #     torch.cuda.synchronize()
        #     t4 = time.perf_counter()
        #     h = self.fc2(h)
        #     torch.cuda.synchronize()
        #     print("fc2 forward time: ", time.perf_counter() - t4)
        #     h.register_hook(hook)
        #     #h = self.readout(h)
        #     h = F.log_softmax(h, dim=0)
        #     h.register_hook(hook)
        # else:
        for l, layer in enumerate(self.layers):
            t1 = time.perf_counter()
            h = layer(g, h)
            print("conv", l, "forward time: ", time.perf_counter() - t1)
            h = F.relu(h)
            # h = self.dropout(h)

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

        # hidden layers
        for i in range(2):
            self.layers.append(SAGEPYGConv(n_hidden, n_hidden))

        # output layer
        self.layers.append(SAGEPYGConv(n_hidden, n_hidden)) # activation None

        self.bn = nn.BatchNorm1d(n_hidden)

        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_classes)

        self.readout = MLPReadout(n_hidden, n_classes)

    def forward(self, h, edge_index, edge_attr, batch):
        # if self.training:
        #     for l, layer in enumerate(self.layers):
        #         t1 = time.perf_counter()
        #         h = layer(h, edge_index)
        #         print("conv", l, "forward time: ", time.perf_counter() - t1)
        #         h.register_hook(hook_gcn)
        #         h = F.relu(h)
        #         h.register_hook(hook_relu)
        #         # h = self.dropout(h)
        #
        #     t2 = time.perf_counter()
        #     h = global_mean_pool(h, batch)
        #     print("pooling forward time: ", time.perf_counter() - t2)
        #     h.register_hook(hook_pool)
        #
        #     t3 = time.perf_counter()
        #     h = self.fc1(h)
        #     print("fc1 forward time: ", time.perf_counter() - t3)
        #     h.register_hook(hook)
        #
        #     h = F.elu(h)
        #     h.register_hook(hook_relu)
        #
        #     t4 = time.perf_counter()
        #     h = self.fc2(h)
        #     print("fc2 forward time: ", time.perf_counter() - t4)
        #     h.register_hook(hook)
        #     # h = self.readout(h)
        #     h = F.log_softmax(h, dim=0)
        #     h.register_hook(hook)
        # else:
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