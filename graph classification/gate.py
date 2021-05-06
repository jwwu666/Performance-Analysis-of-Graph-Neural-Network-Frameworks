from torch_geometric.nn import GatedGraphConv as GatedPYGConv
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GatedGraphConv as GatedDGLConv
from torch_geometric.nn import GatedGraphConv as GatedPYGConv
from torch_geometric.nn import global_mean_pool
from dgl.nn.pytorch.glob import AvgPooling
import time
from pynvml import *
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
                 dropout,
                 device):
        super(Gated_DGL, self).__init__()
        self.device = device
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        # self.activation = activation
        self.embedding_h = nn.Linear(in_feats, n_hidden)
        # input layer
        self.layers.append(GatedDGLConv(in_feats, n_hidden, 1, 1))

        # hidden layers
        for i in range(2):
            self.layers.append(GatedDGLConv(n_hidden, n_hidden, 1, 1))

        # output layer
        self.layers.append(GatedDGLConv(n_hidden, n_hidden, 1, 1))  # activation None

        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_classes)

        self.avgpooling = AvgPooling()

        #self.MLP_layer = MLPReadout(n_hidden, n_classes)

    def forward(self, g, h, e):
        # e = g.edata['feat']#.to(self.device)
        # h = g.ndata['feat']
        #h = self.dropout(inputs)
        if self.training:
            for l, layer in enumerate(self.layers):
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                h = layer(g, h, e)
                torch.cuda.synchronize()
                print("conv", l, "forward time: ", time.perf_counter() - t1)
                h.register_hook(hook_gcn)
                h = F.relu(h)
                h.register_hook(hook_relu)
                #h = self.dropout(h)

            torch.cuda.synchronize()
            t2 = time.perf_counter()
            h = self.avgpooling(g, h)
            torch.cuda.synchronize()
            print("pooling forward time: ", time.perf_counter() - t2)
            h.register_hook(hook_pool)

            torch.cuda.synchronize()
            t3 = time.perf_counter()
            h = self.fc1(h)
            torch.cuda.synchronize()
            print("fc1 forward time: ", time.perf_counter() - t3)
            h.register_hook(hook)

            h = F.elu(h)
            h.register_hook(hook_relu)

            torch.cuda.synchronize()
            t4 = time.perf_counter()
            h = self.fc2(h)
            torch.cuda.synchronize()
            print("fc2 forward time: ", time.perf_counter() - t4)
            h.register_hook(hook)
            #h = self.readout(h)
            h = F.log_softmax(h, dim=0)
            h.register_hook(hook)
        else:
            for l, layer in enumerate(self.layers):
                # t1 = time.perf_counter()
                h = layer(g, h, e)
                # print("conv", l, "forward time: ", time.perf_counter() - t1)
                h = F.relu(h)
                # h = self.dropout(h)

            # t2 = time.perf_counter()
            h = self.avgpooling(g, h)
            # print("pooling forward time: ", time.perf_counter() - t2)

            # t3 = time.perf_counter()
            h = self.fc1(h)
            # print("fc1 forward time: ", time.perf_counter() - t3)

            h = F.elu(h)

            # t4 = time.perf_counter()
            h = self.fc2(h)
            # print("fc2 forward time: ", time.perf_counter() - t4)
            # h = self.readout(h)
            h = F.log_softmax(h, dim=0)
        return h

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
        self.layers.append(GatedPYGConv(in_feats, 1))
        # input layer
        for i in range(2):
            self.layers.append(GatedPYGConv(n_hidden, 1))
        # output layer
        self.layers.append(GatedPYGConv(n_hidden, 1))  # activation None

        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_classes)

        self.MLP_layer = MLPReadout(n_hidden, n_classes)

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
                # t1 = time.perf_counter()
                h = layer(h, edge_index)
                # print("conv", l, "forward time: ", time.perf_counter() - t1)
                h = F.relu(h)
                # h = self.dropout(h)

            # t2 = time.perf_counter()
            h = global_mean_pool(h, batch)
            # print("pooling forward time: ", time.perf_counter() - t2)

            # t3 = time.perf_counter()
            h = self.fc1(h)
            # print("fc1 forward time: ", time.perf_counter() - t3)

            h = F.elu(h)

            # t4 = time.perf_counter()
            h = self.fc2(h)
            # print("fc2 forward time: ", time.perf_counter() - t4)
            # h = self.readout(h)
            h = F.log_softmax(h, dim=0)
        return h