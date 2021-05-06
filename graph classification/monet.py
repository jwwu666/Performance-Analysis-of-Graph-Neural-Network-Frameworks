import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GMMConv as GMMDGLConv
from torch_geometric.nn import GMMConv as GMMPYGConv
from torch_geometric.nn import global_mean_pool
import numpy as np
from dgl.nn.pytorch.glob import AvgPooling
import dgl
import time
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
class MoNet_DGL(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_feats,
                 dim,
                 n_kernels,
                 device):
        super(MoNet_DGL, self).__init__()
        self.device = device
        self.layers = nn.ModuleList()
        self.pseudo_proj = nn.ModuleList()

        # Input layer
        self.layers.append(
            GMMDGLConv(in_feats, n_hidden, dim, n_kernels, aggregator_type = "max", allow_zero_in_degree = True))
        self.pseudo_proj.append(
            nn.Sequential(nn.Linear(2, dim), nn.Tanh()))

        # Hidden layer
        for _ in range(2):
            self.layers.append(GMMDGLConv(n_hidden, n_hidden, dim, n_kernels, aggregator_type = "max", allow_zero_in_degree = True))
            self.pseudo_proj.append(
                nn.Sequential(nn.Linear(2, dim), nn.Tanh()))

        # Output layer
        self.layers.append(GMMDGLConv(n_hidden, n_hidden, dim, n_kernels, aggregator_type = "max", allow_zero_in_degree = True))#, allow_zero_in_degree = True
        self.pseudo_proj.append(
            nn.Sequential(nn.Linear(2, dim), nn.Tanh()))

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, out_feats)

        self.avgpooling = AvgPooling()

    def forward(self, g, h, pseudo):
        # pseudo = g.edata['pseudo'].to(self.device).float()
        #h = self.dropout(inputs)
        # if self.training:
        #     for l, layer in enumerate(self.layers):
        #         t1 = time.perf_counter()
        #         h = layer(g, h, self.pseudo_proj[l](pseudo))
        #         print("conv", l, "forward time: ", time.perf_counter() - t1)
        #         h.register_hook(hook_gcn)
        #         h = F.relu(h)
        #         h.register_hook(hook_relu)
        #         #h = self.dropout(h)
        #
        #     t2 = time.perf_counter()
        #     h = self.avgpooling(g, h)
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
        #     #h = self.readout(h)
        #     h = F.log_softmax(h, dim=0)
        #     h.register_hook(hook)
        # else:
        for l, layer in enumerate(self.layers):
            # t1 = time.perf_counter()
            h = layer(g, h, self.pseudo_proj[l](pseudo))
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

    def compute_pseudo(self, edges):
        # compute pseudo edge features for MoNet
        # to avoid zero division in case in_degree is 0, we add constant '1' in all node degrees denoting self-loop
        srcs = 1 / np.sqrt(edges.src['deg'] + 1)
        dsts = 1 / np.sqrt(edges.dst['deg'] + 1)
        pseudo = torch.cat((srcs.unsqueeze(-1), dsts.unsqueeze(-1)), dim=1)
        return {'pseudo': pseudo}

class MoNet_PYG(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_feats,
                 dim,
                 n_kernels,):
        super(MoNet_PYG, self).__init__()
        self.layers = nn.ModuleList()
        self.pseudo_proj = nn.ModuleList()

        # Input layer
        self.layers.append(
            GMMPYGConv(in_feats, n_hidden, dim, n_kernels, aggr = "max", root_weight = False))
        self.pseudo_proj.append(
            nn.Sequential(nn.Linear(2, dim), nn.Tanh()))

        # Hidden layer
        for _ in range(2):
            self.layers.append(GMMPYGConv(n_hidden, n_hidden, dim, n_kernels, aggr = "max", root_weight = False))
            self.pseudo_proj.append(
                nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
        # Output layer
        self.layers.append(GMMPYGConv(n_hidden, n_hidden, dim, n_kernels, aggr = "max", root_weight = False))
        self.pseudo_proj.append(
            nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, out_feats)

    def forward(self, h, edge_index, edge_attr, batch):

        if self.training:
            for l, layer in enumerate(self.layers):
                t1 = time.perf_counter()
                h = layer(h, edge_index, self.pseudo_proj[l](edge_attr))
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
                h = layer(h, edge_index, self.pseudo_proj[l](edge_attr))
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