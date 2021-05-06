import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from dgl.nn.pytorch import GINConv as GIN_DGLConv
from torch_geometric.nn import GINConv as GIN_PYGConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from dgl.nn.pytorch.glob import AvgPooling
import time

def hook_relu(grad):
    torch.cuda.synchronize()
    print("elu :", time.perf_counter())

def hook_gcn(grad):
    torch.cuda.synchronize()
    print("gcn :", time.perf_counter())

def hook(grad):
    torch.cuda.synchronize()
    print("fc :", time.perf_counter())

def hook_pool(grad):
    torch.cuda.synchronize()
    print("pool :", time.perf_counter())

class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)

class GIN_DGL(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes,
                 neighbor_pooling_type = "sum"):

        super(GIN_DGL, self).__init__()
        #self.learn_eps = learn_eps
        self.avgpooling = AvgPooling()
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.embedding_h = nn.Linear(input_dim, hidden_dim)

        for layer in range(4):
            if layer == 0:
                mlp = MLP(1, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(1, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GIN_DGLConv(ApplyNodeFunc(mlp), aggregator_type = neighbor_pooling_type))
        self.batch_norms = nn.BatchNorm1d(hidden_dim)

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

        for i in range(5):
            self.linears_prediction.append(nn.Linear(hidden_dim, n_classes))

        self.drop = nn.Dropout(0.6)


    def forward(self, g, h):
        # list of hidden representation at each layer (including input)
        #h = self.embedding_h(h)

        #hidden_rep = [h]
        # if self.training:
        #     for l, layer in enumerate(self.ginlayers):
        #         torch.cuda.synchronize()
        #         t1 = time.perf_counter()
        #         h = layer(g, h)
        #         torch.cuda.synchronize()
        #         print("conv", l, "forward time: ", time.perf_counter() - t1)
        #         h.register_hook(hook_gcn)
        #         h = F.relu(h)
        #         h.register_hook(hook_relu)
        #         # h = self.dropout(h)
        #
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
        #     # h = self.readout(h)
        #     h = F.log_softmax(h, dim=0)
        #     h.register_hook(hook)
        # else:
        for l, layer in enumerate(self.ginlayers):
            # t1 = time.perf_counter()
            h = layer(g, h)
            # torch.cuda.synchronize()
            # print("conv", l, "forward time: ", time.perf_counter() - t1)
            h = F.relu(h)
            # h = self.dropout(h)

        # t2 = time.perf_counter()
        h = self.avgpooling(g, h)
        # torch.cuda.synchronize()
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

class GIN_PYG(torch.nn.Module):
    def __init__(self, in_feats, hidden_dim, n_classes):
        super(GIN_PYG, self).__init__()

        #num_features = dataset.num_features

        self.ginlayers = torch.nn.ModuleList()
        self.embedding_h = nn.Linear(in_feats, hidden_dim)

        for layer in range(4):
            if layer == 0:
                mlp = MLP(1, in_feats, hidden_dim, hidden_dim)
            else:
                mlp = MLP(1, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GIN_PYGConv(ApplyNodeFunc(mlp)))
        self.batch_norms = nn.BatchNorm1d(hidden_dim)

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

        self.drop = nn.Dropout(0.5)

    def forward(self, h, edge_index, edge_attr, batch):
        #h = self.embedding_h(x)
        #hidden_rep = [h]
        if self.training:
            for l, layer in enumerate(self.ginlayers):
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
            for l, layer in enumerate(self.ginlayers):
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

        # return score_over_layer