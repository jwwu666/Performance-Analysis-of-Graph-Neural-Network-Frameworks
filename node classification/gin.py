import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from dgl.nn.pytorch import GINConv as GIN_DGLConv
from torch_geometric.nn import GINConv as GIN_PYGConv
import torch.nn.functional as F
import time

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
    def __init__(self, input_dim, n_classes, hidden_dim = 96,
                 neighbor_pooling_type = "sum"):

        super(GIN_DGL, self).__init__()
        #self.learn_eps = learn_eps

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, n_classes)
        for layer in range(2):
            if layer == 0:
                mlp = MLP(1, hidden_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(1, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GIN_DGLConv(ApplyNodeFunc(mlp), neighbor_pooling_type))

        self.batch_norms = nn.BatchNorm1d(hidden_dim)

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        self.linears_prediction.append(nn.Linear(hidden_dim, n_classes))
        self.linears_prediction.append(nn.Linear(hidden_dim, n_classes))
        self.linears_prediction.append(nn.Linear(hidden_dim, n_classes))

        self.drop = nn.Dropout(0.5)


    def forward(self, g, h):
        # list of hidden representation at each layer (including input)
        h = self.embedding_h(h)

        hidden_rep = [h]

        for i in range(2):
            t = time.time()
            h = self.ginlayers[i](g, h)
            print( i, time.time() - t)
            # h = self.batch_norms(h)
            h = self.drop(h)
            # hidden_rep.append(h)

        score_over_layer = 0
        h = self.readout(h)
        # for i, h in enumerate(hidden_rep):
        #     score_over_layer += self.linears_prediction[i](h)
            #self.batch_norms(score_over_layer)

        #return score_over_layer
        return F.log_softmax(h, dim=1)


class GIN_PYG(torch.nn.Module):
    def __init__(self,in_feats,n_classes):
        super(GIN_PYG, self).__init__()

        #num_features = dataset.num_features
        hidden_dim = 96

        self.ginlayers = torch.nn.ModuleList()
        self.embedding_h = nn.Linear(in_feats, hidden_dim)

        for layer in range(2):
            if layer == 0:
                mlp = MLP(1, hidden_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(1, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GIN_PYGConv(ApplyNodeFunc(mlp), 0))   #ApplyNodeFunc(mlp)

        self.batch_norms = nn.BatchNorm1d(hidden_dim)
        self.readout = nn.Linear(hidden_dim, n_classes)

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        self.linears_prediction.append(nn.Linear(hidden_dim, n_classes))
        self.linears_prediction.append(nn.Linear(hidden_dim, n_classes))
        self.linears_prediction.append(nn.Linear(hidden_dim, n_classes))

        self.drop = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        h = self.embedding_h(x)
        hidden_rep = [h]

        for i in range(2):
            t = time.time()
            h = self.ginlayers[i](h, edge_index)
            print(i, time.time() - t)
            # h = self.batch_norms(h)
            h = self.drop(h)
            # hidden_rep.append(h)

        score_over_layer = 0

        # for i, h in enumerate(hidden_rep):
        #     score_over_layer += self.linears_prediction[i](h)
        #     # self.batch_norms(score_over_layer)

        # return score_over_layer
        h = self.readout(h)
        return F.log_softmax(h, dim=1)