
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from torch_geometric.nn import GCNConv,  global_mean_pool
import torch.nn.functional as F
from dgl.nn.pytorch.glob import AvgPooling
from MLP import MLPReadout
import time
from torch.autograd import Variable
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

class GCN_DGL(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 dropout,):
        super(GCN_DGL, self).__init__()
        self.layers = nn.ModuleList()
        self.embedding_h = nn.Linear(in_feats, n_hidden)
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, allow_zero_in_degree = True))#, activation=activation))
        for _ in range(3):
            self.layers.append(GraphConv(n_hidden, n_hidden, allow_zero_in_degree = True))#, activation=activation))
        # output layer
        #self.layers.append(GraphConv(n_hidden, n_hidden))

        self.dropout = nn.Dropout(p=dropout)

        self.MLP = MLPReadout(n_hidden, n_classes)

        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_classes)

        self.bn = nn.BatchNorm1d(n_hidden)

        self.avgpooling = AvgPooling()

    def forward(self, g, h):
        # if self.training:
        #     #t = time.time()
        #     for i, layer in enumerate(self.layers):
        #         torch.cuda.synchronize()
        #         t1 = time.perf_counter()
        #         h = layer(g, h)
        #         h.register_hook(hook_gcn)
        #         torch.cuda.synchronize()
        #         print("conv", i, "forward time: ", time.perf_counter() - t1)
        #         h = F.elu(h)
        #         h.register_hook(hook_relu)
        #         #h = self.bn(h)
        #         #h = self.dropout(h)
        #     #print(time.time() - t)
        #     torch.cuda.synchronize()
        #     t2 = time.perf_counter()
        #     h = self.avgpooling(g, h)
        #     h.register_hook(hook_pool)
        #     torch.cuda.synchronize()
        #     print("pooling forward time: ", time.perf_counter() - t2)
        #
        #     torch.cuda.synchronize()
        #     t3 = time.perf_counter()
        #     h = self.fc1(h)
        #     h.register_hook(hook)
        #     torch.cuda.synchronize()
        #     print("fc1 forward time: ", time.perf_counter() - t3)
        #     h = F.elu(h)
        #     h.register_hook(hook_relu)
        #     t4 = time.perf_counter()
        #     h = self.fc2(h)
        #     h.register_hook(hook)
        #     torch.cuda.synchronize()
        #     print("fc2 forward time: ", time.perf_counter() - t4)
        #     #h = self.MLP(h)
        #     h = F.log_softmax(h, dim=1)
        #     h.register_hook(hook)
        # else:
        t = time.perf_counter()
        for i, layer in enumerate(self.layers):
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            h = layer(g, h)
            torch.cuda.synchronize()
            print("conv", i, "forward time: ", time.perf_counter() - t1)
            h = F.elu(h)
            # h = self.bn(h)
            # h = self.dropout(h)
            # print(time.time() - t)

        t2 = time.perf_counter()
        h = self.avgpooling(g, h)
        torch.cuda.synchronize()
        print("pooling forward time: ", time.perf_counter() - t2)

        t3 = time.perf_counter()
        h = self.fc1(h)

        print("fc1 forward time: ", time.perf_counter() - t3)
        t4 = time.perf_counter()
        h = F.elu(h)
        print("relu: ", time.perf_counter() - t4)
        t4 = time.perf_counter()
        h = self.fc2(h)
        print("fc2 forward time: ", time.perf_counter() - t4)
        # h = self.MLP(h)
        t4 = time.perf_counter()
        h = F.log_softmax(h, dim=1)
        torch.cuda.synchronize()
        print(" model forward: ", time.perf_counter() - t)
        return h

class GCN_PYG(torch.nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, dropout):
        super(GCN_PYG, self).__init__()
        self.layers = nn.ModuleList()
        self.embedding_h = nn.Linear(in_feats, n_hidden)

        # input layer
        self.layers.append(GCNConv(in_feats, n_hidden))
        for _ in range(3):
            self.layers.append(GCNConv(n_hidden, n_hidden))
            #self.layers.append(F.elu())
        self.dropout = nn.Dropout(p=dropout)

        #self.pooling = global_mean_pool()
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_classes)

        #self.layers.append(self.fc1)
        #self.layers.append(self.fc2)

        self.MLP = MLPReadout(n_hidden, n_classes)
        self.bn = nn.BatchNorm1d(n_hidden)

    def forward(self, x, edge_index, edge_weight, batch):
        #self.output = []
        #x = Variable(x.data, requires_grad=True)
        #x.register_hook(hook)
        for i, layer in enumerate(self.layers):

            #t1 = time.perf_counter()
            x = layer(x, edge_index, edge_weight)
            #x.register_hook(hook_gcn)
            #self.output.append(x)
            #x = Variable(x.data, requires_grad=True)
            #x.register_hook(hook)
            #print("conv", i, "forward time: ", time.perf_counter() - t1)

            x = F.elu(x)
            #self.output.append(x)
            #x.register_hook(hook_relu)
            #x.register_hook(hook)
            #x = self.bn(x)
            x = self.dropout(x)

        #t2 = time.perf_counter()
        x = global_mean_pool(x, batch)
        #self.output.append(x)
        #x.register_hook(hook)
        #print("pooling forward time: ", time.perf_counter() - t2)


        #x = self.MLP(x)
        #t3 = time.perf_counter()
        x = self.fc1(x)
        #self.output.append(x)
        #x.register_hook(hook)
        #print("fc1 forward time: ", time.perf_counter() - t3)

        x = F.elu(x)
        #self.output.append(x)
        #x.register_hook(hook_relu)
        #t4 = time.perf_counter()

        x = self.fc2(x)
        #x.register_hook(hook)
        #self.output.append(x)
        #print("fc2 forward time: ", time.perf_counter() - t4)
        x = F.log_softmax(x, dim=1)
        #x.register_hook(hook)
        # self.output.append(x)
        # if self.training:
        #     for i in range(len(self.output)):
        #         self.output[i].register_hook(hook)
        return x



