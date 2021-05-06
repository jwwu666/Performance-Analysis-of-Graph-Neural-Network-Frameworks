import os.path as osp
import os
import time
#from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from MNIST.dgl_listloader import DataListLoader
from graph import train_TUs_dgl
#from MNIST.DataParallel import DataParallel
#os.environ['CUDA_VISIBLE_DEVICES'] = ' 2 '

import torch
import numpy as np
from torch_geometric.datasets import MNISTSuperpixels, GNNBenchmarkDataset
import torch.nn.functional as F
import dgl
import torch
import torch.nn as nn
# import torch.nn.DataParallel as DataParallel
import time
from gcn import GCN_DGL
from dgl import save_graphs, load_graphs
from train_dgl import test_dgl, train_dgl
#from graph\dgl_data import GraphDataLoader
#from train_TUs_pyg import train_pyg, test_pyg
from MNIST.gat import GAT_DGL

epoch_load_time = 0
epoch_batch_time = 0
epoch_forward_time = 0
epoch_backward_time = 0
epoch_update_time = 0

device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.graph_labels = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])

def collate(samples):
    # The input `samples` is a list of pairs (graph, label).
    graphs, labels = map(list, zip(*samples))
    for g in graphs:
        # deal with node feats
        for feat in g.node_attr_schemes().keys():
            # TODO torch.Tensor is not recommended
            # torch.DoubleTensor and torch.tensor
            # will meet error in executor.py@runtime line 472, tensor.py@backend line 147
            # RuntimeError: expected type torch.cuda.DoubleTensor but got torch.cuda.FloatTensor
            g.ndata['feat'] = g.ndata['feat'].float()
        # no edge feats
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    return batched_graph, labels

def accuracy(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    return acc

def train_dgl(model_name, model, optimizer, d, data_loader):
    global epoch_load_time, epoch_forward_time, epoch_backward_time, epoch_batch_time, batch_time, epoch_update_time
    t10 = time.time()
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    t4 = time.time()
    t2 = time.perf_counter()
    batch_time = 0
    #for data in data_loader:
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        # graphs = [data[i][0] for i in range(len(data))]
        # labels = [data[i][1] for i in range(len(data))]

        # print("batch_time: ", batch_time)
        # epoch_load_time = epoch_load_time + time.perf_counter() - t2
        # print("data load time:", time.perf_counter() - t2)
        #total_load.append(time.perf_counter() - t2)
        # t7 = time.perf_counter()
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels
        #pseudo = batch_graphs.edata['pseudo'].to(device).float()
        #batch_e = batch_graphs.edata['feat'].to(device)
        optimizer.zero_grad()

        # t9 = time.perf_counter()
        output = model(batch_graphs)
        #y = torch.cat(labels).to(output.device)
        #batch_scores = model.forward(batch_graphs, batch_x)
        # epoch_forward_time = epoch_forward_time + time.perf_counter() - t9
        # print("forward:", time.perf_counter() - t9)
        #total_forward.append(time.perf_counter() - t9)

        loss = F.nll_loss(output, batch_labels)

        # t3 = time.perf_counter()
        loss.backward()
        # epoch_backward_time = epoch_backward_time + time.perf_counter() - t3
        # print("backward:", time.perf_counter() - t3)
        #total_backward.append(time.perf_counter() - t3)

        # t5 = time.perf_counter()
        optimizer.step()
        # print("update:", time.perf_counter() - t5)
        # epoch_update_time = epoch_update_time + time.perf_counter() - t5
        #total_update.append(time.perf_counter() - t5)

        epoch_loss += loss.item()
        epoch_train_acc += accuracy(output, batch_labels)
        nb_data += len(batch_labels)
        # epoch_batch_time = epoch_batch_time + time.perf_counter() - t7
        # print("a batch:", time.perf_counter() - t7)
        # t2 = time.perf_counter()
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    # print("train:", time.time() - t10)
    return epoch_loss, epoch_train_acc, optimizer

def test_dgl(model_name, model, d, test_loader):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(test_loader):
            # if model_name != 'monet':
            #     batch_graphs = batch_graphs.to(device)
            # else:
            #     batch_graphs.ndata['deg'] = batch_graphs.in_degrees()
            #     batch_graphs.apply_edges(compute_pseudo)
            #     # pseudo = batch_graphs.edata['pseudo'].to(device).float()
            #     batch_graphs = batch_graphs.to(device)
            # graphs = [data[i][0] for i in range(len(data))]
            # labels = [data[i][1] for i in range(len(data))]
            batch_graphs = batch_graphs.to(device)
            batch_labels = batch_labels.to(device)
            output = model(batch_graphs)
            #y = torch.cat(labels).to(output.device)

            loss = F.nll_loss(output, batch_labels)
            epoch_test_loss += loss.item()
            pred =  output.max(dim=1)[1]
            epoch_test_acc += pred.eq(batch_labels).sum().item()
            #epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data

    return epoch_test_loss, epoch_test_acc

t0 = time.time()

path = osp.join('/public-ssd/wujw')
test_path = os.path.join(path, 'mnist_75sp_test.pkl')

graph_lists, label = load_graphs(str(test_path))
graph_labels = []
for i in range(len(label)):
            graph_labels.append(label['{}'.format(i)])

test_dataset = DGLFormDataset(graph_lists, graph_labels)

train_path = os.path.join(path, 'mnist_75sp_train.pkl')
graph_lists, label = load_graphs(str(train_path))
graph_labels = []
for i in range(len(label)):
            graph_labels.append(label['{}'.format(i)])

train_dataset = DGLFormDataset(graph_lists, graph_labels)

train_dataset.graph_lists = [dgl.add_self_loop(g) for g in test_dataset.graph_lists]
test_dataset.graph_lists = [dgl.add_self_loop(g) for g in test_dataset.graph_lists]

print("load data time : {:.4f}".format(time.time() - t0))

# train_loader = DataListLoader(train_dataset, batch_size=4096, shuffle=True)
# test_loader = DataListLoader(test_dataset, batch_size=4096, shuffle=True)

train_loader = DataLoader(train_dataset, batch_size=512, collate_fn= collate, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, collate_fn= collate, shuffle=True)

num_features = test_dataset.graph_lists[0].ndata['feat'][0].shape[0]
n_classes = len(np.unique(np.array(test_dataset[:][1])))

print(n_classes, num_features)

#device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
#model = GCN_DGL(num_features, 96, n_classes, 0)
model = GAT_DGL(num_features, n_classes, 32, 8, 0.5)
model = nn.DataParallel(model)
model = model.to(device)

# total_param = 0
# print("MODEL DETAILS:\n")
# for param in model.parameters():
#     total_param += np.prod(list(param.data.size()))
# print('Total parameters:', total_param)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay = 0)
scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=0.5,
                                                             patience=25,
                                                             verbose=True)


dur = []
for epoch in range(1, 201):
    t1 = time.time()

    train_loss, train_acc, optimizer= train_dgl('gcn', model, optimizer, device, train_loader)

    dur.append(time.time() - t1)

    print('Time: {:.4f}, Time: {:.4f}, Time: {:.4f}, Time: {:.4f}'.format(epoch_load_time, epoch_forward_time, epoch_backward_time, epoch_batch_time))

    epoch_load_time, epoch_forward_time, epoch_backward_time, epoch_batch_time = 0, 0, 0, 0

    test_loss, test_acc = test_dgl('gcn', model, device, test_loader)

    scheduler.step(test_loss)

    print('Epoch: {:03d}, Train Loss: {:.4f}, '
          'Train Acc: {:.4f}, Test Acc: {:.4f}, Time: {:.4f}'.format(epoch, train_loss,
                                                                     train_acc, test_acc, np.mean(dur)))
print('Total Time: {:.4f}'.format(time.time()-t0))