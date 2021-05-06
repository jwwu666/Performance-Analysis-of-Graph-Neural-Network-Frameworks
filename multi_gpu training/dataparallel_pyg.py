import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import MNISTSuperpixels, GNNBenchmarkDataset
from torch_geometric.data import DataListLoader
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv, global_mean_pool, DataParallel
from MNIST.gcn import GCN_PYG
from MNIST.gat import GAT_PYG
import time
from torch_geometric.utils import degree
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(1)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MNIST')
train_dataset = GNNBenchmarkDataset(path, "MNIST").shuffle()
test_dataset = GNNBenchmarkDataset(path, "MNIST", "test").shuffle()

train_loader = DataListLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataListLoader(test_dataset, batch_size=512, shuffle=True)

def accuracy(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    return acc

def train_pyg(model_name, model, train_loader, device, optimizer):
    global epoch_load_time, epoch_forward_time, epoch_backward_time, epoch_batch_time
    t10 = time.time()
    model.train()
    n_data = 0
    loss_all = 0
    epoch_train_acc = 0
    t4 = time.time()
    t2 = time.time()
    for data in train_loader:
        # epoch_load_time = epoch_load_time + time.time() - t2
        # print("data load time:", time.time() - t2)
        # t7 = time.time()
        #data = data.to(device)
        # print("data to gpu:", time.time() - t7)
        if model_name == 'monet':
            row, col = data.edge_index
            deg = degree(col, data.num_nodes)
            data.edge_attr = torch.stack([1 / torch.sqrt(deg[row]), 1 / torch.sqrt(deg[col])], dim=-1)
        optimizer.zero_grad()

        # t9 = time.time()
        output = model(data)
        # epoch_forward_time = epoch_forward_time + time.time() - t9
        # print("forward:", time.time() - t9)
        #print('Outside Model: num graphs: {}'.format(output.size(0)))
        y = torch.cat([data.y for data in data]).to(output.device)
        loss = F.nll_loss(output, y)

        # t3 = time.time()
        loss.backward()
        # epoch_backward_time = epoch_backward_time + time.time() - t3
        # print("backward:", time.time() - t3)

        optimizer.step()
        loss_all += loss.item() * output.size(0)
        n_data += y.size(0)
        epoch_train_acc += accuracy(output, y)
        #epoch_batch_time = epoch_batch_time + time.time() -t7
        #print("a batch:", time.time() - t7)
        #t2 = time.time()
    epoch_train_acc /= n_data
    print("train:", time.time() - t10)
    return loss_all / n_data, epoch_train_acc, optimizer

def test_pyg(model_name, model, test_loader, device):
    model.eval()
    epoch_test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            if model_name == 'monet':
                row, col = data.edge_index
                deg = degree(col, data.num_nodes)
                data.edge_attr = torch.stack([1 / torch.sqrt(deg[row]), 1 / torch.sqrt(deg[col])], dim=-1)
            output = model(data)
            y = torch.cat([data.y for data in data]).to(output.device)
            loss = F.nll_loss(output, y)
            epoch_test_loss += loss.detach().item()
            pred = output.max(dim=1)[1]
            correct += pred.eq(y).sum().item()
    return epoch_test_loss / len(test_dataset), correct / len(test_dataset)


t0 = time.time()

num_features = train_dataset.num_features
n_classes = train_dataset.num_classes
#model = GCN_PYG(num_features, 96, n_classes, 0)
model = GAT_PYG(num_features, n_classes, 32, 8, 0.5)
print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
model = DataParallel(model, [1])
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay = 0)
scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=0.5,
                                                             patience=25,
                                                             verbose=True)

dur = []
for epoch in range(1, 201):
    t1 = time.time()

    train_loss, train_acc, optimizer= train_pyg('gcn', model, train_loader, device, optimizer)

    dur.append(time.time() - t1)

    # print('Time: {:.4f}, Time: {:.4f}, Time: {:.4f}, Time: {:.4f}'.format(epoch_load_time, epoch_forward_time, epoch_backward_time, epoch_batch_time))

    # epoch_load_time, epoch_forward_time, epoch_backward_time, epoch_batch_time = 0, 0, 0, 0

    test_loss, test_acc = test_pyg('gcn', model, test_loader, device)

    scheduler.step(test_loss)

    print('Epoch: {:03d}, Train Loss: {:.4f}, '
          'Train Acc: {:.4f}, Test Acc: {:.4f}, Time: {:.4f}'.format(epoch, train_loss,
                                                                     train_acc, test_acc, np.mean(dur)))
print('Total Time: {:.4f}'.format(time.time()-t0))