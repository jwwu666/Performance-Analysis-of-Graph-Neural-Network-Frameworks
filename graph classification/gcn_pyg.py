import os.path as osp
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = ' 7 '
from torch_geometric.utils import degree

import torch.nn.functional as F
import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from gcn import GCN_PYG
from dgl_data import GraphDataLoader
from train_TUs_pyg import  test_pyg #train_pyg,
from torch.utils.data import IterableDataset
#from ptflops import get_model_complexity_info
epoch_flag = 0
epoch_load_time = 0
epoch_batch_time = 0
epoch_forward_time = 0
epoch_backward_time = 0
epoch_update_time = 0
def accuracy(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    return acc

# def hook(grad):
#     print("loss:", time.perf_counter())

def train_pyg(model_name, model, train_loader, device, optimizer):
    global epoch_load_time, epoch_forward_time, epoch_backward_time, epoch_batch_time, epoch_flag, epoch_update_time
    t10 = time.perf_counter()
    model.train()
    n_data = 0
    loss_all = 0
    epoch_train_acc = 0
    t4 = time.perf_counter()
    t2 = time.perf_counter()
    #with torch.autograd.profiler.profile() as prof2:
    #print(prof2)

    for data in train_loader:
        torch.cuda.synchronize()
        epoch_load_time = epoch_load_time + time.perf_counter() - t2
        print("data load time:", time.perf_counter() - t2)
        t7 = time.perf_counter()
        data = data.to(device)
        torch.cuda.synchronize()
        print("data to gpu:", time.perf_counter() - t7)
        if model_name == 'monet':
            row, col = data.edge_index
            deg = degree(col, data.num_nodes)
            data.edge_attr = torch.stack([1 / torch.sqrt(deg[row]), 1 / torch.sqrt(deg[col])], dim=-1)
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t9 = time.perf_counter()
        #print(data.x.size(), )
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        torch.cuda.synchronize()
        epoch_forward_time = epoch_forward_time + time.perf_counter() - t9
        print("forward:", time.perf_counter() - t9)

        t6 = time.perf_counter()
        loss = F.nll_loss(output, data.y)
        torch.cuda.synchronize()
        print("Compute loss:", time.perf_counter() - t6)

        t3 = time.perf_counter()
        #loss.register_hook(print)
        #loss.register_hook(hook)
        loss.backward()
        torch.cuda.synchronize()
        epoch_backward_time = epoch_backward_time + time.perf_counter() - t3
        print("backward:", time.perf_counter() - t3)
        #print(prof)

        t5 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        epoch_update_time = epoch_update_time + time.perf_counter() - t5
        print("update:", time.perf_counter() - t5)

        loss_all += loss.item() * data.num_graphs
        n_data += data.y.size(0)
        epoch_train_acc += accuracy(output, data.y)
        torch.cuda.synchronize()
        epoch_batch_time = epoch_batch_time + time.perf_counter() -t7
        print("a batch:", time.perf_counter() - t7)
        t2 = time.perf_counter()
    epoch_train_acc /= n_data
    print("train:", time.perf_counter() - t10)
    loss_all = loss_all / n_data
    return loss_all, epoch_train_acc, optimizer


t0 = time.time()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'DD')
dataset = TUDataset(path, name='DD', use_node_attr = True)#.shuffle()

print("load data time : {:.4f}".format(time.time() - t0))

train_loader, valid_loader, test_loader = GraphDataLoader('pyg', dataset, batch_size=256, device=torch.device('cuda'), seed=0, shuffle=True, split_name='fold10').train_valid_test_loader()


num_features = dataset.num_features
n_classes = dataset.num_classes

print(n_classes, num_features)

device = torch.device('cuda')

model = GCN_PYG(num_features, 128, n_classes, 0.0).to(device)

total_param = 0
print("MODEL DETAILS:\n")
for param in model.parameters():
    total_param += np.prod(list(param.data.size()))
print('Total parameters:', total_param)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay = 0)
scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=0.5,
                                                             patience=25,
                                                             verbose=True)

#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

dur = []
total_load = 0
total_fp = 0
total_bp = 0
total_up = 0
total_ot = 0
for epoch in range(1, 201):
    t1 = time.time()

    train_loss, train_acc, optimizer = train_pyg('gcn', model, train_loader, device, optimizer)

    dur.append(time.time() - t1)

    print('load Time: {:.4f},forward Time: {:.4f}, backward Time: {:.4f}, update Time: {:.4f}, batch Time: {:.4f}'.format(epoch_load_time, epoch_forward_time,
                                                                          epoch_backward_time, epoch_update_time, epoch_batch_time))
    if epoch > 50 and epoch < 151:
        total_load = total_load + epoch_load_time
        total_fp = total_fp + epoch_forward_time
        total_bp = total_bp + epoch_backward_time
        total_up = total_up + epoch_update_time
        total_ot = epoch_batch_time - epoch_forward_time - epoch_backward_time - epoch_update_time + total_ot
    epoch_load_time, epoch_forward_time, epoch_backward_time, epoch_batch_time, epoch_update_time = 0, 0, 0, 0, 0



    test_loss, test_acc = test_pyg('gcn', model, test_loader, device)

    scheduler.step(test_loss)

    print('Epoch: {:03d}, Train Loss: {:.4f}, '
          'Train Acc: {:.4f}, Test Acc: {:.4f}, Time: {:.4f}, real Time: {:.4f}'.format(epoch, train_loss,
                                                                     train_acc, test_acc, np.mean(dur), dur[epoch-1]))

print('load: {:04f}, forward: {:.4f}, '
          'backward: {:.4f}, update: {:.4f}, other: {:.4f}'.format(total_load/100, total_fp/100, total_bp/100, total_up/100, total_ot/100))

print('Total Time: {:.4f}'.format(time.time()-t0))
