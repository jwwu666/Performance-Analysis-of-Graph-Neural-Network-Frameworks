import os.path as osp
import numpy as np
import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
import torch.nn.functional as F
from dgl.data import LegacyTUDataset
from dgl_data import GraphDataLoader
from sage import SAGE_DGL
from train_TUs_dgl import train_dgl, test_dgl
import dgl
import time
import ctypes

import torch.nn.functional as F
epoch_load_time = 0
epoch_batch_time = 0
epoch_forward_time = 0
epoch_backward_time = 0
epoch_update_time = 0

total_load = 0
total_forward = 0
total_backward = 0
total_update = 0


def accuracy(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    return acc


# def train_dgl(model_name, model, optimizer, device, data_loader):
#     global epoch_load_time, epoch_forward_time, epoch_backward_time, epoch_batch_time, batch_time, epoch_update_time
#     t10 = time.time()
#     model.train()
#     epoch_loss = 0
#     epoch_train_acc = 0
#     nb_data = 0
#     t4 = time.time()
#     t2 = time.perf_counter()
#     batch_time = 0
#     print("batch_time: ", batch_time)
#     for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
#         print("batch_time: ", batch_time)
#         epoch_load_time = epoch_load_time + time.perf_counter() - t2
#         print("data load time:", time.perf_counter() - t2)
#         #total_load.append(time.perf_counter() - t2)
#         t7 = time.perf_counter()
#
#         #pseudo = batch_graphs.edata['pseudo'].to(device).float()
#         batch_graphs = batch_graphs.to(device)
#         batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
#         #batch_e = batch_graphs.edata['feat'].to(device)
#         batch_labels = batch_labels.to(device)
#
#         optimizer.zero_grad()
#
#         t9 = time.perf_counter()
#         batch_scores = model.forward(batch_graphs, batch_x)
#         #batch_scores = model.forward(batch_graphs, batch_x)
#         epoch_forward_time = epoch_forward_time + time.perf_counter() - t9
#         print("forward:", time.perf_counter() - t9)
#         #total_forward.append(time.perf_counter() - t9)
#
#         loss = F.nll_loss(batch_scores, batch_labels)
#
#         t3 = time.perf_counter()
#         loss.backward()
#         epoch_backward_time = epoch_backward_time + time.perf_counter() - t3
#         print("backward:", time.perf_counter() - t3)
#         #total_backward.append(time.perf_counter() - t3)
#
#         t5 = time.perf_counter()
#         optimizer.step()
#         print("update:", time.perf_counter() - t5)
#         epoch_update_time = epoch_update_time + time.perf_counter() - t5
#         #total_update.append(time.perf_counter() - t5)
#
#         epoch_loss += loss.detach().item()
#         epoch_train_acc += accuracy(batch_scores, batch_labels)
#         nb_data += batch_labels.size(0)
#         epoch_batch_time = epoch_batch_time + time.perf_counter() - t7
#         print("a batch:", time.perf_counter() - t7)
#         t2 = time.perf_counter()
#     epoch_loss /= (iter + 1)
#     epoch_train_acc /= nb_data
#     print("train:", time.time() - t10)
#     return epoch_loss, epoch_train_acc, optimizer

def train_dgl(model_name, model, optimizer, device, data_loader):
    global epoch_load_time, epoch_forward_time, epoch_backward_time, epoch_batch_time,  epoch_update_time
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0

    torch.cuda.synchronize()
    t2 = time.perf_counter()
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        torch.cuda.synchronize()
        end = time.perf_counter() - t2
        epoch_load_time = epoch_load_time + end
        print("data load time:", end)

        torch.cuda.synchronize()
        t7 = time.perf_counter()

        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat']
        batch_labels = batch_labels.to(device)

        torch.cuda.synchronize()
        end = time.perf_counter() - t7

        print("data to gpu:", end)


        optimizer.zero_grad()
        #print("model pre:", time.time() - t7)

        torch.cuda.synchronize()
        t9 = time.perf_counter()
        batch_scores = model.forward(batch_graphs, batch_x)
        torch.cuda.synchronize()
        end = time.perf_counter() - t9
        epoch_forward_time = epoch_forward_time + end
        print("forward:", end)

        torch.cuda.synchronize()
        t = time.perf_counter()
        loss = F.nll_loss(batch_scores, batch_labels)
        torch.cuda.synchronize()
        end = time.perf_counter() - t
        print(" computer loss:", end)

        torch.cuda.synchronize()
        t3 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        end = time.perf_counter() - t3
        epoch_backward_time = epoch_backward_time + end
        print("backward:", end)

        torch.cuda.synchronize()
        tt = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        end = time.perf_counter() - tt
        epoch_update_time = epoch_update_time + end
        print("update:", end)

        epoch_loss += loss.item()
        #epoch_train_acc += accuracy(batch_scores, batch_labels)
        pred = batch_scores.max(dim=1)[1]
        epoch_train_acc += pred.eq(batch_labels).sum().item()
        nb_data += batch_labels.size(0)

        torch.cuda.synchronize()
        end = time.perf_counter() - t7
        epoch_batch_time = epoch_batch_time + end
        print("a batch:", end)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    return epoch_loss, epoch_train_acc, optimizer

t0 =time.time()

dataset = LegacyTUDataset('DD')
print("load data time : {:.4f}".format(time.time() - t0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
for g in dataset.graph_lists:
    g = dgl.add_self_loop(g)
'''

#dataset.graph_lists = [dgl.add_self_loop(g) for g in dataset.graph_lists]

t2 = time.time()
train_loader, valid_loader, test_loader = GraphDataLoader('dgl',
        dataset, batch_size=256, device=torch.device('cuda'),
        seed=0, shuffle=True,
        split_name='fold10').train_valid_test_loader()

print("batch time:", time.time() - t2)

num_features = dataset.graph_lists[0].ndata['feat'][0].shape[0]
n_classes = dataset.num_labels
print(num_features, n_classes)

model = SAGE_DGL(num_features, 96, n_classes, 0.5, "mean", device).to(device)

total_param = 0
print("MODEL DETAILS:\n")
for param in model.parameters():
    total_param += np.prod(list(param.data.size()))
print('Total parameters:', total_param)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=0.5,
                                                             patience=25,
                                                             verbose=True)

dur = []
total_load = 0
total_fp = 0
total_bp = 0
total_up = 0
total_ot = 0
for epoch in range(1, 201):
    t1 = time.time()
    train_loss, train_acc, optimizer= train_dgl('sage', model,  optimizer, device, train_loader)
    dur.append(time.time() - t1)
    print(
        'load Time: {:.4f},forward Time: {:.4f}, backward Time: {:.4f}, update Time: {:.4f}, batch Time: {:.4f}'.format(
            epoch_load_time,
            epoch_forward_time,
            epoch_backward_time,
            epoch_update_time,
            epoch_batch_time))

    if epoch > 50 and epoch < 151:
        total_load = total_load + epoch_load_time
        total_fp = total_fp + epoch_forward_time
        total_bp = total_bp + epoch_backward_time
        total_up = total_up + epoch_update_time
        total_ot = epoch_batch_time - epoch_forward_time - epoch_backward_time - epoch_update_time + total_ot

    epoch_load_time, epoch_forward_time, epoch_backward_time, epoch_update_time, epoch_batch_time = 0, 0, 0, 0, 0
    test_loss, test_acc = test_dgl('sage', model, device, test_loader)
    scheduler.step(test_loss)
    print('Epoch: {:03d}, Train Loss: {:.4f}, '
          'Train Acc: {:.4f}, Test Acc: {:.4f}, Time: {:.4f}'.format(epoch, train_loss,
                                                                     train_acc, test_acc, np.mean(dur)))

print('load: {:04f}, forward: {:.4f}, '
          'backward: {:.4f}, update: {:.4f}, other: {:.4f}'.format(total_load/100, total_fp/100, total_bp/100, total_up/100, total_ot/100))


print('Total Time: {:.4f}'.format(time.time() - t0))