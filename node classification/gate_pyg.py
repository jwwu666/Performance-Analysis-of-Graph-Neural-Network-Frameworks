import os.path as osp
import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.nn as nn
from gate import Gated_PYG

t0 = time.time()

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset,transform=T.NormalizeFeatures())
data = dataset[0]

print("data load time: {:.4f}\n".format(time.time()-t0))

'''
gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
data = gdc(data)
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("cuda: \n",torch.cuda.is_available())

in_feats, n_classes = dataset.num_features,dataset.num_classes
model, data = Gated_PYG(in_feats, 64, n_classes, 0.5).to(device), data.to(device)

total_param = 0
print("MODEL DETAILS:\n")
# print(model)
for param in model.parameters():
    # print(param.data.size())
    total_param += np.prod(list(param.data.size()))
print('MODEL/Total parameters:',total_param)

optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4, lr=0.01)

x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(x, edge_index)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(x, edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

dur = []
#sd = []
best_val_acc = test_acc = 0
for epoch in range(1, 201):
    if epoch >= 3:
        t1 = time.time()

    model.train()
    loss = F.nll_loss(model(x, edge_index)[data.train_mask], data.y[data.train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t1)
    train_acc, val_acc, test_acc = test()
    #sd.append(test_acc)
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Time: {:.4f}'
    print(log.format(epoch, train_acc, val_acc, test_acc, np.mean(dur)))

print("total time: {:.4f}\n".format(time.time()-t0))
#print("s.d.: {:.4f}\n".format(np.std(sd)))