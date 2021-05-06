import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from gate import Gated_DGL

t0 = time.time()

data = CoraGraphDataset(force_reload=True,verbose=False)
g = data[0]

print("data load time: {:.4f}\n".format(time.time()-t0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("cuda: \n",torch.cuda.is_available())

g = g.int().to(device)

features = g.ndata['feat']
labels = g.ndata['label']
train_mask = g.ndata['train_mask']
val_mask = g.ndata['val_mask']
test_mask = g.ndata['test_mask']
print(len(train_mask), len(val_mask), len(test_mask))
in_feats = features.shape[1]
n_classes = data.num_labels
n_edges = data.graph.number_of_edges()

'''
print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.int().sum().item(),
              val_mask.int().sum().item(),
              test_mask.int().sum().item()))
'''
'''
 # add self loop
#if args.self_loop:
g = dgl.remove_self_loop(g)
g = dgl.add_self_loop(g)
n_edges = g.number_of_edges()

# normalization
degs = g.in_degrees().float()
norm = torch.pow(degs, -0.5)
norm[torch.isinf(norm)] = 0
norm = norm.to(device)
g.ndata['norm'] = norm.unsqueeze(1)
'''
#g = g.remove_self_loop().add_self_loop()
n_edges = g.number_of_edges()
pseudo = torch.zeros([n_edges,1])
pseudo = pseudo.to(device)
'''
us, vs = g.edges(order='eid')
udeg, vdeg = 1 / torch.sqrt(g.in_degrees(us).float()), 1 / torch.sqrt(g.in_degrees(vs).float())
pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1)
print(pseudo.size())
'''

# create model
model = Gated_DGL(in_feats,
            64,
            n_classes,
            0.5,)

model.to(device)

total_param = 0
print("MODEL DETAILS:\n")
# print(model)
for param in model.parameters():
    # print(param.data.size())
    total_param += np.prod(list(param.data.size()))
print('MODEL/Total parameters:',total_param)

@torch.no_grad()
def evaluate():
    model.eval()
    accs = []
    logits = model(g, features, pseudo)
    #for mask in (train_mask, val_mask, test_mask):
    pred = logits[train_mask].max(1)[1]
    acc = pred.eq(labels[train_mask]).sum().item() / train_mask.sum().item()
    accs.append(acc)
    pred = logits[val_mask].max(1)[1]
    acc = pred.eq(labels[val_mask]).sum().item() / val_mask.sum().item()
    accs.append(acc)
    pred = logits[test_mask].max(1)[1]
    acc = pred.eq(labels[test_mask]).sum().item() / test_mask.sum().item()
    accs.append(acc)
    return accs

# use optimizer
optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.01,
                                 weight_decay=5e-4)

 # initialize graph
dur = []
best_val_acc = test_acc = 0
for epoch in range(200):
    model.train()
    if epoch >= 3:
        t1 = time.time()
    # forward
    logits = model(g, features, pseudo)
    loss = F.nll_loss(logits[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
         dur.append(time.time() - t1)


    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Time: {:.4f}'
    train_acc, val_acc, tmp_test_acc = evaluate()
    print(log.format(epoch, train_acc, val_acc, tmp_test_acc, np.mean(dur)))

print("total time: {:.4f}\n".format(time.time()-t0))