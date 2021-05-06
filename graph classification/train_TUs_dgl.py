import torch.nn.functional as F
import torch
import time
import numpy as np
import gc
import dgl
def compute_pseudo(edges):
    # compute pseudo edge features for MoNet
    # to avoid zero division in case in_degree is 0, we add constant '1' in all node degrees denoting self-loop
    srcs = 1 / np.sqrt(edges.src['deg'] + 1)
    dsts = 1 / np.sqrt(edges.dst['deg'] + 1)
    pseudo = torch.cat((srcs.unsqueeze(-1), dsts.unsqueeze(-1)), dim=1)
    return {'pseudo': pseudo}

def accuracy(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    return acc


def train_dgl(model_name, model, optimizer, device, data_loader):
    t8 = time.time()
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        t7 = time.time()
        if model_name != 'monet':
            batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        #batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        t9 = time.time()
        batch_scores = model.forward(batch_graphs, batch_x)
        print("forward:", time.time() - t9)
        loss = F.nll_loss(batch_scores, batch_labels)
        t = time.perf_counter()
        loss.backward()
        print("backward:", time.perf_counter() - t)
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
        print("a batch:", time.time() - t7)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    print("train:", time.time() - t8)
    return epoch_loss, epoch_train_acc, optimizer

def test_dgl(model_name, model, device, test_loader):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(test_loader):
            if model_name != 'monet':
                batch_graphs = batch_graphs.to(device)
            else:
                batch_graphs.ndata['deg'] = batch_graphs.in_degrees()
                batch_graphs.apply_edges(compute_pseudo)
                batch_graphs = batch_graphs.to(device)
                pseudo = batch_graphs.edata['pseudo'].float()
            # if model_name == 'gate':
            #     e = batch_graphs.edata['feat']
            batch_x = batch_graphs.ndata['feat']#.to(device)
            #batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)
            if model_name == 'gate':
                batch_scores = model(batch_graphs)
            elif model_name == 'monet':
                batch_scores = model.forward(batch_graphs, batch_x, pseudo)
            else:
                 batch_scores = model(batch_graphs, batch_x)
            loss = F.nll_loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            pred =  batch_scores.max(dim=1)[1]
            epoch_test_acc += pred.eq(batch_labels).sum().item()
            #epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
            # gc.collect()
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data

    return epoch_test_loss, epoch_test_acc