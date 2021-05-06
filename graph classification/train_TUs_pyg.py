import torch.nn.functional as F
from torch_geometric.utils import degree
import torch
import time

def accuracy(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    return acc

def train_pyg(model_name, model, train_loader, device, optimizer):
    t0 = time.time()
    model.train()
    n_data = 0
    loss_all = 0
    epoch_train_acc = 0
    t4 = time.time()
    t2 = time.time()
    for data in train_loader:
        print("data load time:", time.time() - t2)
        t7 = time.time()
        data = data.to(device)
        if model_name == 'monet':
            row, col = data.edge_index
            deg = degree(col, data.num_nodes)
            data.edge_attr = torch.stack([1 / torch.sqrt(deg[row]), 1 / torch.sqrt(deg[col])], dim=-1)
        optimizer.zero_grad()

        t9 = time.time()
        output = model.forward(data.x, data.edge_index, data.edge_attr, data.batch)
        print("forward:", time.time() - t9)

        loss = F.nll_loss(output, data.y)

        t3 = time.time()
        loss.backward()
        print("backward:", time.time() - t3)

        optimizer.step()
        loss_all += loss.item() * data.num_graphs
        n_data += data.y.size(0)
        epoch_train_acc += accuracy(output, data.y)

        print("a batch:", time.time() - t7)
        t2 = time.time()
    epoch_train_acc /= n_data
    print("train:", time.time() - t0)
    return loss_all / n_data, epoch_train_acc, optimizer


def test_pyg(model_name, model, test_loader, device):
    model.eval()
    epoch_test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            if model_name == 'monet':
                row, col = data.edge_index
                deg = degree(col, data.num_nodes)
                data.edge_attr = torch.stack([1 / torch.sqrt(deg[row]), 1 / torch.sqrt(deg[col])], dim=-1)
            output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = F.nll_loss(output, data.y)
            epoch_test_loss += loss.detach().item()
            pred = output.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
    return epoch_test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)