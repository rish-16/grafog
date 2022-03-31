import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as tg
import torch_geometric.nn as tgnn

class GCN(nn.Module):
    def __init__(self, inchannels, hidden, outchannels):
        super().__init__()
        self.conv1 = tgnn.GCNConv(inchannels, hidden)
        self.conv2 = tgnn.GCNConv(hidden, hidden)
        self.conv3 = tgnn.GCNConv(hidden, hidden)
        self.conv4 = tgnn.GCNConv(hidden, outchannels)

    def forward(self, x, edge_idx):
        x = torch.relu(self.conv1(x, edge_idx))
        x = torch.relu(self.conv2(x, edge_idx))
        x = torch.relu(self.conv3(x, edge_idx))
        out = torch.softmax(self.conv4(x, edge_idx), 1)
        return out

class GAT(nn.Module):
    def __init__(self, inchannels, hidden, outchannels):
        super().__init__()
        self.conv1 = tgnn.GATConv(inchannels, hidden)
        self.conv2 = tgnn.GATConv(hidden, hidden)
        self.conv3 = tgnn.GATConv(hidden, hidden)
        self.conv4 = tgnn.GATConv(hidden, outchannels)

    def forward(self, x, edge_idx):
        x = torch.relu(self.conv1(x, edge_idx))
        x = torch.relu(self.conv2(x, edge_idx))
        x = torch.relu(self.conv3(x, edge_idx))
        out = torch.softmax(self.conv4(x, edge_idx), 1)
        return out