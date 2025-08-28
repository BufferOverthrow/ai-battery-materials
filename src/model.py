from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import NNConv, global_mean_pool

class EdgeMLP(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_channels)
        )

    def forward(self, edge_attr):
        return self.net(edge_attr)

class GNNRegressor(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 128, layers: int = 3):
        super().__init__()
        self.proj = nn.Linear(node_dim, hidden)
        convs = []
        self.edge_mlps = nn.ModuleList()

        for _ in range(layers):
            edge_mlp = EdgeMLP(edge_dim, hidden, hidden * hidden)
            self.edge_mlps.append(edge_mlp)
            convs.append(NNConv(hidden, hidden, edge_mlp, aggr='mean'))
        self.convs = nn.ModuleList(convs)

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.proj(x)
        for conv, _mlp in zip(self.convs, self.edge_mlps):
            h = F.silu(conv(h, edge_index, edge_attr))  # NNConv uses edge_mlp internally
        hg = global_mean_pool(h, batch)
        out = self.head(hg)
        return out.view(-1)
