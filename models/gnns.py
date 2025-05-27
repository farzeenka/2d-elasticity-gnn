import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class SimpleGNN(torch.nn.Module):
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats)
        self.conv2 = SAGEConv(hidden_feats, hidden_feats)
        self.lin   = torch.nn.Linear(hidden_feats, out_feats)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)
