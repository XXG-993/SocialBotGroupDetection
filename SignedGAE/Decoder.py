import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
import math


class InnerProductDecoder(nn.Module):

    def __init__(self, in_channels, hidden_channels, output_channels, num_layers=2):
        super(InnerProductDecoder, self).__init__()
            
        self.nodes_lin_start = nn.Linear(in_channels, hidden_channels)
        self.nodes_lin_end = nn.Linear(hidden_channels, output_channels)
        self.nodes_lin_list = nn.ModuleList()
        for _ in range(num_layers):
            self.nodes_lin_list.append(nn.Linear(hidden_channels, hidden_channels))
        
        
    def forward(self, adj, z, z_pos, z_neg, sigmoid=True):
        
        adj = torch.where(adj>0, 1.0, 0.0).to(torch.float32)
        value_pos = torch.mm(z_pos, z_pos.T)
        
        value_neg = torch.mm(z_neg, z_neg.T)
        
        attr = self.nodes_lin_start(z)
        for block in self.nodes_lin_list:
            attr = block(attr)
        attr = self.nodes_lin_end(attr)
        
        return torch.sigmoid(value_pos) if sigmoid else value_pos, torch.sigmoid(value_neg) if sigmoid else value_neg, attr