import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
import math

class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        torch.nn.init.xavier_uniform_(self.weight)
        # self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, adj, input, active=True):

        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        
        if self.bias is not None:
            output = output + self.bias

        if active:
            output = F.relu(output)
        return output
        
        