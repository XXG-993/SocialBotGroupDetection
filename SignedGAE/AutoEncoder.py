import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from sgcn import SignedGraphConvolutionalNetwork, SignedSAGEConvolutionBase, SignedSAGEConvolutionDeep
from Decoder import InnerProductDecoder
from ReadConfig import ReadConfig
import pickle
import networkx as nx
import copy
from GCNLayer import GCNLayer



class SGCNAE(nn.Module):
    
    def __init__(self, X):
        super().__init__()
        
        self.cfg = ReadConfig().read_config()
        self.input_size = self.cfg["input_dim"] 
        self.hidden_size = self.cfg["hidden_dim"]
        self.output_size = self.cfg["output_dim"]
        self.n_cluster = self.cfg["n_cluster"]
        self.neurons = [self.hidden_size, self.hidden_size, self.output_size]
        
        self.layers = self.cfg["layers"]
        self.dropout = nn.Dropout(self.cfg["drop_out"])
        # self.activation = nn.ReLU()
        
        self.X = X
        self.__encoder = SignedGraphConvolutionalNetwork(self.X,
                                                         self.neurons
                                                        )
        
        self.__decoder = InnerProductDecoder(2*self.output_size,
                                               self.hidden_size,
                                               self.input_size)
        
        self.skip = Linear(2*self.output_size,2*self.output_size)
        self.gnn_cluster1 = GCNLayer(2*self.output_size,2*self.output_size)
        self.gnn_cluster2 = GCNLayer(2*self.output_size,2*self.output_size)
        self.classifier = Linear(2*self.output_size, self.n_cluster)
  
    def forward(self, positive_edges, negative_edges, matrix, co_matrix):

        h, pos_h, neg_h = self.__encoder(positive_edges, negative_edges)
        encoded = torch.clone(h)
        
        z = self.gnn_cluster1(co_matrix, encoded, active=False)
        z0 = F.selu(z + self.skip(encoded))
        z_ = self.gnn_cluster2(co_matrix, z0, active=False)
        z_ = F.selu(z_ + self.skip(z0))
        pred = self.classifier(z_)
        pred = F.softmax(pred, dim=1)
        
        value_pos, value_neg, attr = self.__decoder(matrix, h, pos_h, neg_h)
                
        return z_, value_pos, value_neg, attr, pred