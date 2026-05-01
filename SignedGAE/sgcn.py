"""SGCN runner."""

import time
import torch
import random
import numpy as np
import pandas as pd
from tqdm import trange
import torch.nn.init as init
from torch.nn import Parameter
import torch.nn.functional as F
from utils_ import calculate_auc, setup_features
from sklearn.model_selection import train_test_split
from signedsageconvolution import SignedSAGEConvolutionBase, SignedSAGEConvolutionDeep
from signedsageconvolution import ListModule
from utils_ import structured_negative_sampling
from ReadConfig import ReadConfig

class SignedGraphConvolutionalNetwork(torch.nn.Module):

    def __init__(self, X, neurons, seed=123):
        super(SignedGraphConvolutionalNetwork, self).__init__()

        torch.manual_seed(seed)
        self.X = X
        self.neurons = neurons     
        self.setup_layers()

    def setup_layers(self):
        """
        Adding Base Layers, Deep Signed GraphSAGE layers.
        Assing Regression Parameters if the model is not a single layer model.
        """
        
        self.nodes = range(self.X.shape[0])
        self.layers = len(self.neurons)
        self.positive_base_aggregator = SignedSAGEConvolutionBase(self.X.shape[1]*2,
                                                                  self.neurons[0])
        self.negative_base_aggregator = SignedSAGEConvolutionBase(self.X.shape[1]*2,
                                                                  self.neurons[0])
        self.positive_aggregators = []
        self.negative_aggregators = []
        for i in range(1, self.layers):
            self.positive_aggregators.append(SignedSAGEConvolutionDeep(3*self.neurons[i-1],
                                                                       self.neurons[i]))

            self.negative_aggregators.append(SignedSAGEConvolutionDeep(3*self.neurons[i-1],
                                                                       self.neurons[i]))

        self.positive_aggregators = ListModule(*self.positive_aggregators)
        self.negative_aggregators = ListModule(*self.negative_aggregators)
        
        # self.regression_weights = Parameter(torch.Tensor(4*self.neurons[-1], 3))
        # self.regression_bias = Parameter(torch.FloatTensor(3))
        # init.xavier_normal_(self.regression_weights)
        # self.regression_bias.data.fill_(0.0)


    def forward(self, positive_edges, negative_edges):
        """
        Model forward propagation pass. Can fit deep and single layer SGCN models.
        :param positive_edges: Positive edges.
        :param negative_edges: Negative edges.
        :param target: Target vectors.
        :return loss: Loss value.
        :return self.z: Hidden vertex representations.
        """
        
        self.h_pos, self.h_neg = [], []
        self.h_pos.append(torch.tanh(self.positive_base_aggregator(self.X, positive_edges)))
        self.h_neg.append(torch.tanh(self.negative_base_aggregator(self.X, negative_edges)))
        for i in range(1, self.layers):
            self.h_pos.append(torch.tanh(self.positive_aggregators[i-1](self.h_pos[i-1], self.h_neg[i-1], positive_edges, negative_edges)))
            self.h_neg.append(torch.tanh(self.negative_aggregators[i-1](self.h_neg[i-1], self.h_pos[i-1], positive_edges, negative_edges)))
        self.z = torch.cat((self.h_pos[-1], self.h_neg[-1]), 1)
        return self.z, self.h_pos[-1], self.h_neg[-1]
