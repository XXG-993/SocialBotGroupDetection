import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from ReadConfig import ReadConfig

torch.manual_seed(1)

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMEncoder, self).__init__()
        
        self.lstm = nn.LSTM(input_size, 
                            hidden_size,
                            batch_first=True)

    def forward(self, x):
        x, (hidden_state, _) = self.lstm(x)
        return x
        return hidden_state.squeeze(0)


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(LSTMDecoder, self).__init__()
        
        self.lstm = nn.LSTM(hidden_size, output_size, batch_first=True)

    def forward(self, x, ):
        # output, _ = self.lstm(x.unsqueeze(1), (hidden_state.unsqueeze(0), torch.zeros_like(hidden_state).unsqueeze(0)))
        x, _ = self.lstm(x)
        return x
        return output.squeeze(1)

class AttentionAggregator(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionAggregator, self).__init__()
        self.cfg = ReadConfig().read_config()
        self.hidden_size = self.cfg["hidden_dim"]
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        attention_weights = F.softmax(self.attention(x), dim=1)
        context_vector = torch.sum(attention_weights * x, dim=1)
        return context_vector, attention_weights

class AutoEncoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.cfg = ReadConfig().read_config()
        self.input_size = self.cfg["input_dim"] 
        self.hidden_size = self.cfg["hidden_dim"]
        self.hidden_num_layer = self.cfg["hidden_num_layer"]
        self.output_size = self.cfg["output_dim"]
        self.classify_size = self.cfg["classify_dim"]
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        
        self.encoder = LSTMEncoder(self.input_size, self.hidden_size)
        self.decoder = LSTMDecoder(self.hidden_size, self.input_size)
        self.aggregator = AttentionAggregator(self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, self.classify_size)

        
    def forward(self, x):
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        context_vector, attention_weights = self.aggregator(encoded)
        y = self.classifier(context_vector)
        
        y_cate = self.softmax(y[:,:2])
        y_content = self.sigmoid(y[:,2:])
        
        return encoded, decoded, context_vector, y_cate, y_content, attention_weights
        