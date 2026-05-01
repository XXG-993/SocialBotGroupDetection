import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
# from info_nce import InfoNCE

class CLDataLoader(Dataset):
    
    def __init__(self, dataloader, X) -> None:
        super(CLDataLoader, self).__init__()
        
        self.dataloader = dataloader
        # self.X = np.array(X.cpu().detach())
        self.X = X
        self.users = self.dataloader.users
        self.pos_adj_norm, self.neg_adj_norm = self.dataloader.pos_adj_norm, self.dataloader.neg_adj_norm
        self.pos_edges, self.neg_edges = self.dataloader.pos_edges, self.dataloader.neg_edges
        self.pos_row, self.pos_col, self.neg_row, self.neg_col = self.similarity_matrix()
        self.emb_row, self.emb_col, self.emb_label = self.emb_pair()
        
    def __getitem__(self, index):
        
        emb_row = self.emb_row[index]
        emb_col = self.emb_col[index]
        emb_label = self.emb_label[index]
        
        return emb_row, emb_col, emb_label
    
    def __getall__(self):
        
        return torch.tensor(self.X).to(torch.float32), torch.tensor(self.X).to(torch.float32)
    
    def __len__(self):
        
        return len(self.emb_row)
    
    def similarity_matrix(self):
        
        simi_pos_adj = self.pos_adj_norm @ self.pos_adj_norm.T
        simi_neg_adj = self.neg_adj_norm @ self.neg_adj_norm.T
        simi_adj = simi_pos_adj + simi_neg_adj
        
        threshold_upon = simi_adj.max()/2
        threshold_down = 1
        pos_row, pos_col = np.where(simi_adj>=threshold_upon)
        neg_row, neg_col = np.where(simi_adj<=threshold_down)
        
        return pos_row, pos_col, neg_row, neg_col
    
    def emb_pair(self):
        
        # emb_pos_row = np.zeros((len(self.pos_row), self.X.shape[1]))
        emb_pos_row = self.X[self.pos_row]
        # emb_pos_col = np.zeros((len(self.pos_col), self.X.shape[1]))
        emb_pos_col = self.X[self.pos_col]
        # emb_neg_row = np.zeros((len(self.neg_row), self.X.shape[1]))
        emb_neg_row = self.X[self.neg_row]
        # emb_neg_col = np.zeros((len(self.neg_col), self.X.shape[1]))
        emb_neg_col = self.X[self.neg_col]
        emb_row = np.concatenate([emb_pos_row, emb_neg_row], axis=0)
        emb_col = np.concatenate([emb_pos_col, emb_neg_col], axis=0)
        emb_label = np.zeros((len(self.pos_row)+len(self.neg_row), 1))
        emb_label[len(self.pos_row):] = 1

        return emb_row, emb_col, emb_label
    
class CL(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CL, self).__init__()
        
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )
        
    def forward(self, x_i, x_j):
        x_i = torch.tensor(x_i).to(torch.float32)
        x_j = torch.tensor(x_j).to(torch.float32)
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        
        return h_i, h_j, z_i, z_j

class CLLoss(nn.Module):
    
    def __init__(self, margin=2.0):
        super(CLLoss, self).__init__()
        self.margin = margin
        
        # self.cri = torch.nn.CrossEntropyLoss()
        
    def forward(self, z_i, z_j, label):
        euclidean_distance = F.pairwise_distance(z_i, z_j)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
            
