from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
import torch
# import dgl
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
import networkx as nx
from prettytable import PrettyTable


import os
import copy
from tqdm import tqdm
from utils_ import structured_negative_sampling

from AutoEncoder import SGCNAE
from ReadConfig import ReadConfig
from DataLoader import AutoEncoderDataLoader
from ConstractiveLearning import CLDataLoader, CL, CLLoss

class Train:
    
    def __init__(self) -> None:
        # Train.__find_gpu()
        
        self.cfg = ReadConfig().read_config()
        self.epoch = self.cfg['epoch']
        self.epoch_pre = self.cfg['epoch_pre']
        self.lr = self.cfg['lr']
        self.n_cluster = self.cfg['n_cluster']
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        
        self.lambda_ = self.cfg['lambda']
        self.beta_ = self.cfg["beta"]
        # self.beta_ = self.cfg['beta']
        
        self.K = 10
        self.dataLoader = AutoEncoderDataLoader(K=self.K)
        self.X, self.label = self.dataLoader.get_all()
        self.matrix = self.dataLoader.matrix
        self.pos_edges, self.neg_edges = self.dataLoader.pos_edges.to(self.device), self.dataLoader.neg_edges.to(self.device)
        self.pos_adj, self.neg_adj = self.dataLoader.pos_adj_norm, self.dataLoader.neg_adj_norm
        self.opinion_leaders = self.dataLoader.opinion_leaders
        self.KNN_g = torch.tensor(self.dataLoader.KNN_g, dtype=torch.float32).to(self.device)
        self.KNN_g_norm = torch.tensor(self.dataLoader.KNN_g_norm, dtype=torch.float32).to(self.device)
        self.coor_g = torch.tensor(self.dataLoader.coor_g, dtype=torch.float32).to(self.device)
        self.coor_g_norm = torch.tensor(self.dataLoader.coor_g_norm, dtype=torch.float32).to(self.device)
        self.con_g = torch.tensor(self.dataLoader.con_g, dtype=torch.float32).to(self.device)
        self.con_g_norm = torch.tensor(self.dataLoader.con_g_norm, dtype=torch.float32).to(self.device)
        
    def prepare_data(self):
        # train_dataset = AutoEncoderDataLoader()
        print("len of dataset",self.dataLoader.__len__())
        self.train_dataloader = DataLoader(self.dataLoader, batch_size=self.cfg['batch_size'], shuffle=True)
        

    def modularity(self, pred):
        
        # C = (pred == pred.max(dim=1, keepdim=True)[0]).to(dtype=torch.float32)
        C = pred
        d = self.con_g.sum(axis=1)
        B = self.con_g - (d*d.T)/self.con_g.sum()
        
        cr = (torch.sqrt(torch.tensor(self.n_cluster, dtype=torch.float32).to(self.device))/C.shape[1])*torch.norm(C.T.sum(axis=0))

        return -((C.T@B)@C).trace()/self.con_g.sum() + cr - 1

    def loss_fn(self, z, X, X_, pred, A_pos, A_neg, positive_edges, negative_edges):
        """
        Calculating the embedding losses, regression loss and weight regularization loss.
        :param z: Node embedding.
        :param positive_edges: Positive edge pairs.
        :param negative_edges: Negative edge pairs.
        :param target: Target vector.
        :return loss: Value of loss.
        """
        
        mod_loss = self.modularity(pred)
        rec_loss = F.mse_loss(X, X_)
        struc_pos_loss_ = F.mse_loss(torch.tensor(self.dataLoader.pos_adj_norm).to(torch.float32).to(self.device), A_pos)
        struc_neg_loss_ = F.mse_loss(torch.tensor(self.dataLoader.neg_adj_norm).to(torch.float32).to(self.device), A_neg)
        
        loss_term = mod_loss + rec_loss + (struc_pos_loss_ + struc_neg_loss_)
        return loss_term

    def feature_optimize(self, X):
        
        CL_model = CL(X.shape[1], 128, 128).to(self.device)
        optimizer = torch.optim.Adam(CL_model.parameters(), lr=0.001)
        criterion = CLLoss().to(self.device)
        CL_dataloader = CLDataLoader(self.dataLoader, X)
        train_dataloader = DataLoader(CL_dataloader, batch_size=self.cfg['batch_size'], shuffle=True)
        
        for epoch in tqdm(range(self.epoch_pre)):
            loss_epoch = 0
            for batch_idx, data in enumerate(train_dataloader):
                x_i, x_j, label = data
                optimizer.zero_grad()
                x_i, x_j, label = x_i.to(self.device), x_j.to(self.device), label.to(self.device)
                h_i, h_j, z_i, z_j = CL_model(x_i, x_j)
                loss = criterion(z_i, z_j, label)
                loss.backward()
                optimizer.step()
                
                if (batch_idx+1) % 1000 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(\
                        epoch+1, self.epoch_pre, batch_idx+1, len(train_dataloader), loss.item()))
                loss_epoch = loss_epoch + loss.item()
                
            print('Epoch [{}/{}],  Loss: {:.4f}'.format(\
                        epoch+1, self.epoch_pre, loss_epoch))
            
        with torch.no_grad():
            x_i, x_j = CL_dataloader.__getall__()
            x_i, x_j = x_i.to(self.device), x_j.to(self.device)
            h_i, h_j, z_i, z_j = CL_model(x_i, x_j)
        
        h_i_np = h_i.cpu().detach().numpy()
        np.save(self.cfg["save_path"]+"/emb_argument-top"+self.K+".npy", h_i_np)
        return h_i
    
    def SignedGAE_trianing(self, X):
        
        self.model = SGCNAE(X)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)   
        self.model = self.model.to(self.device)
        self.model.train()
        self.matrix = self.matrix.to(torch.float32).to(self.device)
        positive_edges, negative_edges = self.dataLoader.pos_edges, self.dataLoader.neg_edges
        positive_edges, negative_edges = positive_edges.to(self.device), negative_edges.to(self.device)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=50, factor=0.5)

        
        for epoch in tqdm(range(self.epoch)):
            
            loss_0 = 0
            # for batch_id, item in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            
            z, A_pos, A_neg, X_, pred = self.model(positive_edges, negative_edges, self.matrix, self.con_g_norm)
            loss = self.loss_fn(z, X, X_, pred, A_pos, A_neg, positive_edges, negative_edges)
            loss_item = loss.item()
            loss_0 += loss_item
            
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step(loss)
            print("epoch:", epoch, ",loss:", loss_0)
        
        z = z.cpu().detach().numpy()
        # np.save(self.cfg["save_path"]+"/before_classification.npy", z)
        
        return pred
    
    def train(self):
        
        X, label = self.dataLoader.get_all()
        
        print("-------embedding argument--------")
        X = self.feature_optimize(X)
        
        print("-------group detection--------")
        pred = self.SignedGAE_trianing(X)
  
        pred = pred.cpu().detach().numpy().argmax(1)
        groups = self.group_detection(pred)
        groups = self.rank(groups)

        bot_rate, pr, nr, cons = self.eva(groups, label, "final")
        
        
        # torch.save(self.model.state_dict(), self.cfg["save_path"]+"/1128-SGCNAE-01.pth")
        print("-------finish training--------")   
        return bot_rate, pr, nr, cons  
    
    
    def group_detection(self, y):
        def get_index(lst=None, item=''):
            return [index for (index,value) in enumerate(lst) if value == item]
        
        cate = list(set(y))
        groups = []
        for c in cate:
            group_cate = get_index(y, c)
            group_cate = list(set(group_cate)-set(self.opinion_leaders))
            if len(group_cate) >= 20:
                groups.append(group_cate)
        return groups
    
    def rank(self, groups):
        
        others = len(self.matrix)-len(self.opinion_leaders)
        standard_pos_rate, standard_neg_rate, standard_consistency = self.group_consistency(list(range(others)))
        print(standard_pos_rate, standard_neg_rate, standard_consistency)
        beta = 2
        
        results = []
        
        for i,g in enumerate(groups):
            pos_rate, neg_rate, consistency = self.group_consistency(g)
            if pos_rate>beta*standard_pos_rate \
                and neg_rate>beta*standard_neg_rate \
                and consistency>beta*standard_consistency:
                results.append(g)
        return results
    
    def eva(self, groups, label, epoch):
        
        bot_rates = []
        pos_rates = []
        neg_rates = []
        conss = []
        bot_number = 0
        table = PrettyTable()
        table.field_names = ["group_num",\
                            "group_size", \
                            "bot_rate", \
                            "pos_rate", \
                            "neg_rate", \
                            "cons"]
        for i,g in enumerate(groups):
            bot_rate = float(label[np.array(g)].sum()/len(g))
            pos_rate, neg_rate, consistency = self.group_consistency(g)
            table.add_row([i, \
                            len(g),\
                            bot_rate, \
                            pos_rate, \
                            neg_rate, \
                            consistency])

            bot_rates.append(bot_rate)
            pos_rates.append(pos_rate)
            neg_rates.append(neg_rate)
            conss.append(consistency)
            bot_number += label[np.array(g)].sum()
            
        print(table)
        print(epoch, "bot_rate_mean:", np.array(bot_rates).mean())
        print(epoch, "pos_rate_mean:", np.array(pos_rates).mean())
        print(epoch, "neg_rate_mean:", np.array(neg_rates).mean())
        print(epoch, "cons_mean:", np.array(conss).mean())
        print(epoch, "bot_completeness:", float(bot_number/self.label.sum()))
        return np.array(bot_rates).mean(), np.array(pos_rates).mean(), np.array(neg_rates).mean(),  np.array(conss).mean()
        
    def group_consistency(self, g):
        
        opinion_leaders_num = len(self.opinion_leaders)
        pos = self.pos_adj[np.array(g), :][:, -opinion_leaders_num:]
        neg = self.neg_adj[np.array(g), :][:, -opinion_leaders_num:]
        pos = (pos @ pos.T) / len(self.opinion_leaders)
        neg = (neg @ neg.T) / len(self.opinion_leaders)
        con = pos + neg
        con[~np.eye(con.shape[0],dtype=bool)].reshape(con.shape[0],-1)
        pos[~np.eye(pos.shape[0],dtype=bool)].reshape(pos.shape[0],-1)
        neg[~np.eye(neg.shape[0],dtype=bool)].reshape(neg.shape[0],-1)

        return pos.mean(), neg.mean(), con.mean()
    
    @staticmethod
    def __find_gpu():
        if torch.cuda.is_available():
            os.system("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp")
            mem_ava = [int(x.split()[2]) for x in open("tmp", 'r').readlines()]
            gpu_id = int(np.argmax(mem_ava))
            torch.cuda.set_device(gpu_id)

botrates, pos_rates, neg_rates, conss = [], [], [], []
seeds = [1,2,3]
for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = Train()
    bt, pr, nr, cons = train.train()
    botrates.append(bt)
    pos_rates.append(pr)    
    neg_rates.append(nr)
    conss.append(cons)
    
print("bot_rate_mean:", np.array(botrates).mean(), np.array(botrates).std())
print("pos_rate_mean:", np.array(pos_rates).mean(), np.array(pos_rates).std())     
print("neg_rate_mean:", np.array(neg_rates).mean(), np.array(neg_rates).std())
print("cons_mean:", np.array(conss).mean(), np.array(conss).std())
            
