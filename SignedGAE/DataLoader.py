from ast import literal_eval
import numpy as np
import pandas as pd
# from scipy import sparse
import pickle
import torch
import networkx as nx
import random
# import dgl
from itertools import combinations, permutations
from ReadConfig import ReadConfig
from torch.utils.data import Dataset

from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize


class AutoEncoderDataLoader(Dataset):
    
    def __init__(self, K=10, Ksimilar=10) -> None:
        super().__init__()
        self.cfg = ReadConfig().read_config()
        self.data_path = self.cfg["data_path"]
        
        self.K = K
        self.Ksimilar = Ksimilar
        self.retweet_matrix = self.load_file(self.cfg["retweet_path"])
        self.users = []
        
        self.tweet_emb = self.load_file(self.data_path+"tweet_emb.pickle")
        self.attr_emb = self.load_file(self.data_path+"attr_emb.npy")
        # self.struc_emb = self.load_file(self.data_path+"structure_emb.npy")
        self.struc_emb = self.load_file("hyperwalk_10.npy")
        self.user_emb = self.initialised()
        self.label = torch.tensor(self.load_file(self.data_path+"node_labels.npy"))
        
        self.pos_adj = self.load_file(self.data_path+"pos_adj_top"+str(K)+".npy")
        self.neg_adj = self.load_file(self.data_path+"neg_adj_top"+str(K)+".npy")
        self.matrix = self.load_g()
        self.pos_adj_norm, self.neg_adj_norm = self.build_signed_graph()
        self.pos_edges, self.neg_edges = self.pos_neg_edges()
        self.KNN_g, self.KNN_g_norm = self.construct_similar_graph()
        self.coor_g, self.coor_g_norm = self.construct_coor_graph()
        self.con_g, self.con_g_norm = self.construct_con_graph()
        
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        label = self.user_emb[idx]
        data = self.user_emb[idx,:]
        return data, label
    
    def get_all(self):
        label = self.label[np.array(self.users)]
        data = self.user_emb[np.array(self.users)]
        return data, label
    
    def load_file(self, filename) -> np.array:
        cate = filename.split(".")[-1]
        if cate == "pickle":
            with open(filename, "rb") as f:
                return pickle.load(f)
        elif cate == "npy":
                return np.load(filename)
        
    def initialised(self):
    
        # tweet_emb_mean = np.zeros((len(self.tweet_emb),len(self.tweet_emb[0][0])))
        # print(self.attr_emb.shape[1],self.struc_emb.shape[1],tweet_emb_mean.shape[1])
        
        # id_empty = []
        # for i, te in enumerate(self.tweet_emb):
        #     if len(te) != 0:
        #         tweet_emb_mean[i] = np.sum(np.array(te), axis=0)
        #     else:
        #         tweet_emb_mean[i] = np.array([np.nan]*768)
        #         id_empty.append(i)

        # emb_mean = np.nanmean(tweet_emb_mean, axis=0)
        # tweet_emb_mean[id_empty] = emb_mean
        
        for i in [0,2,3,4,5]:
            self.attr_emb[:,i] = standardization(self.attr_emb[:,i])
            
        # user_emb =  np.concatenate((self.attr_emb,self.struc_emb,tweet_emb_mean),axis=1)
        user_emb =  np.concatenate((self.attr_emb,self.struc_emb),axis=1)
        print("user_emb:",user_emb.shape)
        return user_emb
    
    def load_g(self):
        
        retweet_matrix_norm = np.where(self.retweet_matrix>0,1,0)
        retweeted_users = retweet_matrix_norm.T.sum(axis=1)
        retweeted_users_rank = np.argsort(-retweeted_users)[:self.K]
        self.opinion_leaders = retweeted_users_rank
        
        def Target_users(id_):
            Fox = retweet_matrix_norm.T[retweeted_users_rank[id_]]
            return [i for i,v in enumerate(Fox) if v > 0]
        
        top_k_users = []
        for i in range(self.K):
            top_k_users.extend(Target_users(i))
        top_k_users = sorted(list(set(top_k_users)-set(retweeted_users_rank)))
        self.users = top_k_users + list(retweeted_users_rank)
        
        matrix = np.where(self.pos_adj + self.neg_adj>0,1,0)
        return torch.tensor(matrix)
    
    def build_signed_graph(self):
        
        pos_adj_norm = np.where(self.pos_adj>self.neg_adj,1,0)
        pos_adj_norm = np.where(np.logical_and(self.pos_adj==self.neg_adj,self.pos_adj>0),1,pos_adj_norm)
        neg_adj_norm = np.where(self.pos_adj<self.neg_adj,1,0)
        
        # pos_adj_norm = self.matrix
        # neg_adj_norm = np.zeros_like(pos_adj_norm)
        return pos_adj_norm, neg_adj_norm
        
    def pos_neg_edges(self):
        
        pos_row, pos_col = np.where(np.triu(self.pos_adj_norm)>0)
        pos_edges = torch.tensor([pos_row, pos_col]).to(torch.long)
        neg_row, neg_col = np.where(np.triu(self.neg_adj_norm)>0)
        neg_edges = torch.tensor([neg_row, neg_col]).to(torch.long)
        
        print("N.O. positive edges:",len(pos_row),"\nN.O. negative edges:", len(neg_col))
        return pos_edges, neg_edges
    
    def construct_similar_graph(self):

        inds = []
        
        pos = self.pos_adj[:, -self.K:]
        neg = self.neg_adj[:, -self.K:]
        pos = (pos @ pos.T) / self.K
        neg = (neg @ neg.T) / self.K
        con = pos + neg
        dist = con
        con[~np.eye(con.shape[0],dtype=bool)].reshape(con.shape[0],-1)
        
        KNN_g = np.zeros((len(self.users),len(self.users)))
        for i in range(dist.shape[0]):
            ind = np.argpartition(dist[i,:], -(self.Ksimilar+1))[-(self.Ksimilar+1):]
            KNN_g[i,np.array(ind)] = 1
            inds.append(ind)
        
        # KNN_g = normalised(KNN_g)
        return KNN_g, normalised(KNN_g)
    def construct_coor_graph(self):
        
        def coordination_matrix(matrix:np.array):
            c_matrix = np.zeros_like(matrix)
            
            for m in matrix.T:
                u_ids = np.where(m > 0)[0]
                nodes_list = list(combinations(u_ids, 2))
                for n in nodes_list:
                    c_matrix[int(n[0]),int(n[1])] += 1
                    c_matrix[int(n[1]),int(n[0])] += 1
            return c_matrix
        
        pos_coor = coordination_matrix(self.pos_adj_norm)
        neg_coor = coordination_matrix(self.neg_adj_norm)
        coor_matrix = pos_coor + neg_coor
        coor_matrix = np.where(coor_matrix>0, 1, 0)
        
        return coor_matrix, normalised(coor_matrix)
    
    def construct_con_graph(self):
        
        con_g = (self.pos_adj @ self.pos_adj.T) + (self.neg_adj @ self.neg_adj) 

        return con_g, normalised(con_g)
    
    
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def normalised(A):
    D = np.sum(A, axis=1)
    D = np.sqrt(1.0 / (D + 1e-9))

    D = np.diag(D)

    A_hat = D.dot(A).dot(D)
    return A_hat
    