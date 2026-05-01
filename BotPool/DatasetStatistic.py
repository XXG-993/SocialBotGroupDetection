from ast import literal_eval
import numpy as np
import pandas as pd
import pickle
import copy
import torch
import networkx as nx
import random
import re
import os
from MyLogger import MyLogger
from itertools import combinations, permutations
from ReadConfig import ReadConfig
from torch.utils.data import Dataset
from tqdm import tqdm

import dgl
from dgl import save_graphs,  load_graphs
from dgl.data.utils import save_info, load_info
from torch.utils.data import DataLoader
from dgl.data import MiniGCDataset
from dgl.nn.pytorch import GraphConv, GATConv, HeteroGraphConv
from sklearn.metrics import accuracy_score, recall_score, f1_score
from dgl.data import DGLDataset
import torch.nn as nn
import torch.nn.functional as F

random.seed(12345)
np.random.seed(12345)
torch.manual_seed(12345)


class BuildData():
    
    def __init__(self, motif_cate=3, dataset="Twibot-20", rate=0.5) -> None:
        
        self.cfg = ReadConfig().read_config()
        self.dataset = dataset
        if dataset == "Twibot-20":
            self.data_path = self.cfg["Twibot20_data_path"]
        elif dataset == "Renmin":
            self.data_path = self.cfg["Renmin_data_path"]
        print("Using dataset:", dataset)
        
        self.retweet_matrix = self.load_file(self.data_path+"process/retweet_matrix_label.pickle")
        self.mention_matrix = self.load_file(self.data_path+"process/mention_matrix_label.pickle")
        self.interaction_matrix = self.retweet_matrix + self.mention_matrix
        self.co_retweet_matrix = self.coordination_matrix(self.retweet_matrix)
        self.co_mention_matrix = self.coordination_matrix(self.mention_matrix)
        self.motif_cate = motif_cate
        if self.motif_cate == 2 :
            self.motifs = self.load_file(self.data_path+"process/interaction_2motifs.pickle")
        elif self.motif_cate == 3:
            self.motifs = self.load_file(self.data_path+"process/interaction_3motifs.pickle")
        self.bot_motif, self.other_motif = self.graph_sampling(self.motifs)

        self.rate = rate
        
        self.process()
        
        
        
    def load_file(self, filename) -> np.array:
        cate = filename.split(".")[-1]
        if cate == "pickle":
            with open(filename, "rb") as f:
                return pickle.load(f)
        elif cate == "npy":
                return np.load(filename, allow_pickle=True)
    
    def coordination_matrix(self, matrix:np.array):
        c_matrix = np.zeros_like(matrix)
        
        for m in matrix:
            u_ids = np.where(m > 0)[0]
            nodes_list = list(combinations(u_ids, 2))
            for n in nodes_list:
                c_matrix[int(n[0]),int(n[1])] += 1
                c_matrix[int(n[1]),int(n[0])] += 1
        return c_matrix
       
    def initialised(self):
    
        tweet_emb_mean = np.zeros((len(self.tweet_emb),len(self.tweet_emb[0][0])))
        print(self.attr_emb.shape[1],self.struc_emb.shape[1],tweet_emb_mean.shape[1])
        
        id_empty = []
        for i, te in enumerate(self.tweet_emb):
            if len(te) != 0:
                tweet_emb_mean[i] = np.sum(np.array(te), axis=0)
            else:
                tweet_emb_mean[i] = np.array([np.nan]*768)
                id_empty.append(i)

        emb_mean = np.nanmean(tweet_emb_mean, axis=0)
        tweet_emb_mean[id_empty] = emb_mean
        
        for i in [0,2,3,4,5]:
            self.attr_emb[:,i] = standardization(self.attr_emb[:,i])
        
        # user_emb =  np.concatenate((self.attr_emb,self.struc_emb,tweet_emb_mean),axis=1)
        user_emb = np.concatenate((self.attr_emb, tweet_emb_mean),axis=1)
        # user_emb = tweet_emb_mean
        # user_emb = np.random.random(tweet_emb_mean.shape)
        
        print("user_emb:",user_emb.shape)
        return user_emb
    
    def graph_sampling(self, motifs):
        motifs = [list(m) for m in motifs]
        
        if self.motif_cate == 3:
            motif1, motif2, motif3, motif4, motif5, \
            motif6, motif7, motif8, motif9, motif10 = motifs
            motif_num = [len(m) for m in motifs]
            
            bot_motif = motif6 + motif10
            other_motif = motif1 + motif2 + motif3 + motif4 \
                        + motif5 + motif7 + motif8 + motif9
        elif self.motif_cate == 2:
            motif1, motif2, motif3 = motifs
            motif_num = [len(m) for m in motifs]
            
            bot_motif = motif3
            other_motif = motif1 + motif2
        
        # sample_motif = None
        # if self.dataset == "Renmin":
        #     sample_motif = random.choices(other_motif, k=4*len(bot_motif))

        # elif self.dataset == "Twibot-20":
        #     bot_motif = random.choices(bot_motif, k=200)
        #     sample_motif = random.choices(other_motif, k=4*len(bot_motif))
        
        bot_motif = [list(m) for m in bot_motif]
        sample_motif = [list(m) for m in other_motif]
        
        print("bot_motif", len(bot_motif))
        print("sample_motif", len(sample_motif))
        return bot_motif, sample_motif
    
    def build_graph(self, motif):
            
        edges = {('ego','retweeted','user'):[[],[]],
                 ('ego','mentioned','user'):[[],[]],
                 ('user','retweeted','ego'):[[],[]],
                 ('user','mentioned','ego'):[[],[]],
                 ('ego','retweeted','ego'):[[],[]],
                 ('ego','mentioned','ego'):[[],[]]}
        nodes_set = {'user':set(), 'ego':set()}
        inter_graph = dict()
        
        for node in motif:
            nodes_set['ego'].add(node)
            inter_graph[node] = dict()
            
            inter_nodes = None
            targets = np.where(self.retweet_matrix[node,:]>0)[0]
            inter_nodes = set(targets)
            for t in targets:
                if t in motif:
                    edges[('ego','retweeted','ego')][0].append(t)
                    edges[('ego','retweeted','ego')][1].append(node)
                else:
                    edges[('user','retweeted','ego')][0].append(t)
                    edges[('user','retweeted','ego')][1].append(node)
                    nodes_set['user'].add(t)
            targets = np.where(self.retweet_matrix[:,node]>0)[0]
            inter_nodes = set(targets) | inter_nodes
            for t in targets:
                if t in motif:
                    edges[('ego','retweeted','ego')][0].append(t)
                    edges[('ego','retweeted','ego')][1].append(node)
                else:
                    edges[('ego','retweeted','user')][0].append(node)
                    edges[('ego','retweeted','user')][1].append(t)
                    nodes_set['user'].add(t)

            targets = np.where(self.mention_matrix[node,:]>0)[0]
            inter_nodes = set(targets) | inter_nodes
            for t in targets:
                if t in motif:
                    edges[('ego','mentioned','ego')][0].append(t)
                    edges[('ego','mentioned','ego')][1].append(node)
                else:
                    edges[('user','mentioned','ego')][0].append(t)
                    edges[('user','mentioned','ego')][1].append(node)
                    nodes_set['user'].add(t)
            targets = np.where(self.mention_matrix[:,node]>0)[0]
            inter_nodes = set(targets) | inter_nodes
            for t in targets:
                if t in motif:
                    edges[('ego','mentioned','ego')][0].append(t)
                    edges[('ego','mentioned','ego')][1].append(node)
                else:
                    edges[('ego','mentioned','user')][0].append(node)
                    edges[('ego','mentioned','user')][1].append(t)
                    nodes_set['user'].add(t)     
            
            inter_graph[node] = inter_nodes | set([node])
            
        return edges, nodes_set, inter_graph
    
    def shuffle_motif(self):
        
        bot_labels = np.ones((len(self.bot_motif)))
        other_labels = np.zeros((len(self.other_motif)))
        shuffle_labels = np.concatenate((bot_labels, other_labels), axis=0)
        bot_motif = np.array([list(m) for m in self.bot_motif])
        other_motif = np.array([list(m) for m in self.other_motif])
        shuffle_motif = np.concatenate((bot_motif, other_motif), axis=0)
        shuffle_ix = np.random.permutation(np.arange(len(shuffle_labels)))
        shuffle_labels = shuffle_labels[shuffle_ix]
        # shuffle_labels = np.array([random.randint(0,1) for i in range(len(shuffle_labels))])
        shuffle_motif = shuffle_motif[shuffle_ix]
        # print(shuffle_motif.shape, shuffle_labels.shape)
        return shuffle_motif, shuffle_labels
    
    def get_all_data(self, motifs):
        
        nodes_num = []
        edges = []
        inter_graphs = []
        for m in tqdm(motifs):
            old2new_node_mapping = dict()
            edge, nodes, inter_graph = self.build_graph(m)
            nodes_list = sorted(list(nodes["ego"])) + sorted(list(nodes["user"]))
            nodes_new_index = [i for i,n in enumerate(sorted(list(nodes["ego"])))] + \
                            [i for i,n in enumerate(sorted(list(nodes["user"])))]
            
            for i in range(len(nodes_list)):
                old2new_node_mapping[nodes_list[i]] = nodes_new_index[i]
            
            for k in edge.keys():
                edge[k][0] = np.array([old2new_node_mapping[e] for e in edge[k][0]])
                edge[k][1] = np.array([old2new_node_mapping[e] for e in edge[k][1]])
                edge[k] = (edge[k][0], edge[k][1])
            edges.append(edge)

            
            nodes_num.append(len(nodes_list))
            
            new_inter_graph = dict()
            for k in inter_graph.keys():
                nodes_ = inter_graph[k]
                nodes_ = [old2new_node_mapping[n] for n in nodes_]
                new_inter_graph[old2new_node_mapping[k]] = nodes_
            inter_graphs.append(new_inter_graph)
            # print("node num:", len(nodes), "edge num:", len(edge[0]))
        
        return edges, nodes_num, inter_graphs
    
    def process(self):

        all_motif = np.concatenate((self.bot_motif, self.other_motif), axis=0)
        edge_index_list, nodes_num, inter_graphs = self.get_all_data(all_motif)
        print("avg node num:", sum(nodes_num)/len(nodes_num))
        
        # gs = []
        # for m in tqdm(range(len(edge_index_list))):
        #     edge = edge_index_list[m]
        #     g = dgl.heterograph(edge)
        #     gs.append(g)
        
        # for g in gs:
        #     idg = g.in_degrees(g.nodes("ego"), etype=('ego','retweeted','user'))
        #     if idg[1] > 1000:
        #         print("__________________")
        #         print(g)
        #         print(g.edges(etype=('ego','retweeted','user')))
        #         print(idg)
        #         print(g.in_degrees(torch.tensor(1), etype=('ego','retweeted','user')))
        #         print(len(g.edges(etype=('user','retweeted','ego')))+len(g.edges(etype=('user','retweeted','ego'))))
        
        # gs_in_r = [(g.in_degrees(g.nodes("ego"), etype=('ego','retweeted','user')) + g.in_degrees(g.nodes("ego"), etype=('ego','retweeted','ego'))) for g in gs]
        # gs_out_r = [g.out_degrees(g.nodes("ego"), etype=('ego','retweeted','user')).sum() + g.out_degrees(g.nodes("ego"), etype=('ego','retweeted','ego')).sum() for g in gs]
        # gs_in_m = [g.in_degrees(g.nodes("ego"), etype=('ego','mentioned','user')).sum() + g.in_degrees(g.nodes("ego"), etype=('ego','mentioned','ego')).sum() for g in gs]
        # gs_out_m = [g.out_degrees(g.nodes("ego"), etype=('ego','mentioned','user')).sum() + g.out_degrees(g.nodes("ego"), etype=('ego','mentioned','ego')).sum() for g in gs]
        # gs_in_inter = gs_in_r + gs_in_m
        # gs_out_inter = gs_out_r + gs_out_m
        
        # print([g.in_degrees(g.nodes("ego"), etype=('ego','retweeted','user')) for g in gs])
        # print([g.in_degrees(g.nodes("ego"), etype=('ego','retweeted','ego')) for g in gs])
        
        # print("retweet in", sum(gs_in_r)/len(gs_in_r))
        # print("retweet out", sum(gs_out_r)/len(gs_out_r))
        # print("mention in", sum(gs_in_m)/len(gs_in_m))
        # print("mention out", sum(gs_out_m)/len(gs_out_m))
        # print("inter in", sum(gs_in_inter)/len(gs_in_inter))
        # print("inter out", sum(gs_out_inter)/len(gs_out_inter))
        
        return 
        

dataname = "Twibot-20"
motif_size = 3
bd = BuildData(motif_cate=motif_size, dataset=dataname)
