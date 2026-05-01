import numpy as np
import pandas as pd
import pickle
import copy
import torch
import networkx as nx
import random
import os
from itertools import combinations
from ReadConfig import ReadConfig
from tqdm import tqdm

import dgl
from dgl import save_graphs,  load_graphs
from dgl.data.utils import save_info, load_info
from dgl.data import DGLDataset
    
class BuildData():

    def __init__(self, motif_cate=3, dataset="Twibot-20", rate=0.5) -> None:
        
        self.cfg = ReadConfig().read_config()
        self.dataset = dataset
        self.motif_cate = motif_cate
        if dataset == "Twibot-20":
            self.data_path = self.cfg["Twibot20_data_path"]
            # if self.motif_cate == 2 :
            #     self.motifs = self.load_file(self.data_path+"process/interaction_2motifs.pickle")
            # elif self.motif_cate == 3:
            #     self.motifs = self.load_file(self.data_path+"process/interaction_3motifs.pickle")
        elif dataset == "Renmin":
            self.data_path = self.cfg["Renmin_data_path"]
            if self.motif_cate == 2 :
                self.motifs = self.load_file(self.data_path+"process/interaction_2motifs.pickle")
            elif self.motif_cate == 3:
                self.motifs = self.load_file(self.data_path+"process/interaction_3motifs.pickle")
        print("Using dataset:", dataset)
        
        self.retweet_matrix = self.load_file(self.data_path+"process/retweet_matrix_label.pickle")
        self.mention_matrix = self.load_file(self.data_path+"process/mention_matrix_label.pickle")
        self.interaction_matrix = self.retweet_matrix + self.mention_matrix
        self.co_retweet_matrix = self.coordination_matrix(self.retweet_matrix)
        self.co_mention_matrix = self.coordination_matrix(self.mention_matrix)
        self.label = self.load_file(self.data_path+"node_labels.npy")
        self.bot_index = np.where(self.label>0)[0]
        
        if dataset == "Renmin":
            self.bot_motif, self.other_motif = self.graph_sampling(self.motifs)
        elif dataset == "Twibot-20":
            self.bot_motif, self.other_motif = self.graph_sampling_Twibot()
        
        self.tweet_emb = self.load_file(self.data_path+"history_emb_attention.npy")
        self.attr_emb = self.load_file(self.data_path+"attr_emb.npy")
        self.user_emb = self.initialised()   
        self.rate = rate
        
        self.train_data, self.test_data = self.process()
        

    def load_file(self, filename) -> np.array:
        cate = filename.split(".")[-1]
        if cate == "pickle":
            with open(filename, "rb") as f:
                return pickle.load(f)
        elif cate == "npy":
                return np.load(filename, allow_pickle=True)
    
    def coordination_matrix(self, matrix:np.array):
        c_matrix = np.zeros_like(matrix)
        
        for m in matrix.T:
            u_ids = np.where(m > 0)[0]
            nodes_list = list(combinations(u_ids, 2))
            for n in nodes_list:
                c_matrix[int(n[0]),int(n[1])] += 1
                c_matrix[int(n[1]),int(n[0])] += 1
        return c_matrix
       
    def initialised(self):
    
        print(self.tweet_emb.shape)
        for i in [0,2,3,4,5]:
            self.attr_emb[:,i] = standardization(self.attr_emb[:,i])
        
        user_emb = np.concatenate((self.attr_emb, self.tweet_emb),axis=1)
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
        
        bot_motif = random.choices(bot_motif, k=200)
        sample_motif = random.choices(other_motif, k=4*len(bot_motif))
        
        bot_motif = [list(m) for m in bot_motif]
        sample_motif = [list(m) for m in sample_motif]
        self.bot_motif = bot_motif
        return bot_motif, sample_motif
    
    def graph_sampling_Twibot(self):
        
        if self.motif_cate == 2 :
            with open(self.data_path+"process/interaction_2motifs-bot-sampling.pickle", "rb") as f:
                bot_motif = pickle.load(f)
            with open(self.data_path+"process/interaction_2motifs-other-sampling.pickle", "rb") as f:
                other_motif = pickle.load(f)
        if self.motif_cate == 3 :
            with open(self.data_path+"process/interaction_3motifs-bot-sampling.pickle", "rb") as f:
                bot_motif = pickle.load(f)
            with open(self.data_path+"process/interaction_3motifs-other-sampling.pickle", "rb") as f:
                other_motif = pickle.load(f)
        
        bot_motif = [list(m) for m in bot_motif]
        sample_motif = [list(m) for m in other_motif]
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
        shuffle_motif = shuffle_motif[shuffle_ix]
        return shuffle_motif, shuffle_labels
    
    def get_all_data(self, motifs):
        
        nodes_num = []
        edges = []
        features = []
        inter_graphs = []
        for m in motifs:
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
            
            feature = self.user_emb[np.array(nodes_list)]
            feature = feature.astype(float)
            feature = torch.tensor(feature).float()
            features.append(feature)
            
            nodes_num.append(len(nodes_list))
            
            new_inter_graph = dict()
            for k in inter_graph.keys():
                nodes_ = inter_graph[k]
                nodes_ = [old2new_node_mapping[n] for n in nodes_]
                new_inter_graph[old2new_node_mapping[k]] = nodes_
            inter_graphs.append(new_inter_graph)
            # print("node num:", len(nodes), "edge num:", len(edge[0]))
        
        return edges, features, nodes_num, inter_graphs
    
    def get_coordination_data(self, motifs):
        
        def neighbors(i, edge):
            neigs = set()
            idxs = list(np.where(np.array(edge[('user','retweeted','ego')][1]) == i)[0])
            neigs = neigs | set(list(np.array(edge[('user','retweeted','ego')][0])[np.array(idxs, dtype=int)]))
            idxs = list(np.where(np.array(edge[('ego','retweeted','ego')][1]) == i)[0])
            neigs = neigs | set(list(np.array(edge[('ego','retweeted','ego')][0])[np.array(idxs, dtype=int)]))
            idxs = list(np.where(np.array(edge[('user','mentioned','ego')][1]) == i)[0])
            neigs = neigs | set(list(np.array(edge[('user','mentioned','ego')][0])[np.array(idxs, dtype=int)]))
            idxs = list(np.where(np.array(edge[('ego','mentioned','ego')][1]) == i)[0])
            neigs = neigs | set(list(np.array(edge[('ego','mentioned','ego')][0])[np.array(idxs, dtype=int)]))
            
            return neigs
        
        co_edges = []
        for m in motifs:
            
            edge, nodes, inter_graph = self.build_graph(m)
            co_edge = [[],[]]
            
            for j in m:
                for k in m:
                    if k != j:
                        k_neigh = neighbors(k, edge)
                        j_neigh = neighbors(j, edge)
                        if len(k_neigh & j_neigh) > 0:
                            co_edge[0].append(j)
                            co_edge[1].append(k)
            co_edges.append(tuple(co_edge))
        
        return co_edges
    
    def process(self):

        shuffle_motif, y = self.shuffle_motif()
        y = torch.tensor(y).long()
        edge_index_list, x_list, nodes_num, inter_graphs = self.get_all_data(shuffle_motif)
        self.all_data = [edge_index_list, nodes_num, x_list, y, shuffle_motif, inter_graphs]
        self.co_edges = self.get_coordination_data(shuffle_motif)
        
        test_data = [edge_index_list[int(self.rate*y.shape[0]):],\
                        nodes_num[int(self.rate*y.shape[0]):],\
                        x_list[int(self.rate*y.shape[0]):],\
                        y[int(self.rate*y.shape[0]):],\
                        shuffle_motif[int(self.rate*y.shape[0]):],\
                        inter_graphs[int(self.rate*y.shape[0]):]]
        
        train_data = [edge_index_list[:int(self.rate*y.shape[0])],\
                        nodes_num[:int(self.rate*y.shape[0])],\
                        x_list[:int(self.rate*y.shape[0])],\
                        y[:int(self.rate*y.shape[0])],\
                        shuffle_motif[:int(self.rate*y.shape[0])],\
                        inter_graphs[:int(self.rate*y.shape[0])]]
        
        return train_data, test_data

class GraphData(DGLDataset):
    
    def __init__(self, data ,save_path="./data"):
        
        
        self.cfg = ReadConfig().read_config()
        self.data_path = self.cfg["Renmin_data_path"]
 
        self.G_list, self.nodes_num, self.features, self.G_labels, self.shuffle_motif, self.inter_graphs = data
        
        super().__init__(name='Renmin', url=None)
        self.process()
        
    def download(self):
        # download raw data to local disk
        pass
    
    def adj2edges(self, g):

        adjs = g.edges()
        adjs = [list(e) for e in adjs]
        adjs = torch.tensor(adjs).long().t()
        # print(adjs, len(g.nodes()))
        return adjs
    
    def ego_graph(self, g, inter_graph):
        # inter_graph = self.inter_graphs[i]
        for k in inter_graph.keys():
            mask_tensor = torch.tensor([1 if i in inter_graph[k] else 0 for i in g.nodes("user")])
            g.nodes["user"].data[str(k)+"_"] = mask_tensor.unsqueeze(dim=1)
            mask_tensor = torch.tensor([1 if i in inter_graph[k]else 0 for i in g.nodes("ego")])
            g.nodes["ego"].data[str(k)+"_"] = mask_tensor.unsqueeze(dim=1)
        return g
    
    def process(self):
        # process raw data to graphs, labels, splitting masks
        print("___start_processing_____")
        self.graphs = []
        for i in tqdm(range(len(self.G_list))):
            edge = self.G_list[i]
            g = dgl.heterograph(edge)
            g.nodes["ego"].data["x"] = torch.tensor(self.features[i][:len(g.nodes("ego"))]).float()
            g.nodes["user"].data["x"] = torch.tensor(self.features[i][len(g.nodes("ego")):]).float()
            g = self.ego_graph(g, self.inter_graphs[i])
            # print("x.shape:",g.ndata["x"].shape)
            self.graphs.append(g)
            
        self.labels = torch.tensor(self.G_labels)
        self.num_classes = len(set(self.G_labels))
        

    def save(self):
        # save graphs and labels
        graph_path = os.path.join(self.save_path, self.name + '_dgl_graph.bin')
        save_graphs(graph_path, self.graphs, {'labels': self.labels})
        # save other information in python dict
        info_path = os.path.join(self.save_path, self.name + '_info.pkl')
        save_info(info_path, {'num_classes': self.num_classes})

    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(self.save_path, self.name + '_dgl_graph.bin')
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']
        info_path = os.path.join(self.save_path, self.name + '_info.pkl')
        self.num_classes = load_info(info_path)['num_classes']

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, self.name + '_dgl_graph.bin')
        info_path = os.path.join(self.save_path, self.name + '_info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)

    def __getitem__(self, idx):
        # get one example by index
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        # number of data examples
        return len(self.graphs)
    
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    
    new_graphs = [g for g in graphs]
    
    return dgl.batch(new_graphs), torch.tensor(labels, dtype=torch.long) 

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