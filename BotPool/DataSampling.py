import random
import numpy as np
import pickle
import networkx as nx
from itertools import combinations
from tqdm import tqdm

from ReadConfig import ReadConfig

class DataSampling():
    
    def __init__(self, motif_cate=2, dataset="Twibot-20") -> None:
        
        self.cfg = ReadConfig().read_config()
        self.dataset = dataset
        self.motif_cate = motif_cate
        if dataset == "Twibot-20":
            self.data_path = self.cfg["Twibot20_data_path"]
            if self.motif_cate == 2 :
                self.motifs = self.load_file(self.data_path+"process/interaction_2motifs-1000.pickle")
            elif self.motif_cate == 3:
                self.motifs = self.load_file(self.data_path+"process/interaction_3motifs-1000.pickle")
                
                
        self.retweet_matrix = self.load_file(self.data_path+"process/retweet_matrix_label.pickle")
        self.mention_matrix = self.load_file(self.data_path+"process/mention_matrix_label.pickle")
        self.retweet_matrix[np.eye(len(self.retweet_matrix),dtype=np.bool_)] = 0
        self.mention_matrix[np.eye(len(self.retweet_matrix),dtype=np.bool_)] = 0
        self.interaction_matrix = self.retweet_matrix + self.mention_matrix
        self.co_retweet_matrix = self.coordination_matrix(self.retweet_matrix)
        self.co_mention_matrix = self.coordination_matrix(self.mention_matrix)
        self.label = self.load_file(self.data_path+"node_labels.npy")
        self.bot_index = np.where(self.label>0)[0]
        
        self.bot_motif, self.other_motif = self.graph_sampling(self.motifs)
        self.save_motif()
        
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
    
    def get_coordination_data(self, motifs):
        
        def neighbors(i):
            neigs = set()
            idxs = set(list(np.where(self.interaction_matrix[i] > 0))[0])
            # neigs = neigs | set(list(np.array(edge[0])[np.array(idxs, dtype=int)]))
            return idxs
        
        co_edges = []
        for m in tqdm(motifs):
            co_edge = None
            for j in m:
                for k in m:
                    if k != j:
                        k_neigh = neighbors(k)
                        j_neigh = neighbors(j)
                        if len(k_neigh & j_neigh) > 0:
                            co_edge = (k,j)
            if co_edge != None:
                co_edges.append(co_edge)
        
        return co_edges
    
    def bot_motif_sampling(self, bot_motif):
        
        co_edges = self.get_coordination_data(bot_motif)
        g_ = nx.Graph()
        g_.add_edges_from(co_edges)
        g_connected = [g_.subgraph(c).copy() for c in nx.connected_components(g_)][0]
        
        new_bot_motifs = set()
        g_connected_nodes = list(g_connected.nodes)
        ego = random.choices(g_connected_nodes, k=1)[0]
        while len(new_bot_motifs) < 200:
            if self.motif_cate == 2:
                neighbors = g_connected.neighbors(ego)
                one_neighbor = random.choices(list(neighbors), k=1)[0]
                new_bot_motifs.add(frozenset([ego, one_neighbor]))
                ego = random.choices([ego, one_neighbor], k=1)[0]
            
            elif self.motif_cate == 3:
                neighbors_1 = g_connected.neighbors(ego)
                neighbor_1 = random.choices(list(neighbors_1), k=1)[0]
                neighbors_2 = g_connected.neighbors(neighbor_1)
                neighbor_2 = random.choices(list(set(neighbors_2)-set([ego])), k=1)[0]
                new_bot_motifs.add(frozenset([ego, neighbor_1, neighbor_2]))
                ego = random.choices([ego, neighbor_1, neighbor_2], k=1)[0]
            print(len(new_bot_motifs))
        return new_bot_motifs 
    
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


        bot_motif = list(self.bot_motif_sampling(bot_motif))
        sample_motif = random.choices(other_motif, k=4*len(bot_motif))
        
        bot_motif = [list(m) for m in bot_motif]
        sample_motif = [list(m) for m in sample_motif]
        return bot_motif, sample_motif
    
    def save_motif(self):
        
        if self.motif_cate == 2 :
            with open(self.data_path+"process/interaction_2motifs-bot-sampling.pickle", "wb") as f:
                pickle.dump(self.bot_motif, f)
            with open(self.data_path+"process/interaction_2motifs-other-sampling.pickle", "wb") as f:
                pickle.dump(self.other_motif, f)
        if self.motif_cate == 3 :
            with open(self.data_path+"process/interaction_3motifs-bot-sampling.pickle", "wb") as f:
                pickle.dump(self.bot_motif, f)
            with open(self.data_path+"process/interaction_3motifs-other-sampling.pickle", "wb") as f:
                pickle.dump(self.other_motif, f)
                
ds = DataSampling(motif_cate=3)