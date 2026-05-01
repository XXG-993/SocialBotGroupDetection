import random

import numpy as np
import pandas as pd
from HyperGraph import HyperGraph
from tqdm import tqdm
import pickle
import copy

class pathGenerator:
    
    def __init__(self, 
                 mention_graph:HyperGraph,
                 retweet_matrix) -> None:
        
        self.size = 1
        self.N = 11826
        self.STAY = -1
        self.BACK  = -2
        self.MENTION = -3
        self.RETWEET = -4
        
        self.mention_graph = mention_graph
        self.mention_matrix = self.mention_graph.edge_matrix
        self.retweet_matrix = retweet_matrix
        
        self.ego_mention_matrix = self.mention_matrix
        self.ego_retweet_matrix = np.zeros_like(self.retweet_matrix)
        
        self.ep = 0.5
        self.b = 0.2
        
        self.walk_length = 20
        self.walk_times = 50
        
    def get_ego_network(self, ego, matrix):
        
        neighbors = []
        ego_1_neighbor = list(set(np.where(matrix[ego]>0)[0]))
        if len(ego_1_neighbor)==0:
            return np.zeros_like(matrix)
        for n in ego_1_neighbor:
            ego_2_neighbor = list(set(np.where(matrix[n]>0)[0]))
            neighbors.extend(ego_2_neighbor)
        neighbors = np.array(list(set(neighbors)))
  
        ego_matrix = np.zeros_like(matrix)
        ego_matrix[neighbors,:] = matrix[neighbors,:]
        ego_matrix[:,neighbors] = matrix[:,neighbors]
        
        return ego_matrix
        
    
    def transferMatrix(self):
        pass
    
    def chooseRelation(self, node):
        
        mention_sum = self.ego_mention_matrix[node].sum()
        retweet_sum = self.ego_retweet_matrix[node].sum()
        
        if  mention_sum == 0 and retweet_sum == 0:
            return "None"
        elif mention_sum == 0 and retweet_sum != 0:
            return "retweet"
        elif mention_sum != 0 and retweet_sum == 0:
            return "mention"
        
        if random.random() <= self.ep:
            return "mention"
        else:
            return "retweet"
    
    
    def isBack(self, pre_node, cur_node):
        
        if random.random() < self.b and pre_node != -1:
            return True
        else:
            choose = self.chooseRelation(cur_node)
            if choose == "None" and pre_node != -1:
                return True
            elif choose == "None" and pre_node == -1:
                return False
            else:
                return choose
        
    def mentionSelect(self, node):
        
        choose_edges = self.ego_mention_matrix[node]
        choose_edges = choose_edges / choose_edges.sum()
        edge_index = np.random.choice(list(range(len(choose_edges))), p=choose_edges.ravel())
        
        edge = self.mention_graph.edges[edge_index]
        edge = list(edge)
        edge.remove(node)
        node_index = np.random.choice(edge, p=[1/len(edge)]*(len(edge)))
        
        return node_index
    
    def retweetSelect(self, node):
        
        choose_nodes = self.ego_retweet_matrix[node]
        neighbor_nodes = np.where(choose_nodes>0)[0]
        choose_nodes = choose_nodes / choose_nodes.sum()
        choose_nodes = np.delete(choose_nodes, np.where(choose_nodes == 0))
        node_index = np.random.choice(neighbor_nodes, p=choose_nodes.ravel())
                
        return node_index
        
    
    def nextNode(self, pre_node, cur_node) -> list:
        choose = self.isBack(pre_node, cur_node)
        if choose == True:
            return [self.BACK, pre_node]
        elif choose == False:
            return [self.STAY, cur_node]
        elif choose == "mention":
            next_node = self.mentionSelect(cur_node)
            return [self.MENTION, next_node]
        elif choose == "retweet":
            next_node = self.retweetSelect(cur_node)
            return [self.RETWEET, next_node]
    
    def getPath(self, ego):
        
        self.ego_retweet_matrix = self.get_ego_network(ego, self.retweet_matrix.T)
        walk_seqs = []
        for wt in range(self.walk_times):
            walk = [ego]
            new_walk = self.nextNode(-1, walk[-1])
            if new_walk[0] != self.STAY:
                walk.extend(new_walk)
                for wl in range(self.walk_length-2):
                    walk.extend(self.nextNode(walk[-3],walk[-1]))
                    # walk.extend(self.nextNode(walk[-3],walk[-1])) 
            else:
                walk.extend(new_walk*(self.walk_length-1))
            walk_seqs.append(walk)
        
        return walk_seqs



def load_data(pth:str):
    print("load "+pth.split("/")[-1])
    with open(pth, "rb") as f:
        return pickle.load(f)            
