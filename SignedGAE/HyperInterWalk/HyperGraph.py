import numpy as np


class HyperGraph:
    
    def __init__(self,
                 nodes:list,
                 edges_weight:dict) -> None:
        
        self.nodes = nodes
        self.N = len(self.nodes)
        self.edges_weights = edges_weight
        self.edges = list(self.edges_weights.keys())
        self.edge_matrix = self.hyperEdges()
        
    def hyperEdges(self):
        
        if len(self.edges_weights) == 0:
            return np.zeros([len(self.nodes),1])
            
        hyper_edge = np.zeros([len(self.edges),len(self.nodes)])
        for i in range(len(self.edges)):
            for n in self.edges[i]:
                j = self.nodes.index(n)
                hyper_edge[i,j] = self.edges_weights[self.edges[i]]
                
        return hyper_edge.T   
    
    