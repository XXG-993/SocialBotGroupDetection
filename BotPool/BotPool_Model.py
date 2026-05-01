import numpy as np
import pandas as pd
import networkx as nx
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, GATConv, HeteroGraphConv


random.seed(12345)
np.random.seed(12345)
torch.manual_seed(12345)

# class InterGATLayer(torch.nn.Module):
#     def __init__(self, in_dim, hidden_dim):
#         super(InterGATLayer,self).__init__()
#         # self.in_channels = in_channels
#         self.hgnn1 = HeteroGraphConv({
#                                     "retweeted": GATConv(hidden_dim, hidden_dim, num_heads=1, allow_zero_in_degree=True, feat_drop=dropout_rate ),
#                                     "mentioned": GATConv(hidden_dim, hidden_dim, num_heads=1, allow_zero_in_degree=True, feat_drop=dropout_rate )},
#                                      aggregate="mean")
#         self.hgnn2 = HeteroGraphConv({
#                                     "retweeted": GATConv(hidden_dim, hidden_dim, num_heads=1, allow_zero_in_degree=True, feat_drop=dropout_rate ),
#                                     "mentioned": GATConv(hidden_dim, hidden_dim, num_heads=1, allow_zero_in_degree=True, feat_drop=dropout_rate )},
#                                      aggregate="mean")
        
#     def forward(self, g):

#         h = {'user':g.nodes["user"].data["h"], 
#              'ego':g.nodes["ego"].data["h"]}
#         ego_h = h["ego"][0]
#         h1 = self.hgnn1(g, h)  
#         h1 = {k:F.relu(v) for k, v in h1.items()} 
#         h1 = {k:h1[k]+ego_h for k in h1.keys()}  
        
#         # ego_h = h1["ego"][0]
#         # h2 = self.hgnn2(g, h1)  
#         # h2 = {k:F.relu(v) for k, v in h2.items()} 
#         # h2 = {k:h2[k]+ego_h for k in h2.keys()} 
#         # g.nodes["user"].data["h"] = h1["user"]
#         # g.nodes["ego"].data["h"] = h1["ego"]
#         return h1
    
class InterGAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(InterGAT,self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        
    def forward(self, g):

        h = {'user':g.nodes["user"].data["h"], 
             'ego':g.nodes["ego"].data["h"]}
        sgs_, egos = self.build_subgraph(g)
        # self.convlayers = []
        # for i in range(len(sgs_)):
        #     self.convlayers.append(InterGATLayer(self.in_dim, self.hidden_dim).to(g.device))
        
        hgs = []
        for i in range(len(sgs_)):
            h = {'user':sgs_[i].nodes["user"].data["h"], 
             'ego':sgs_[i].nodes["ego"].data["h"]}
            # h = self.convlayers[i](sgs_[i])
            sgs_[i].nodes["user"].data["h"] = h["user"]
            sgs_[i].nodes["ego"].data["h"] = h["ego"]
            hg_user = dgl.readout_nodes(sgs_[i], 'h', op="mean", ntype="user")
            hg_ego = dgl.readout_nodes(sgs_[i], 'h', op="mean", ntype="ego")
            hg = hg_user + hg_ego
            hgs.append(hg.squeeze(dim=1))
        hgs = torch.stack(hgs)
        hgs = hgs.transpose(0,1).reshape(-1, self.hidden_dim)
        new_g = self.connect_supernode(g, hgs, egos)
        
        return new_g
    
    def build_subgraph(self, g):
        
        ug = dgl.unbatch(g)
        egos = list(ug[0].nodes("ego"))
        sgs_ = [[] for _ in range(len(egos))]
        
        for i in range(g.batch_size):
            for j,ego in enumerate(egos):
                ego = ego.cpu().detach().item()
                nodes = dict()
                nodes["user"] = torch.tensor([n for n in ug[i].nodes("user") if ug[i].nodes["user"].data[str(ego)+"_"][n] > 0]).to(g.device)
                nodes["ego"] = torch.tensor([n for n in ug[i].nodes("ego") if ug[i].nodes["ego"].data[str(ego)+"_"][n] > 0]).to(g.device)
                new_sg = dgl.node_subgraph(ug[i], nodes)     
                new_node_dict = {int(nodes["ego"][index].cpu().detach()):\
                                int(new_sg.nodes("ego")[index].cpu().detach()) \
                                for index in range(len(nodes["ego"]))}  

                for n in nodes["ego"]:
                    n = n.cpu().detach().item()
                    removed_ego = []
                    if n != ego:
                        attr = {k:torch.unsqueeze(v[new_node_dict[n]], dim=0) for k,v in new_sg.nodes["ego"][0].items()}
                        old_user_index = set([int(n) for n in new_sg.nodes("user")])
                        new_sg.add_nodes(1, data=attr, ntype='user')
                        new_node = list(set([int(n) for n in new_sg.nodes("user")])-old_user_index)[0]

                        
                        for inter_type in ["retweeted", "mentioned"]:
                            edges = [None, None]
                            indexs = torch.where(ug[i].edges(etype=("ego",inter_type,"ego"))[1]==n)
                            edges[0] = ug[i].edges(etype=("ego",inter_type,"ego"))[0][indexs]
                            edges[1] = torch.ones_like(edges[0])*new_node
                            if len(edges[0]) > 0:
                                new_sg.add_edges(edges[0], edges[1], etype=("ego",inter_type,"user"))
                            
                            edges = [None, None]
                            indexs = torch.where(ug[i].edges(etype=("ego",inter_type,"ego"))[0]==n)
                            edges[1] = ug[i].edges(etype=("ego",inter_type,"ego"))[1][indexs]
                            edges[0] = torch.ones_like(edges[1])*new_node
                            if len(edges[0]) > 0:
                                new_sg.add_edges(edges[0], edges[1], etype=("user",inter_type,"ego"))
                        removed_ego.append(new_node_dict[n])
                new_sg.remove_nodes(torch.tensor(removed_ego, dtype=torch.int64).to(g.device), ntype="ego")        
                sgs_[j].append(new_sg)
        
        sgs_ = [dgl.batch(gs) for gs in sgs_]
        return sgs_, egos
    
    def connect_supernode(self, g, hgs, egos):
        
        ug = dgl.unbatch(g)
        new_gs = []
        for i in range(g.batch_size):
            edge = [[],[]]
            for j in range(len(egos)):
                for k in range(len(egos)):
                    if k != j:
                        k_neigh = self.neighbor_set(ug[i], egos[k])
                        j_neigh = self.neighbor_set(ug[i], egos[j]) 
                        if len(k_neigh & j_neigh) > 0:
                            edge[0].extend([egos[j], egos[k]])
                            edge[1].extend([egos[k], egos[j]])
            edge = tuple([torch.tensor(e).int() for e in edge])
            new_g = dgl.graph(edge)
            new_gs.append(new_g)
        
        new_gs = dgl.batch(new_gs).to(g.device)
        new_gs.ndata["h"] = hgs.to(g.device)
        return new_gs

    def neighbor_set(self, g, i):
        neighs = set()
        for etype in g.canonical_etypes:
            if etype[2] == "ego":
                neigh = g.predecessors(i, etype=etype)
                neigh = set([int(n.cpu().detach()) for n in neigh])
                neighs = neighs | neigh
        return neighs
        


class Net(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, dropout_rate):
        super(Net, self).__init__()
        

        self.pool1 = InterGAT(hidden_dim, hidden_dim)
        
        self.hgnn1 = HeteroGraphConv({
                                    "retweeted": GATConv(in_dim, hidden_dim, num_heads=1, allow_zero_in_degree=True, feat_drop=dropout_rate, residual=True ),
                                    "mentioned": GATConv(in_dim, hidden_dim, num_heads=1, allow_zero_in_degree=True, feat_drop=dropout_rate, residual=True )},
                                     aggregate="mean")
        self.hgnn2 = HeteroGraphConv({
                                    "retweeted": GATConv(hidden_dim, hidden_dim, num_heads=1, allow_zero_in_degree=True, feat_drop=dropout_rate, residual=True ),
                                    "mentioned": GATConv(hidden_dim, hidden_dim, num_heads=1, allow_zero_in_degree=True, feat_drop=dropout_rate, residual=True )},
                                     aggregate="mean")
        
        self.conv3 = GraphConv(hidden_dim, hidden_dim, norm='both', allow_zero_in_degree=True )
        self.conv4 = GraphConv(hidden_dim, hidden_dim, norm='both', allow_zero_in_degree=True )

        self.classifier = torch.nn.Linear(3*hidden_dim, n_classes)

    def forward(self, g):

        h = {'user':g.nodes["user"].data["x"], 
             'ego':g.nodes["ego"].data["x"]}
        
        h1 = self.hgnn1(g, h)  
        h1_ = {k:F.relu(v) for k, v in h1.items()}  
        h1 = self.hgnn2(g, h1_) 
        h1 = {k:F.relu(v) for k, v in h1.items()}  
        g.nodes["user"].data["h"] = h1["user"]
        g.nodes["ego"].data["h"] = h1["ego"]
        hg1 = dgl.readout_nodes(g, 'h', op="mean", ntype="ego")
        hg1 = torch.squeeze(hg1, dim=1)
        
        new_g = self.pool1(g)
        hg2 = dgl.mean_nodes(new_g, 'h')
        hg2 = torch.unsqueeze(hg2, dim=1)
        
        h2 = new_g.ndata["h"]
        h2_ = F.relu(self.conv3(new_g, h2))
        h2 = F.relu(self.conv4(new_g, h2_))
        new_g.ndata["h"] = h2_
        hg3 = dgl.mean_nodes(new_g, 'h')
        hg3 = torch.unsqueeze(hg3, dim=1)
        # print(hg1.shape, hg2.shape, hg3.shape)
        
        hg = torch.cat((hg1, hg2, hg3), dim=2)
        
        hg = self.classifier(hg)
        hg = torch.squeeze(hg, dim=1)

        return hg