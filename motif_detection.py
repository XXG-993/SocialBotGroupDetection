
#! /home/xxy/anaconda3/envs/py38/bin/python

import networkx as nx
import numpy as np
from scipy import sparse
import time
import pandas as pd
from tqdm import tqdm
import pickle
import copy
from itertools import combinations, permutations

np.random.seed(123)


# def motif_parallel(adj,node_list):

#     time1 = time.time()
#     motifs = [[],[],[]]
#     for i in tqdm(node_list):
#         motif_1, motif_2, motif_3 = motif_find1(adj,i)
#         motifs[0].extend(motif_1)
#         motifs[1].extend(motif_2)
#         motifs[2].extend(motif_3)
        
#     time2 = time.time()
#     # motifs = frozenset(motifs)
#     motifs = [frozenset(m) for m in motifs]
#     print([len(m) for m in motifs])
#     print('time:',time2-time1)
#     return motifs

# # (a,b) a-b
# def motif_find1(adj, node):
#     '''
#     Enumerate motif type 1 starting from node.
#     :param adj:
#     :param node:
#     :return:
#     '''
#     motif_1, motif_2, motif_3,= [],[],[]
#     node2 = motif_part1(adj, node)
#     for i in node2:
#         if i != node:
#             tmp = []
#             tmp.append(node)
#             tmp.append(i)
#             is_i_bot = i in bot_index
#             if node in bot_index:
#                 if is_i_bot:
#                     motif_3.append(frozenset(tmp))
#                 else:
#                     motif_2.append(frozenset(tmp))
#             else:
#                 if is_i_bot:
#                     motif_2.append(frozenset(tmp))
#                 else:
#                     motif_1.append(frozenset(tmp))                    

#     return motif_1, motif_2, motif_3

def motif_parallel(adj,node_list):
    '''
    Parallel motif enumeration over node_list.
    :param adj: lil_matrix (efficient row slicing)
    :param node_list: nodes each worker starts from
    :return:
    '''
    '''for i in node_list:
        tmp = motif_find3(adj,i)'''
    time1 = time.time()
    motifs = [[],[],[],[],[],[],[],[],[],[]]
    for i in tqdm(node_list):
        motif_1, motif_2, motif_3, motif_4, motif_5, motif_6 = motif_find1(adj,i)
        motifs[0].extend(motif_1)
        motifs[1].extend(motif_2)
        motifs[2].extend(motif_3)
        motifs[3].extend(motif_4)
        motifs[4].extend(motif_5)
        motifs[5].extend(motif_6)
        
        motif_7, motif_8, motif_9, motif_10 = motif_find2(adj,i)
        motifs[6].extend(motif_7)
        motifs[7].extend(motif_8)
        motifs[8].extend(motif_9)
        motifs[9].extend(motif_10)
        
    time2 = time.time()
    # motifs = frozenset(motifs)
    motifs = [frozenset(m) for m in motifs]
    print([len(m) for m in motifs])
    print('time:',time2-time1)
    return motifs

# (a,b,c) a-b,a-c
def motif_find1(adj, node):
    '''
    Motif family 1 from node: (a,b,c) with a-b and a-c, no b-c.
    :param adj:
    :param node:
    :return:
    '''
    motif_1, motif_2, motif_3, motif_4, motif_5, motif_6,= [],[],[],[],[],[]
    node2 = motif_part1(adj, node)
    node3 = motif_part1(adj, node)
    for i in node2:
        for j in node3:
            if adj[i, j] == 0 and adj[j, i] == 0 and i != j:
                tmp = []
                tmp.append(node)
                tmp.append(i)
                tmp.append(j)
                is_i_bot = i in bot_index
                is_j_bot = j in bot_index
                if node in bot_index:
                    # abc=bot
                    if is_i_bot and is_j_bot:
                        motif_6.append(frozenset(tmp))
                    # a=bot, bc=human
                    elif not is_i_bot and not is_j_bot:
                        motif_2.append(frozenset(tmp))
                    else:
                        motif_4.append(frozenset(tmp))
                else:
                    # bc=bot
                    if is_i_bot and is_j_bot:
                        motif_5.append(frozenset(tmp))
                    # abc=human
                    elif not is_i_bot and not is_j_bot:
                        motif_1.append(frozenset(tmp))
                    else:
                        motif_3.append(frozenset(tmp))                    

    return motif_1, motif_2, motif_3, motif_4, motif_5, motif_6

# (a,b,c) a-b,b-c,c-a
def motif_find2(adj, node):
    '''
    Motif family 2 from node: (a,b,c) with a-b, b-c, c-a (triangle).
    :param adj:
    :param node:
    :return:
    '''
    motif_7, motif_8, motif_9, motif_10 = [], [], [], []
    node2 = motif_part1(adj, node)
    node3 = motif_part1(adj, node)
    for i in node2:
        for j in node3:
            if adj[i, j] == 1 and adj[j, i] == 1 and i != j:
                is_i_bot = i in bot_index
                is_j_bot = j in bot_index
                tmp = []
                tmp.append(node)
                tmp.append(i)
                tmp.append(j)
                if node in bot_index:
                    # abc=bot
                    if is_i_bot and is_j_bot:
                        motif_10.append(frozenset(tmp))
                    # a=bot, bc=human
                    elif not is_i_bot and not is_j_bot:
                        motif_8.append(frozenset(tmp))
                    else:
                        motif_9.append(frozenset(tmp))
                else:
                    if not is_i_bot and not is_j_bot:
                        motif_7.append(frozenset(tmp))

    
    return motif_7, motif_8, motif_9, motif_10


def motif_part1(adj, node):
    '''
    Neighbors j with both directed edges j->node and node->j (mutual).
    :param adj:
    :param node:
    :return:
    '''
    tmp_list = []
    nozero = list(adj.rows)
    for i in nozero[node]:
        if adj[i, node] != 0:
            # print('step1:({},{}) {}; ({},{}) {}'.format(node, i, adj[node, i], i, node, adj[i, node]))
            tmp_list.append(i)
    return tmp_list


def load_data(pth:str):
    print("load "+pth.split("/")[-1])
    with open(pth, "rb") as f:
        return pickle.load(f)

def find_index(query, value):
    index_list = []
    for q in tqdm(query):
        index_list.extend(list(np.where(np.array(value) == q)[0]))
    return set(index_list)

def drop_users(nodes, matrix:np.array):
    
    new_matrix = copy.deepcopy(matrix)
    nodes = np.array(list(nodes)) 
    new_matrix[nodes,:] = np.zeros((len(nodes),len(matrix)))
    new_matrix[:,nodes] = np.zeros((len(matrix),len(nodes)))
    
    return new_matrix

def coordination_matrix(matrix:np.array):
    c_matrix = np.zeros_like(matrix)
    
    for m in matrix.T:
        u_ids = np.where(m > 0)[0]
        nodes_list = list(combinations(u_ids, 2))
        for n in nodes_list:
            c_matrix[int(n[0]),int(n[1])] += 1
            c_matrix[int(n[1]),int(n[0])] += 1
    return c_matrix

if __name__ == '__main__':
    #max_weight
    
    dirs =  "./dataset/Twibot-20/"
    retweet_matrix = load_data(dirs+"retweet_matrix_label.pickle")
    mention_matrix = load_data(dirs+"mention_matrix_label.pickle")
    
    retweet_matrix[np.eye(len(retweet_matrix),dtype=np.bool_)] = 0
    mention_matrix[np.eye(len(retweet_matrix),dtype=np.bool_)] = 0
    
    co_retweet_matrix = coordination_matrix(retweet_matrix)
    co_mention_matrix = coordination_matrix(mention_matrix)
    
    random_k = 200
    co_retweet_matrix = co_retweet_matrix[:random_k,:][:,:random_k]
    co_mention_matrix = co_mention_matrix[:random_k,:][:,:random_k]
    
    node_ids = load_data(dirs+"node_list.pickle")
    labels = np.load(dirs+"node_labels.npy")
    
    bot_index = np.where(labels>0)[0]
    human_index = np.where(labels==0)[0]
 
    print("interaction_3motifs")    
    G3 = nx.from_numpy_array(co_retweet_matrix+co_mention_matrix)
    G3 = nx.convert_node_labels_to_integers(G3)
    adj = sparse.lil_matrix(co_retweet_matrix+co_mention_matrix)
    node_list = nx.nodes(G3)
    motifs = motif_parallel(adj, node_list)
    with open(dirs+"interaction_3motifs-200.pickle", "wb") as f:
        pickle.dump(motifs, f)
        
        
    # print("retweet_3motifs")
    # G1 = nx.from_numpy_array(co_retweet_matrix)
    # G1 = nx.convert_node_labels_to_integers(G1)
    # adj = nx.adjacency_matrix(G1)
    # adj = sparse.lil_matrix(co_retweet_matrix)
    # node_list = nx.nodes(G1)
    # motifs = motif_parallel(adj, node_list)
    # with open(dirs+"retweet_3motifs-1000.pickle", "wb") as f:
    #     pickle.dump(motifs, f)
    
    # print("mention_3motifs")
    # G2 = nx.from_numpy_array(co_mention_matrix)
    # G2 = nx.convert_node_labels_to_integers(G2)
    # adj = sparse.lil_matrix(co_mention_matrix)
    # node_list = nx.nodes(G2)
    # motifs = motif_parallel(adj, node_list)
    # with open(dirs+"mention_3motifs-1000.pickle", "wb") as f:
    #     pickle.dump(motifs, f)
    
