import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
from pathGenerator import pathGenerator
from HyperGraph import HyperGraph
import pickle
from tqdm import tqdm

class SubgraphEncoder():
    
    def __init__(self, walks, d) -> None:
        self.walks = walks
        self.d = d
    
    def learnFeature(self):
        
        model = Word2Vec(sentences=self.walks,
                         vector_size=self.d,
                         min_count=0,
                         sg=1,
                         workers=3)
        f = model.wv
        return f


# test
def get_cos_similar(v1: list, v2: list):
    num = float(np.dot(v1, v2))
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

# retweet_matrix = np.array([[0,1,1,1,1,0,0,0,0,0],
#                            [0,0,0,0,0,0,0,0,0,0],
#                            [0,0,0,0,0,0,0,0,0,0],
#                            [0,0,0,0,0,0,0,0,0,0],
#                            [0,0,0,0,0,0,0,0,0,0],
#                            [0,0,0,0,0,0,1,0,0,0],
#                            [0,0,0,0,0,0,0,0,0,0],
#                            [0,0,0,0,0,0,0,0,1,0],
#                            [0,0,0,0,0,0,0,0,0,0],
#                            [0,0,0,0,0,0,0,0,0,0]])
# nodes = list(range(10))
# edges_weights = {(0,6,8):1,
#                 (1,7,9):1,
#                 (0,6):2,
#                 (1,7):2}

# with open("/home/xxy/project/Coordination-detection/dataset/Twibot-20/process/retweet_matrix_label.pickle", "rb") as f:
#     retweet_matrix = pickle.load(f)
# with open("/home/xxy/project/Coordination-detection/dataset/Twibot-20/process/mention_matrix_label.pickle", "rb") as f:
#     mention_matrix = pickle.load(f)

# retweet_matrix = np.array([[0,1,1,1,1],
#                               [1,0,1,0,0],
#                               [1,1,0,1,0],
#                               [1,0,0,0,0],
#                               [1,0,0,0,0]])
# mention_graph = HyperGraph(nodes,edges_weights)
# mention_graph = HyperGraph(list(range(len(retweet_matrix))),dict())

# pg_ = pathGenerator(mention_graph,retweet_matrix)
# fs = []
# all_walks = []
# for ego in tqdm(range(len(retweet_matrix))):
#     walks = pg_.getPath(ego)
#     # print(ego,walks[0])
#     # all_walks.extend(walks)
#     sg = SubgraphEncoder(walks)
#     f = sg.learnFeature()
#     fs.append(list(f[ego]))
# for i in range(len(retweet_matrix)):
#     fs.append(f[i])
# fs = np.array(fs)
# print(len(f),len(fs))
# print(fs)


# f_0 = f[0]
# cos_matrix = np.zeros_like(retweet_matrix, dtype=np.float32)
# for i in range(retweet_matrix.shape[0]):
#     for j in range(retweet_matrix.shape[1]):
#         cos_matrix[i][j] = get_cos_similar(fs[i],fs[j])
# print(cos_matrix)
# print(cos_matrix)

# walks = pg.getPath(1)
# sg = SubgraphEncoder(walks)
# f = sg.learnFeature()
# f_1 = f[1]
# print(get_cos_similar(list(f_0),list(f_1)))

# fs = []
# for ego in tqdm(range(len(retweet_matrix))):
#     print("walk...")
#     walks = pg.getPath(ego)
#     print("walk already...")
#     sg = SubgraphEncoder(walks)
#     print("learning...")
#     f = sg.learnFeature()
#     fs.append(list(f[ego]))
    
# fs = np.array(fs)
# print(fs)
# np.save("/home/xxy/project/Coordination-detection/dataset/Twibot-20/walk_emb-0-500.npy", fs)
    

