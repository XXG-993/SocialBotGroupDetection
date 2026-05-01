import numpy as np
import pandas as pd
import pickle
import copy
import networkx as nx
import random
from MyLogger import MyLogger
from itertools import combinations, permutations
from ReadConfig import ReadConfig

from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl import save_graphs,  load_graphs
from dgl.data.utils import save_info, load_info
from torch.utils.data import DataLoader
from dgl.data import MiniGCDataset
from dgl.nn.pytorch import GraphConv, GATConv, HeteroGraphConv
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from dgl.data import DGLDataset


from DataLoader import BuildData, GraphData, collate
from BotPool_Model import Net
from Evaluation import get_bot_motif,bot_rate

def train(model, data_loader, criterion, optimizer, DEVICE):
    model.train()
    epoch_loss = 0
    for iter, (batchg, label) in enumerate(data_loader):
        batchg, label = batchg.to(DEVICE), label.to(DEVICE)
        prediction = model(batchg)
        # print(prediction.shape, label.shape)
        loss = criterion(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)
    
    return epoch_loss

def test(model, test_loader, DEVICE):
    model.eval()
    test_pred, test_label = [], []
    with torch.no_grad():
        for it, (batchg, label) in enumerate(test_loader):
            batchg, label = batchg.to(DEVICE), label.to(DEVICE)
            pred = torch.softmax(model(batchg), 1)
            pred = torch.max(pred, 1)[1].view(-1)
            test_pred += pred.detach().cpu().numpy().tolist()
            test_label += label.cpu().numpy().tolist()
    return accuracy_score(test_label, test_pred), \
            precision_score(test_label, test_pred), \
            recall_score(test_label, test_pred), \
            f1_score(test_label, test_pred)


def main():

    dataname = "Twibot-20"
    motif_size = 3

    cfg = ReadConfig().read_config()
    logger = MyLogger.__call__().get_logger()
    logger.info("dataname:{dataname}-motif size:{motif_size}".format(dataname=dataname, motif_size=motif_size))

    bd = BuildData(motif_cate=motif_size, dataset=dataname)
    trainset = GraphData(bd.train_data)
    testset = GraphData(bd.test_data)
    print("feature size:",bd.train_data[2][0].shape)

    train_loader = DataLoader(trainset, batch_size=64, shuffle=True,
                            collate_fn=collate)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False,
                            collate_fn=collate)
    
    dropout_rate = 0.2
    epochs = 30
    DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model = Net(64+6, 256, 2, dropout_rate=dropout_rate).to(DEVICE)
    print(model)
    # model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Only perform weight-decay on first convolution.
    criterion = nn.CrossEntropyLoss()
    logger.info("optimizer:Adam-lr:0.001")

    bot_rates = []
    seeds = [1,2,3]

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        model.train()
        for epoch in range(1,epochs+1):
            epoch_loss = train(model, train_loader, criterion, optimizer, DEVICE)
            test_acc, test_pre, test_rec, test_f1 = test(model, test_loader, DEVICE)

            logger.info('Epoch {}, \
                loss {:.4f}, \
                test_acc {:.4f},\
                test_pre {:.4f},\
                test_rec {:.4f},\
                test_f1 {:.4f}'.format(epoch, 
                            epoch_loss, 
                            test_acc,
                            test_pre,
                            test_rec,
                            test_f1))
        

        bot_index = bd.bot_index
        all_data = bd.all_data
        bot_motif_index = get_bot_motif(all_data, model, DEVICE)
        co_edges = np.array(bd.co_edges)
        bot_motif_edges = co_edges[np.array(bot_motif_index)]

        co_graph = nx.Graph()
        for e in tqdm(bot_motif_edges):
            e = zip(e[0],e[1])
            # print(bot_rate([list(e)[0]], bot_index))
            co_graph.add_edges_from(e)
        bot_groups = [co_graph.subgraph(c).copy() for c in nx.connected_components(co_graph)]

        bot_group_len = [len(g.nodes) for g in bot_groups]
        bot_group_rate = bot_rate(bot_groups, bot_index)
        print(bot_group_len)
        print(bot_group_rate)
        bot_rates.append(sum(bot_group_rate)/len(bot_group_rate))

        logger.info('bot_group_len {}, \
            bot_group_rate {}'.format(bot_group_len, 
                        bot_group_rate))

        logger.info('avg len {:.4f}, \
            avg bot rate {:.4f}'.format(sum(bot_group_len)/len(bot_group_len), 
                        sum(bot_group_rate)/len(bot_group_rate)))
    
    print("avg bot rate:", np.array(bot_rates).mean(), "std:", np.array(bot_rates).std())     

if __name__=="__main__":
    main()
 
