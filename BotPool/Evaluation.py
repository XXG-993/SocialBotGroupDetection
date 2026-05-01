import numpy as np
import networkx as nx

import torch
from torch.utils.data import DataLoader

from DataLoader import GraphData, collate
from BotPool_Model import Net

def get_bot_motif(all_data, model, DEVICE):
    model.eval()
    batch_size = 64
    data_loader = DataLoader(GraphData(all_data), batch_size=batch_size, shuffle=False,
                         collate_fn=collate)
    bot_motif_indexs = []
    with torch.no_grad():
        for it, (batchg, label) in enumerate(data_loader):
            batchg, label = batchg.to(DEVICE), label.to(DEVICE)
            pred = torch.softmax(model(batchg), 1)
            pred = torch.max(pred, 1)[1].view(-1)
            bot_motif_index = list(np.where(pred.detach().cpu().numpy()>0)[0])
            bot_motif_index = [i+it*batch_size for i in bot_motif_index]
            bot_motif_indexs.extend(bot_motif_index)

    return bot_motif_indexs

def bot_rate(groups, bot_index):
    g_botrate = []
    for g in groups:
        nodes = set(list(g.nodes))
        g_botrate.append(len(nodes & set(list(bot_index)))/len(nodes))
    return g_botrate