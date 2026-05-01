from ast import literal_eval
import numpy as np
import pandas as pd
import pickle
import torch
from ReadConfig import ReadConfig
from torch.utils.data import Dataset


class AutoEncoderDataLoader(Dataset):
    
    def __init__(self, dataset="Twibot-20") -> None:
        super().__init__()
        self.cfg = ReadConfig().read_config()
        self.dataset = dataset
        if dataset == "Twibot-20":
            self.data_path = self.cfg["Twibot20_data_path"]
        elif dataset == "Renmin":
            self.data_path = self.cfg["Renmin_data_path"]
        
        self.tweet_emb = self.load_file(self.data_path+"tweet_emb.pickle")
        self.cate_encode = self.load_file(self.data_path+"cate_encode.pickle")
        self.data = self.padding()
        # self.label = torch.tensor(self.load_file(self.data_path+"node_labels.npy"))
        # self.df = pd.read_csv(self.data_path + "history_with_label-2.csv")
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label = self.data[idx,-1,:]
        seqs = self.data[idx,:-1,:]
        return seqs, label
    
    def get_all(self):
        label = self.data[:,-1,:]
        seqs = self.data[:,:-1,:]
        return seqs, label
    
    def load_file(self, filename):
        cate = filename.split(".")[-1]
        if cate == "pickle":
            with open(filename, "rb") as f:
                return pickle.load(f)
        elif cate == "npy":
                return np.load(filename)
        
    def padding(self):
        tweet_data = np.zeros((len(self.tweet_emb),200,768))
        for i,l in enumerate(self.tweet_emb):
            if len(l) > 0:
                l = np.flipud(np.array(l))
                if len(l) > 200:
                    l = l[len(l)-200:]
                tweet_data[i,-len(l):] = l
        cate_data = np.zeros((len(self.cate_encode),200,6))
        for i,l in enumerate(self.cate_encode):
            if self.dataset == "Twibot-20":
                l = self.cate_encode[l]
            if len(l) > 0:
                l = np.flipud(np.array(l))
                if len(l) > 200:
                    l = l[len(l)-200:]
                cate_data[i,-len(l):] = l
        data = np.concatenate((tweet_data, cate_data), axis=2)
        return torch.tensor(data)
