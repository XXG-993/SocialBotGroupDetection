from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from AutoEncoder import AutoEncoder
from ReadConfig import ReadConfig
from DataLoader import AutoEncoderDataLoader

class Train:
    
    def __init__(self) -> None:
        # Train.__find_gpu()
        # self.model = AutoEncoder() # type: ignore
        self.model = AutoEncoder()
        self.cfg = ReadConfig().read_config()
        self.epoch = self.cfg['epoch']
        self.lr = self.cfg['lr']
        self.seq_len = self.cfg['seq_len']
        self.train_dataloader = None   
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)     
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mls_loss = torch.nn.MultiLabelSoftMarginLoss()
        self.dataset = "Twibot-20"
        self.prepare_data()
        
    def prepare_data(self):
        train_dataset = AutoEncoderDataLoader(dataset=self.dataset)
        print(train_dataset.__len__())
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.cfg['batch_size'], shuffle=False)
       
    def loss_fn(self, decoded_seq, seqs_origin, y_cate, y_content, label):

        AE = F.mse_loss(decoded_seq, seqs_origin)
        CATE = self.bce_loss(y_cate, label[:,-6:-4])
        CONTENT = self.mls_loss( y_content,label[:,-4:])
        
        return AE + CATE + CONTENT
    
    def accuracy(self, y_cate, y_content, label):

        _, predicted = torch.max(y_cate, dim=1)
        _, label_cate = torch.max(label[:,-6:-4], dim=1)
        crt_cate = (predicted == label_cate).sum().item()
        
        y_content = torch.round(y_content)
        label_cont = label[:,-4:]
        crt_cont = (y_content == label_cont).all(dim=1).sum().item()
        total = label.size(0)
        acc_cate = crt_cate / total
        acc_cont = crt_cont / total
        return acc_cate, acc_cont
    
    def train(self):
        self.model.train()
        self.model = self.model.to(self.device)
        

        for epoch in tqdm(range(self.epoch)):
            loss_0 = 0
            for batch_id, item in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                seqs, label = item
                seqs, label = seqs.to(self.device), label.to(self.device)
                seqs, label = seqs.float(),label.float()
                encoded_seq, decoded_seq, vector, y_cate, y_content = self.model(seqs)


                loss = self.loss_fn(decoded_seq, seqs, y_cate, y_content, label)
                loss_item = loss.item()
                loss_0 += loss_item
                loss.backward()
                
                self.optimizer.step()
                acc_cate, acc_cont = self.accuracy(y_cate,y_content,label)
                print("acc_cate:",acc_cate,"acc_cont",acc_cont)
            print("loss:",loss_0)
        
        
        torch.save(self.model.state_dict(), self.cfg["save_path"]+"/AE-content+cate-"+self.dataset+".pth")
        print("-------finish training--------")
    
    @staticmethod
    def __find_gpu():
        if torch.cuda.is_available():
            os.system("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp")
            mem_ava = [int(x.split()[2]) for x in open("tmp", 'r').readlines()]
            gpu_id = int(np.argmax(mem_ava))
            torch.cuda.set_device(gpu_id)

train = Train()
train.train()            
