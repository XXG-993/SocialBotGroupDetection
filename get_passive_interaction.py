import numpy as np
import pandas as pd
import torch
from ast import literal_eval
from tqdm import tqdm

dirs = "./dataset/Twibot-20/"
interaction_simple_pd = pd.read_csv(dirs+"interaction_simple_pd.csv")

tmp_pd_list = []
interaction_passive_simple_pd = pd.DataFrame()
for i,row in tqdm(interaction_simple_pd.iterrows()):    
    
    if row["relation"] == "retweet":
        tmp_pd = dict()
        tmp_pd["relation"] = "retweeted"
        tmp_pd["tweet_id"] = row["tweet_id"]
        tmp_pd["id"] = row["target_id"]
        tmp_pd["action_id"] = row["id"]
        tmp_pd = pd.DataFrame([tmp_pd])
        tmp_pd_list.append(tmp_pd)
    else:
        t_ids = literal_eval(row["target_id"])
        for t_id in t_ids:
            tmp_pd = dict()
            tmp_pd["relation"] = "mentioned"
            tmp_pd["tweet_id"] = row["tweet_id"]
            tmp_pd["id"] = t_id
            tmp_pd["action_id"] = row["id"]
            tmp_pd = pd.DataFrame([tmp_pd])
            tmp_pd_list.append(tmp_pd)
            
    if i % 40000 == 0:
        print(i/len(interaction_simple_pd))

tmp_pd = pd.concat(tmp_pd_list)
interaction_passive_simple_pd = pd.concat([interaction_passive_simple_pd,tmp_pd])
interaction_passive_simple_pd.to_csv(dirs+"process/interaction_passive_simple_pd.csv")