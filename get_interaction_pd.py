import json
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
import re
import pickle

def No_hashtag(content):
    return content.count("#") 

def No_url(content):
    return content.count("https://")

def is_retweet(content):
    pattern = re.compile(r"RT.*?:")
    m = pattern.match(content)
    if m!= None:
        return m.group(0)[4:-1]
    else:
        return False
    
def mention_list(content):
    content = content.replace("\n"," ")
    pattern = re.compile(r"@[a-zA-Z0-9_]*")
    result = pattern.findall(content)
    if result != None:
        result = [m[1:] for m in result]
        if is_retweet(content) == False:
            return result
        else:
            return result[1:]
    else:
        return []

def names2ids(names):
    ids = []
    for name in names:
        id = node_pd.loc[node_pd["username"]==name]["id"]
        if len(id.values) == 0:
            ids.append(np.nan)
        else:
            ids.append(list(id.values)[0])
    return ids

def find_index(node, node_list):
    low = 0
    high = len(node_list)
    mid = 0
    while low < high:
        mid_tmp = mid
        mid = (low + high)//2
        if mid_tmp-mid == 0:
            return "unkwnown"
        temp = node_list[mid]
        if temp == node:
            return mid
        elif temp > node:
            high = mid
        else:
            low = mid
        if low >= high:
            print(node)
    return "unkwnown"

def name_find_id(name):
    
    node_id = find_index(name, list(node_name_id_sorted.keys()))
    if node_id == "unkwnown":
        return "unkwnown"
    else:
        target_id = list(node_name_id_sorted.values())[node_id]
        return find_index(target_id, list(node_id_name_sorted.keys()))
        
    



dirs = "./dataset/Twibot-20/"

with open(dirs+"test.json",'r',encoding='utf8')as fp:
    test_json = json.load(fp)
with open(dirs+"train.json",'r',encoding='utf8')as fp:
    train_json = json.load(fp)
with open(dirs+"support.json",'r',encoding='utf8')as fp:
   support_json = json.load(fp)
with open(dirs+"dev.json",'r',encoding='utf8')as fp:
   dev_json = json.load(fp)
with_label_json = test_json + train_json + support_json + dev_json

del test_json,train_json,support_json,dev_json


node_pd = pd.read_csv(dirs+"node_with_label.csv")
node_ids_label = list(node_pd["id"])
node_names_label = list(node_pd["username"])
node_names_label = [n[:-1] for n in node_names_label]

node_label_dict = {k:v for k,v in zip(node_ids_label,node_names_label)}
node_id_name_sorted = dict(sorted(node_label_dict.items(), key=lambda v:v[0]))
node_label_dict = {k:v for k,v in zip(node_names_label,node_ids_label)}
node_name_id_sorted = dict(sorted(node_label_dict.items(), key=lambda v:v[0]))



retweet_matrix_label = np.zeros((len(node_ids_label),len(node_ids_label)))
mention_matrix_label = np.zeros((len(node_ids_label),len(node_ids_label)))

for u in tqdm(with_label_json):
    user_id = 'u' + u["ID"]
    if "label" in u:
        node_index = find_index(user_id, list(node_id_name_sorted.keys()))
        if node_index != "unkwnown":
            tweet_list = u["tweet"]
            
            if tweet_list != None:
                for t in tweet_list:
                    retweet_obj = is_retweet(t)
                    if retweet_obj != False:
                        target_index = name_find_id(retweet_obj)
                        if target_index != "unkwnown":
                            retweet_matrix_label[node_index,target_index] += 1
                    mention_objs = mention_list(t)
                    if len(mention_objs) != 0:
                        for mo in mention_objs:
                            mo_index = name_find_id(mo)
                            if mo_index != "unkwnown":
                                mention_matrix_label[node_index,mo_index] += 1
                                
with open(dirs+"retweet_matrix_label.pickle","wb") as f:
    pickle.dump(retweet_matrix_label, f)
with open(dirs+"mention_matrix_label.pickle","wb") as f:
    pickle.dump(mention_matrix_label, f)