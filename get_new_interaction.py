import numpy as np
import pandas as pd
import pickle
import copy
from tqdm import tqdm

def find(ls, item):
    low = 0
    high = len(ls)-1
    while low < high:
        mid = (low + high)//2
        temp = ls[mid]
        if temp == item:
            return mid
        elif temp > item:
            high = mid - 1
        else:
            low = mid + 1
    return False

dirs = "./dataset/Twibot-20/"
# with open (dirs+"interaction_simple_pd.pickle", 'rb') as f:
#     interaction_simple_pd = pickle.load(f)
interaction_simple_pd = pd.read_csv(dirs+"interaction_simple_pd.csv")
with open (dirs+"union_name.pickle", 'rb') as f:
    union_name = pickle.load(f)
with open (dirs+"node_id_sorted.pickle", 'rb') as f:
    node_id_sorted = pickle.load(f)
    
update_row_list = []
for index,row in tqdm(interaction_simple_pd.iterrows()):
    if index%10000 == 0:
        print(index/len(interaction_simple_pd))
    search_obj = row["target_name"]
    target_id = None
    if type(search_obj) == str:
        idx = find(union_name, search_obj)
        if idx != False:
            target_id = node_id_sorted[idx]
            
            new_row = copy.deepcopy(row)
            new_row["target_id"] = target_id
            new_row["target_name"] = search_obj
            update_row_list.append(new_row)
        
    elif type(search_obj) == list:
        target_id = []
        target_name = search_obj
        for so in search_obj:
            idx = find(union_name, so)
            if idx != False:
                target_id.append(node_id_sorted[idx])
            else:
                target_name.remove(so)
    
        if len(target_id) != 0:

            new_row = copy.deepcopy(row)
            new_row["target_id"] = target_id
            new_row["target_name"] = target_name
            update_row_list.append(new_row)
    
    
interaction_simple_pd = pd.DataFrame(update_row_list)
interaction_simple_pd.to_csv(dirs+"interaction_simple_pd.csv")    


