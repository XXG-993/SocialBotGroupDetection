import argparse
import json
import pickle
from typing import Any, List, Optional
import re

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer



node_pd: Optional[pd.DataFrame] = None
retweet_matrix: Optional[np.ndarray] = None
sorted_node_ids: Optional[List[Any]] = None
with_label_json: Optional[list] = None


def load_file(filename) -> np.ndarray:
    cate = filename.split(".")[-1]
    if cate == "pickle":
        with open(filename, "rb") as f:
            return pickle.load(f)
    elif cate == "npy":
        return np.load(filename)
    raise ValueError(filename)


def is_retweet(content):
    pattern = re.compile(r"RT.*?:")
    m = pattern.match(content)
    if m is not None:
        return m.group(0)[4:-1]
    else:
        return False


def ids2names(ids):
    assert node_pd is not None
    names = []
    for id_ in ids:
        name = node_pd.loc[node_pd["id"] == id_]["username"]
        if len(name.values) == 0:
            names.append(np.nan)
        else:
            names.append(list(name.values)[0][:-1])
    return names


def _normalize_dirs(dirs: str) -> str:
    return dirs if dirs.endswith("/") else dirs + "/"


def _load_stance(model_name: str):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def preprocess(text):
        new_text = []
        for t in text.split(" "):
            t = "@user" if t.startswith("@") and len(t) > 1 else t
            t = "http" if t.startswith("http") else t
            new_text.append(t)
        return " ".join(new_text)

    def pos_or_neg(text):
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors="pt")
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        return config.id2label[ranking[0]]

    return pos_or_neg


def get_signed_network(topK: int, pos_or_neg):

    assert retweet_matrix is not None and sorted_node_ids is not None and with_label_json is not None

    retweet_matrix_norm = np.where(retweet_matrix > 0, 1, 0)
    retweeted_users = retweet_matrix_norm.T.sum(axis=1)
    retweeted_users_rank = np.argsort(-retweeted_users)

    all_engaged_users = []
    for ego in retweeted_users_rank[:topK]:
        engaged_users = retweet_matrix_norm.T[ego]
        engaged_users = [i for i, u in enumerate(engaged_users) if u > 0]
        engaged_users = list(set(engaged_users) - set(retweeted_users_rank[:topK]))
        all_engaged_users.extend(engaged_users)
    all_engaged_users = sorted(list(set(all_engaged_users)))
    all_engaged_users.extend(retweeted_users_rank[:topK])
    all_engaged_users_ids = [sorted_node_ids[u] for u in all_engaged_users]
    topK_leader_ids = [sorted_node_ids[l] for l in retweeted_users_rank[:topK]]
    topK_leader_names = ids2names(topK_leader_ids)

    pos_adj = np.zeros((len(all_engaged_users), len(all_engaged_users)))
    neg_adj = np.zeros((len(all_engaged_users), len(all_engaged_users)))

    for u in tqdm(with_label_json):
        user_id = "u" + u["ID"]
        if user_id in all_engaged_users_ids:
            tweet_list = u["tweet"]
            for t in tweet_list:
                retweet_obj_name = is_retweet(t)
                if is_retweet != False:
                    if retweet_obj_name in topK_leader_names:
                        retweet_obj_id = topK_leader_ids[topK_leader_names.index(retweet_obj_name)]
                        content = t[5 + len(retweet_obj_name) :]
                        if pos_or_neg(content) == "negative":
                            neg_adj[
                                all_engaged_users_ids.index(user_id),
                                all_engaged_users_ids.index(retweet_obj_id),
                            ] += 1
                            neg_adj[
                                all_engaged_users_ids.index(retweet_obj_id),
                                all_engaged_users_ids.index(user_id),
                            ] += 1
                        else:
                            pos_adj[
                                all_engaged_users_ids.index(user_id),
                                all_engaged_users_ids.index(retweet_obj_id),
                            ] += 1
                            pos_adj[
                                all_engaged_users_ids.index(retweet_obj_id),
                                all_engaged_users_ids.index(user_id),
                            ] += 1

    return pos_adj, neg_adj


def run(dirs: str, model_name: str) -> None:
    global node_pd, retweet_matrix, sorted_node_ids, with_label_json

    dirs = _normalize_dirs(dirs)
    pos_or_neg = _load_stance(model_name)

    node_pd = pd.read_csv(dirs + "node.csv")
    retweet_matrix = load_file(dirs + "retweet_matrix_label.pickle")
    node_ids = load_file(dirs + "node_list.pickle")
    sorted_node_ids = sorted(node_ids)

    with open(dirs + "test.json", "r", encoding="utf8") as fp:
        test_json = json.load(fp)
    with open(dirs + "train.json", "r", encoding="utf8") as fp:
        train_json = json.load(fp)
    with open(dirs + "support.json", "r", encoding="utf8") as fp:
        support_json = json.load(fp)
    with open(dirs + "dev.json", "r", encoding="utf8") as fp:
        dev_json = json.load(fp)
    with_label_json = test_json + train_json + support_json + dev_json

    for topK in (5, 10, 20):
        print("build graph topK=", topK)
        pos_adj, neg_adj = get_signed_network(topK=topK, pos_or_neg=pos_or_neg)
        np.save(dirs + f"pos_adj_top{topK}.npy", pos_adj)
        np.save(dirs + f"neg_adj_top{topK}.npy", neg_adj)


def main():
    p = argparse.ArgumentParser(description="Build pos/neg signed adjacency .npy (from signed_network.py).")
    p.add_argument("--dirs", type=str, required=True, help="Dataset root (Twibot-20 layout).")
    p.add_argument(
        "--model",
        type=str,
        default="stance_detection_model",
        help="HF model id (original used a local twitter-roberta path).",
    )
    args = p.parse_args()
    run(args.dirs, args.model)


if __name__ == "__main__":
    main()
