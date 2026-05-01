import argparse
import datetime
import json
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import No_hashtag, No_url, find_index, is_retweet, mention_list


def build_sorted_train_labels(train_pd: pd.DataFrame):

    node_id_with_label = list(train_pd["ID"])
    node_id_with_label = ["u" + str(n) for n in node_id_with_label]
    node_label_with_label = list(train_pd["Label"])
    node_id_label_with_label_dict = {
        id_: label for id_, label in zip(node_id_with_label, node_label_with_label)
    }
    node_id_label_with_label_dict_sorted = dict(
        sorted(node_id_label_with_label_dict.items(), key=lambda x: x[0], reverse=False)
    )
    node_id_with_label_sorted = list(node_id_label_with_label_dict_sorted.keys())
    node_label_with_label_sorted = list(node_id_label_with_label_dict_sorted.values())
    return node_id_with_label_sorted, node_label_with_label_sorted


def build_node_with_label_rows(
    user_json: list,
    node_id_with_label_sorted: list,
    node_label_with_label_sorted: list,
) -> list:
    
    rows = []
    for u in tqdm(user_json):
        user_id = "u" + u["ID"]
        node_id_index = find_index(user_id, node_id_with_label_sorted)
        if node_id_index != "unknown":
            user_profile = u["profile"]
            if u["profile"] is not None:
                username = user_profile["screen_name"]
                if username is None:
                    continue
                location = user_profile["location"]
                verified = user_profile["verified"]
                followers_count = user_profile["followers_count"]
                following_count = user_profile["friends_count"]
                created_at = user_profile["created_at"]
                listed_count = user_profile["listed_count"]
                description = user_profile["description"]
                tweet_count = user_profile["statuses_count"]
                label = node_label_with_label_sorted[node_id_index]
                row = [
                    user_id,
                    username,
                    location,
                    verified,
                    followers_count,
                    following_count,
                    created_at,
                    tweet_count,
                    listed_count,
                    description,
                    label,
                ]
                rows.append(row)
    return rows


def save_node_with_label_csv(dirs: str, rows: list) -> pd.DataFrame:

    node_pd = pd.DataFrame(rows)
    node_pd.columns = [
        "id",
        "username",
        "location",
        "verified",
        "followers_count",
        "following_count",
        "created_at",
        "tweet_count",
        "listed_count",
        "description",
        "label",
    ]
    node_pd.to_csv(dirs + "node_with_label.csv")
    return node_pd


def calculate_days(times: str) -> int:
    sample_time = "%a %b %d %H:%M:%S +0000 %Y"
    date = datetime.datetime.strptime(times, sample_time)
    basetime = datetime.datetime.strptime("Sat Jan 01 00:00:00 +0000 2021", sample_time)
    return (basetime - date).days


def load_renmin_train_and_user(dirs: str):
    train_pd = pd.read_csv(dirs + "train.csv", sep="\t")
    with open(dirs + "user.json", "r", encoding="utf8") as fp:
        user_json = json.load(fp)
    return train_pd, user_json


def build_node_id_name_maps(node_pd: pd.DataFrame):
    node_ids_label = list(node_pd["id"])
    node_names_label = list(node_pd["username"])
    node_ids_label_sorted = sorted(node_ids_label)

    node_label_dict = {k: v for k, v in zip(node_ids_label, node_names_label)}
    node_id_name_sorted = dict(sorted(node_label_dict.items(), key=lambda v: v[0]))
    node_label_dict = {k: v for k, v in zip(node_names_label, node_ids_label)}
    node_name_id_sorted = dict(sorted(node_label_dict.items(), key=lambda v: v[0]))

    return (
        node_ids_label,
        node_ids_label_sorted,
        node_id_name_sorted,
        node_name_id_sorted,
    )


def save_attr_emb_npy(dirs: str, node_pd: pd.DataFrame) -> None:
    node_attr_np = node_pd[
        [
            "created_at",
            "verified",
            "followers_count",
            "following_count",
            "tweet_count",
            "listed_count",
        ]
    ].to_numpy()
    node_attr_np[:, 0] = np.array([calculate_days(t) for t in node_attr_np[:, 0]])
    np.save(dirs + "attr_emb.npy", node_attr_np)


def save_node_labels_npy_and_bot_human_pickles_renmin(
    dirs: str, train_pd: pd.DataFrame, node_id_name_sorted: dict
) -> None:

    node_id_sorted = list(node_id_name_sorted.keys())
    node_label_sorted = [
        train_pd.loc[train_pd["ID"] == int(i[1:])]["Label"].values[0] for i in node_id_sorted
    ]

    bot_index = [i for i, l in enumerate(node_label_sorted) if l == 1]
    human_index = [i for i, l in enumerate(node_label_sorted) if l == 0]
    bot_ids = [node_id_sorted[i] for i in bot_index]
    human_ids = [node_id_sorted[i] for i in human_index]

    np.save(dirs + "node_labels.npy", node_label_sorted)
    proc = dirs + "process/"
    with open(proc + "bot_ids.pickle", "wb") as f:
        pickle.dump(bot_ids, f)
    with open(proc + "human_ids.pickle", "wb") as f:
        pickle.dump(human_ids, f)


def build_interaction_with_label_csv_renmin(
    dirs: str,
    user_json: list,
    node_ids_label: list,
    node_ids_label_sorted: list,
    node_name_id_sorted: dict,
) -> None:

    t_id = 0
    interaction_with_label_list = []

    for u in tqdm(user_json):
        user_id = "u" + u["ID"]
        if u["profile"] is not None and user_id in node_ids_label:
            tweet_list = u["tweet"]
            if tweet_list is not None:
                for t in tweet_list:
                    no_hashtag = No_hashtag(t)
                    no_url = No_url(t)
                    retweet_obj = is_retweet(t)
                    mention_objs = mention_list(t)

                    if retweet_obj is not False:
                        if find_index(retweet_obj, list(node_name_id_sorted.keys())) != "unknown":
                            target_id = node_name_id_sorted[retweet_obj]
                            target_index = find_index(target_id, node_ids_label_sorted)
                            interaction_dict = {
                                "id": user_id,
                                "relation": "retweet",
                                "target_id": target_id,
                                "tweet_id": "t" + str(t_id),
                                "hashtag_nums": no_hashtag,
                                "url_nums": no_url,
                            }
                            interaction_with_label_list.append(interaction_dict)

                    if len(mention_objs) != 0:
                        mention_ids = [
                            node_name_id_sorted[m]
                            for m in mention_objs
                            if find_index(m, list(node_name_id_sorted.keys())) != "unknown"
                        ]
                        if len(mention_ids) != 0:
                            interaction_dict = {
                                "id": user_id,
                                "relation": "mention",
                                "target_id": mention_ids,
                                "tweet_id": "t" + str(t_id),
                                "hashtag_nums": no_hashtag,
                                "url_nums": no_url,
                            }
                            interaction_with_label_list.append(interaction_dict)

                    t_id += 1

    interaction_with_label_pd = pd.DataFrame(interaction_with_label_list)
    interaction_with_label_pd.to_csv(dirs + "interaction_with_label.csv")



def load_twibot_with_label_json(dirs: str) -> list:
    with open(dirs + "test.json", "r", encoding="utf8") as fp:
        test_json = json.load(fp)
    with open(dirs + "train.json", "r", encoding="utf8") as fp:
        train_json = json.load(fp)
    with open(dirs + "support.json", "r", encoding="utf8") as fp:
        support_json = json.load(fp)
    with open(dirs + "dev.json", "r", encoding="utf8") as fp:
        dev_json = json.load(fp)
    return test_json + train_json + support_json + dev_json


def build_interaction_with_label_csv_twibot(dirs: str) -> None:
    with_label_json = load_twibot_with_label_json(dirs)
    t_id = 0
    interaction_with_label_list = []

    for u in tqdm(with_label_json):
        user_id = "u" + u["ID"]
        if u["profile"] is not None and u["profile"]["screen_name"] is not None:
            user_name = u["profile"]["screen_name"][:-1]
        else:
            user_name = np.nan
        if "label" in u:
            tweet_list = u["tweet"]
            if tweet_list is not None:
                for t in tweet_list:
                    no_hashtag = No_hashtag(t)
                    no_url = No_url(t)
                    retweet_obj = is_retweet(t)
                    mention_objs = mention_list(t)

                    if retweet_obj is not False:
                        interaction_dict = {
                            "id": user_id,
                            "username": user_name,
                            "relation": "retweet",
                            "target_name": retweet_obj,
                            "tweet_id": "t" + str(t_id),
                            "hashtag_nums": no_hashtag,
                            "url_nums": no_url,
                        }
                        interaction_with_label_list.append(interaction_dict)

                    if len(mention_objs) != 0:
                        interaction_dict = {
                            "id": user_id,
                            "username": user_name,
                            "relation": "mention",
                            "target_name": mention_objs,
                            "tweet_id": "t" + str(t_id),
                            "hashtag_nums": no_hashtag,
                            "url_nums": no_url,
                        }
                        interaction_with_label_list.append(interaction_dict)

                    t_id += 1

    interaction_with_label_pd = pd.DataFrame(interaction_with_label_list)
    interaction_with_label_pd.to_csv(dirs + "interaction_with_label.csv")


# --- test_data.ipynb ---


def save_bot_human_pickles_from_label_csv(dirs: str) -> None:

    labels_pd = pd.read_csv(dirs + "label.csv")
    bot_ids = list(set(list(labels_pd[labels_pd["label"] == "bot"]["id"])))
    human_ids = list(set(list(labels_pd[labels_pd["label"] == "human"]["id"])))
    proc = dirs + "process/"
    with open(proc + "bot_ids.pickle", "wb") as f:
        pickle.dump(bot_ids, f)
    with open(proc + "human_ids.pickle", "wb") as f:
        pickle.dump(human_ids, f)


def run_renmin_pipeline(dirs: str) -> None:
    train_pd, user_json = load_renmin_train_and_user(dirs)
    node_id_with_label_sorted, node_label_with_label_sorted = build_sorted_train_labels(train_pd)
    rows = build_node_with_label_rows(
        user_json, node_id_with_label_sorted, node_label_with_label_sorted
    )
    node_pd = save_node_with_label_csv(dirs, rows)
    (
        node_ids_label,
        node_ids_label_sorted,
        node_id_name_sorted,
        node_name_id_sorted,
    ) = build_node_id_name_maps(node_pd)

    save_attr_emb_npy(dirs, node_pd)
    save_node_labels_npy_and_bot_human_pickles_renmin(dirs, train_pd, node_id_name_sorted)
    build_interaction_with_label_csv_renmin(
        dirs, user_json, node_ids_label, node_ids_label_sorted, node_name_id_sorted
    )


def is_renmin_layout(dirs: str) -> bool:
    return os.path.isfile(dirs + "train.csv") and os.path.isfile(dirs + "user.json")


def run_twibot_pipeline(dirs: str) -> None:
   
    build_interaction_with_label_csv_twibot(dirs)
    if os.path.isfile(dirs + "label.csv"):
        save_bot_human_pickles_from_label_csv(dirs)


def main():
    p = argparse.ArgumentParser(
        description="Generate attr_emb / node_labels / pickles / interaction CSV in one run (from notebooks)."
    )
    p.add_argument(
        "--dirs",
        type=str,
        default="./dataset/Renmin/",
        help="Dataset root: Renmin (train.csv, user.json) or Twibot JSON splits.",
    )
    args = p.parse_args()
    dirs = args.dirs
    if not dirs.endswith("/"):
        dirs = dirs + "/"

    if is_renmin_layout(dirs):
        run_renmin_pipeline(dirs)
    else:
        run_twibot_pipeline(dirs)


if __name__ == "__main__":
    main()
