import argparse
import importlib
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import clear_data, tweet_cate_encode

def _normalize_dirs(dirs: str) -> str:
    return dirs if dirs.endswith("/") else dirs + "/"


def load_user_json_renmin(dirs: str) -> list:
    with open(dirs + "user.json", "r", encoding="utf8") as fp:
        return json.load(fp)


def load_user_json_twibot(dirs: str) -> list:
    with open(dirs + "test.json", "r", encoding="utf8") as fp:
        test_json = json.load(fp)
    with open(dirs + "train.json", "r", encoding="utf8") as fp:
        train_json = json.load(fp)
    with open(dirs + "support.json", "r", encoding="utf8") as fp:
        support_json = json.load(fp)
    with open(dirs + "dev.json", "r", encoding="utf8") as fp:
        dev_json = json.load(fp)
    return test_json + train_json + support_json + dev_json


def load_node_ids_label(dirs: str) -> list:
    node_pd = pd.read_csv(dirs + "node_with_label.csv")
    return list(node_pd["id"])


def build_tweet_emb_dict(
    user_json: list,
    node_ids_label: list,
    model,
    user_start: int = 0,
) -> dict:

    tweet_emb_dict: dict = {}
    for u in tqdm(user_json[user_start:]):
        user_id = "u" + u["ID"]
        if u["profile"] is not None and user_id in node_ids_label:
            tweet_list = u["tweet"]
            if tweet_list is not None:
                tweet_list = [clear_data(t) for t in tweet_list]
                tweet_emb_dict[user_id] = model.encode(tweet_list)
            else:
                tweet_emb_dict[user_id] = []
    return tweet_emb_dict


def build_tweet_cate_dict(user_json: list, node_ids_label: list) -> dict:

    tweet_cate_dict: dict = {}
    for u in tqdm(user_json):
        user_id = "u" + u["ID"]
        if u["profile"] is not None and user_id in node_ids_label:
            tweet_list = u["tweet"]
            if tweet_list is not None:
                tweet_list = [tweet_cate_encode(t) for t in tweet_list]
                tweet_cate_dict[user_id] = tweet_list
            else:
                tweet_cate_dict[user_id] = []
    return tweet_cate_dict


def save_tweet_emb_and_cate_pickles(dirs: str, tweet_emb_dict: dict, tweet_cate_dict: dict) -> None:

    keys = sorted(tweet_cate_dict.keys())
    with open(dirs + "cate_encode.pickle", "wb") as f:
        pickle.dump([tweet_cate_dict[k] for k in keys], f)
    with open(dirs + "tweet_emb.pickle", "wb") as f:
        pickle.dump([tweet_emb_dict.get(k, []) for k in keys], f)


def load_merged_lstm_cfg(lstm_config_path: str, dirs: str, dataset: str) -> dict:
    with open(lstm_config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    d = _normalize_dirs(dirs)
    if dataset == "Twibot-20":
        cfg["Twibot20_data_path"] = d
    elif dataset == "Renmin":
        cfg["Renmin_data_path"] = d
    return cfg


def inject_read_config(lstm_root: str, merged_cfg: dict):
    if lstm_root not in sys.path:
        sys.path.insert(0, lstm_root)
    import ReadConfig as rc

    class ReadConfig:
        def read_config(self):
            return merged_cfg

    rc.ReadConfig = ReadConfig


def load_lstm_ae_classes(lstm_root: str, merged_cfg: dict):
   
    inject_read_config(lstm_root, merged_cfg)
    importlib.invalidate_caches()
    ae_mod = importlib.import_module("AutoEncoder")
    dl_mod = importlib.import_module("DataLoader")
    return ae_mod.AutoEncoder, dl_mod.AutoEncoderDataLoader


class LSTMTrainer:

    def __init__(self, lstm_root: str, merged_cfg: dict, dataset: str, device: str) -> None:
        AutoEncoderCls, DataLoaderCls = load_lstm_ae_classes(lstm_root, merged_cfg)
        self.AutoEncoder = AutoEncoderCls
        self.model = self.AutoEncoder()
        self.cfg = merged_cfg
        self.epoch = self.cfg["epoch"]
        self.lr = self.cfg["lr"]
        self.dataset = dataset
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.device = device
        self.bce_loss = nn.BCELoss()
        self.mls_loss = torch.nn.MultiLabelSoftMarginLoss()
        self.AutoEncoderDataLoader = DataLoaderCls
        train_dataset = self.AutoEncoderDataLoader(dataset=self.dataset)
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.cfg["batch_size"], shuffle=False
        )

    def loss_fn(self, decoded_seq, seqs_origin, y_cate, y_content, label):
        AE = F.mse_loss(decoded_seq, seqs_origin)
        CATE = self.bce_loss(y_cate, label[:, -6:-4])
        CONTENT = self.mls_loss(y_content, label[:, -4:])
        return AE + CATE + CONTENT

    def accuracy(self, y_cate, y_content, label):
        _, predicted = torch.max(y_cate, dim=1)
        _, label_cate = torch.max(label[:, -6:-4], dim=1)
        crt_cate = (predicted == label_cate).sum().item()
        y_content = torch.round(y_content)
        label_cont = label[:, -4:]
        crt_cont = (y_content == label_cont).all(dim=1).sum().item()
        total = label.size(0)
        return crt_cate / total, crt_cont / total

    def train(self):
        self.model.train()
        self.model = self.model.to(self.device)
        n_batches = len(self.train_dataloader)
        for epoch in tqdm(range(self.epoch)):
            loss_0 = 0.0
            for _, item in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                seqs, label = item
                seqs, label = seqs.to(self.device), label.to(self.device)
                seqs, label = seqs.float(), label.float()
                _enc, decoded_seq, _vec, y_cate, y_content, _att = self.model(seqs)
                loss = self.loss_fn(decoded_seq, seqs, y_cate, y_content, label)
                loss_0 += loss.item()
                loss.backward()
                self.optimizer.step()
                self.accuracy(y_cate, y_content, label)
            print("loss:", loss_0 / max(n_batches, 1))
        save_name = self.cfg["save_path"] + "/AE-content+cate-" + self.dataset + ".pth"
        os.makedirs(self.cfg["save_path"], exist_ok=True)
        torch.save(self.model.state_dict(), save_name)
        print("-------finish training--------", save_name)
        return save_name

    def export_history_emb_attention(self, ckpt_path: str, out_npy: str) -> None:
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        all_ctx = []
        with torch.no_grad():
            for item in self.train_dataloader:
                seqs, _label = item
                seqs = seqs.float().to(self.device)
                _enc, _dec, context_vector, _yc, _yt, _aw = self.model(seqs)
                all_ctx.append(context_vector.cpu().numpy())
        emb = np.concatenate(all_ctx, axis=0)
        np.save(out_npy, emb)
        print("saved", out_npy, emb.shape)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", type=str, required=True, help="Dataset root (see node_with_label.csv, etc.)")
    p.add_argument("--dataset", type=str, choices=["Renmin", "Twibot-20"], default="Renmin")
    p.add_argument(
        "--lstm-root",
        type=str,
        default=os.path.join(root, "BotPool", "LSTM-AE"),
        help="Directory containing AutoEncoder.py, DataLoader.py, config.yaml",
    )
    p.add_argument(
        "--lstm-config",
        type=str,
        default=None,
        help="Override path to LSTM-AE yaml (default: <lstm-root>/config.yaml)",
    )
    p.add_argument(
        "--sentence-model",
        type=str,
        default="sentence-transformers/stsb-xlm-r-multilingual",
        help="SentenceTransformer id or local path (notebook used a hub cache path)",
    )
    p.add_argument("--user-start", type=int, default=0, help="Resume index into user_json (notebook used 15919).")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device (default: cuda if available else cpu)",
    )
    p.add_argument("--skip-sentence", action="store_true", help="Skip SentenceTransformer + pickle step.")
    p.add_argument("--skip-train", action="store_true", help="Skip LSTM training (still export if --ckpt set).")
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
    )
    p.add_argument("--skip-export", action="store_true", help="Do not write history_emb_attention.npy")
    args = p.parse_args()

    dirs = _normalize_dirs(args.dirs)
    lstm_config = args.lstm_config or os.path.join(args.lstm_root, "config.yaml")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not args.skip_sentence:
        from sentence_transformers import SentenceTransformer

        node_ids_label = load_node_ids_label(dirs)
        if args.dataset == "Renmin":
            user_json = load_user_json_renmin(dirs)
        else:
            user_json = load_user_json_twibot(dirs)

        model_st = SentenceTransformer(args.sentence_model)
        tweet_emb_dict = build_tweet_emb_dict(
            user_json, node_ids_label, model_st, user_start=args.user_start
        )
        tweet_cate_dict = build_tweet_cate_dict(user_json, node_ids_label)
        save_tweet_emb_and_cate_pickles(dirs, tweet_emb_dict, tweet_cate_dict)

    merged_cfg = load_merged_lstm_cfg(lstm_config, dirs, args.dataset)
    ckpt = args.ckpt

    if not args.skip_train:
        trainer = LSTMTrainer(args.lstm_root, merged_cfg, args.dataset, device)
        ckpt = trainer.train()
    elif ckpt is None:
        save_path = merged_cfg["save_path"]
        ckpt = save_path + "/AE-content+cate-" + args.dataset + ".pth"

    if not args.skip_export:
        export_trainer = LSTMTrainer(args.lstm_root, merged_cfg, args.dataset, device)
        export_trainer.export_history_emb_attention(ckpt, dirs + "history_emb_attention.npy")


if __name__ == "__main__":
    main()
