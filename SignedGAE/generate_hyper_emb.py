import argparse
import os
import pickle
import sys
from ast import literal_eval

import numpy as np
import pandas as pd
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_HYPER_INTER_WALK = os.path.join(_SCRIPT_DIR, "HyperInterWalk")
for _p in (_HYPER_INTER_WALK, _SCRIPT_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import find_index

from .HyperGraph import HyperGraph
from .pathGenerator import pathGenerator
from subgraphEncoder import SubgraphEncoder


def load_pickle(path: str):
    print("load " + os.path.basename(path))
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_dirs(dirs: str) -> str:
    return dirs if dirs.endswith(os.sep) or dirs.endswith("/") else dirs + "/"


def build_node_name_maps(node_pd: pd.DataFrame):
   
    node_ids_label = list(node_pd["id"])
    node_names_label = list(node_pd["username"])
    node_label_dict = {k: v for k, v in zip(node_names_label, node_ids_label)}
    node_name_id_sorted = dict(sorted(node_label_dict.items(), key=lambda v: v[0]))
    return node_ids_label, node_name_id_sorted


def save_hyperedge_matrix_npy(
    hyper_mention_id: dict,
    node_ids_label: list,
    node_name_id_sorted: dict,
    out_npy: str,
) -> None:
    
    hyperedge_matrix_passive = np.zeros((len(node_name_id_sorted), len(hyper_mention_id)))
    _hyperedge_matrix_positive = np.zeros((len(node_name_id_sorted), len(hyper_mention_id)))

    node_ids_label_sorted = sorted(node_ids_label)
    for i, k in tqdm(
        enumerate(list(hyper_mention_id.keys())),
        desc="hyperedge matrix",
        total=len(hyper_mention_id),
    ):
        for t in k:
            hyperedge_matrix_passive[
                find_index(t, node_ids_label_sorted), i
            ] = hyper_mention_id[k]

    np.save(out_npy, hyperedge_matrix_passive)
    print("saved", out_npy, hyperedge_matrix_passive.shape)


def build_hyperedge_dict_pickle(
    dirs: str,
    interaction_csv: str,
    node_with_label_csv: str,
    out_pickle: str,
    out_hyperedge_npy: str | None = None,
) -> None:

    node_pd = pd.read_csv(node_with_label_csv)
    node_ids_label, node_name_id_sorted = build_node_name_maps(node_pd)

    mention_pd = pd.read_csv(interaction_csv)
    mention_pd = mention_pd.loc[mention_pd["relation"] == "mention"]

    hyper_mention_id: dict = {}

    if "target_name" in mention_pd.columns:
        for _i, row in mention_pd.iterrows():
            target_names = literal_eval(str(row["target_name"]))
            source_id = row["id"]
            target_names_label = []
            for t in target_names:
                if t in node_name_id_sorted:
                    target_names_label.append(node_name_id_sorted[t])
            if len(target_names_label) > 0:
                hyperedge = tuple([source_id] + [t for t in target_names_label])
                if hyperedge in hyper_mention_id:
                    hyper_mention_id[hyperedge] += 1
                else:
                    hyper_mention_id[hyperedge] = 1
    elif "target_id" in mention_pd.columns:
        for _i, row in mention_pd.iterrows():
            target_ids = row["target_id"]
            if isinstance(target_ids, str):
                target_ids = literal_eval(str(target_ids))
            source_id = row["id"]
            if len(target_ids) > 0:
                hyperedge = tuple([source_id] + sorted(target_ids))
                if hyperedge in hyper_mention_id:
                    hyper_mention_id[hyperedge] += 1
                else:
                    hyper_mention_id[hyperedge] = 1
    else:
        raise ValueError(
            "interaction CSV needs column 'target_name' (Twibot) or 'target_id' (Renmin)."
        )

    node_ids_label_sorted = sorted(node_ids_label)
    hyper_mention_index = {}
    for k in tqdm(list(hyper_mention_id.keys()), desc="hyperedge index"):
        k_index = tuple([find_index(ki, node_ids_label_sorted) for ki in k])
        hyper_mention_index[k_index] = hyper_mention_id[k]

    with open(out_pickle, "wb") as f:
        pickle.dump(hyper_mention_index, f)
    print("saved", out_pickle, "n_edges", len(hyper_mention_index))

    hyperedge_npy_path = out_hyperedge_npy or (normalize_dirs(dirs) + "hyperedge.npy")
    save_hyperedge_matrix_npy(hyper_mention_id, node_ids_label, node_name_id_sorted, hyperedge_npy_path)


def main():
    p = argparse.ArgumentParser(description="Generate hyperwalk_10.npy (test_subgraphEncoder.ipynb).")
    p.add_argument(
        "--dirs",
        type=str,
        required=True,
        help="Dataset root (e.g. Twibot-20) containing process/ and hyperedge_dict.pickle.",
    )
    p.add_argument(
        "--retweet-pickle",
        type=str,
        default=None,
        help="Override path to retweet_matrix_label.pickle",
    )
    p.add_argument(
        "--hyperedge-pickle",
        type=str,
        default=None,
        help="Override path to hyperedge_dict.pickle",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .npy path (default: <repo>/TestClassification/Model/hyperwalk_10.npy)",
    )
    p.add_argument(
        "--d",
        type=int,
        default=10,
        help="Word2Vec vector size (notebook uses 10 → hyperwalk_10.npy).",
    )
    p.add_argument(
        "--interaction-csv",
        type=str,
        default=None,
        help="interaction_with_label.csv (for building hyperedge_dict).",
    )
    p.add_argument(
        "--node-with-label-csv",
        type=str,
        default=None,
        help="node_with_label.csv (for building hyperedge_dict).",
    )
    p.add_argument(
        "--rebuild-hyperedge",
        action="store_true",
        help="Regenerate hyperedge_dict.pickle even if it exists.",
    )
    p.add_argument(
        "--only-hyperedge",
        action="store_true",
        help="Only build hyperedge_dict.pickle; skip hyperwalk embedding.",
    )
    p.add_argument(
        "--hyperedge-npy",
        type=str,
        default=None,
        help="Output path for hyperedge.npy (default: <dirs>/hyperedge.npy).",
    )
    args = p.parse_args()

    dirs = normalize_dirs(args.dirs)
    retweet_path = args.retweet_pickle or (dirs + "process/retweet_matrix_label.pickle")
    hyperedge_path = args.hyperedge_pickle or (dirs + "hyperedge_dict.pickle")
    out_path = args.out or os.path.join(_REPO_ROOT, "hyperwalk_10.npy")
    interaction_csv = args.interaction_csv or (dirs + "interaction_with_label.csv")
    node_csv = args.node_with_label_csv or (dirs + "node_with_label.csv")

    if args.rebuild_hyperedge or not os.path.isfile(hyperedge_path):
        if not os.path.isfile(interaction_csv):
            raise FileNotFoundError(
                f"Need {interaction_csv} to build hyperedge_dict.pickle (or place existing pickle)."
            )
        if not os.path.isfile(node_csv):
            raise FileNotFoundError(
                f"Need {node_csv} to build hyperedge_dict.pickle (or place existing pickle)."
            )
        build_hyperedge_dict_pickle(
            dirs,
            interaction_csv,
            node_csv,
            hyperedge_path,
            out_hyperedge_npy=args.hyperedge_npy,
        )

    if args.only_hyperedge:
        return

    retweet_matrix = load_pickle(retweet_path)
    with open(hyperedge_path, "rb") as f:
        hyperedge_passive = pickle.load(f)

    mention_graph_ = HyperGraph(list(range(len(retweet_matrix))), hyperedge_passive)
    pg_ = pathGenerator(mention_graph_, retweet_matrix)

    fs = []
    for ego in tqdm(range(len(retweet_matrix))):
        walks = pg_.getPath(ego)
        sg = SubgraphEncoder(walks, d=args.d)
        f = sg.learnFeature()
        fs.append(list(f[ego]))

    fs_arr = np.array(fs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, fs_arr)
    print("saved", out_path, fs_arr.shape)


if __name__ == "__main__":
    main()
