# Social Bot Group Detection

Research codebase for **social bot group detection** on social graphs: BotPool and SignedGAE.

---

## Dataset

- **Twibot-20** (primary public benchmark): [Twibot-20 on GitHub](https://github.com/bunsenfeng/twibot-20)  

Place datasets under paths you configure in **`BotPool/config.yaml`**, and  **`SignedGAE/config.yaml`**.

---

## Repository layout (high level)

| Path | Role |
|------|------|
| **`base_data_prepare.py`** | One-shot base tables: `node_with_label.csv`, `attr_emb.npy`, `interaction_with_label.csv`, `node_labels.npy`, `bot_ids.pickle`, `human_ids.pickle` (Renmin vs Twibot auto-detected by files in `--dirs`). |
| **`generate_tweet_emb.py`** | Tweet/category sequences → `tweet_emb.pickle`, `cate_encode.pickle`; LSTM-AE training → `history_emb_attention.npy` (see `BotPool/LSTM-AE/`). |
| **`build_signed_network.py`** | Signed stance graph → `pos_adj_top{5,10,20}.npy`, `neg_adj_top{5,10,20}.npy` (from `others/signed_network.py` logic). |
| **`SignedGAE/generate_hyper_emb.py`** | Hypergraph random-walk embeddings: `hyperedge_dict.pickle`, `hyperedge.npy`, `hyperwalk_10.npy` (and uses `SignedGAE/HyperInterWalk/*`). |
| **`BotPool/`** | Motif mining, DGL-style data build, training (`main.py`), evaluation (`Evaluation.py`). |
| **`SignedGAE/`** | Signed GAE: `train.py`, `DataLoader.py`, conv layers, contrastive helpers, `HyperInterWalk/` for hypergraph walks. |
| **`utils.py`** | Shared text/graph helpers (`find_index`, `mention_list`, `tweet_cate_encode`, etc.). |
---

## Environment

- **Python 3.8+** is a reasonable target (project historically used 3.8).  
- Typical stacks: **PyTorch**, **PyG/DGL** (BotPool), **pandas**, **numpy**, **networkx**, **sentence-transformers** (tweet encoding), **gensim** (hyper walk embeddings), **transformers** (sentiment for signed network), **yaml**.


---

## Data preparation (quick reference)

### 1. Base tables — `base_data_prepare.py`

```bash
python base_data_prepare.py --dirs /path/to/dataset/Twibot-20/
```
  
**Twibot**: writes interaction CSV and (if `label.csv` exists) bot/human pickles.

### 2. Tweet / history embeddings — `generate_tweet_emb.py`

```bash
python generate_tweet_emb.py --dirs /path/to/Twibot-20/ --dataset Twibot-20 \
  --lstm-root ./BotPool/LSTM-AE --sentence-model sentence-transformers/stsb-xlm-r-multilingual
```

Produces **`tweet_emb.pickle`**, **`cate_encode.pickle`**, **`history_emb_attention.npy`** (and trains LSTM-AE per `BotPool/LSTM-AE/config.yaml`).

### 3. Signed stance network — `build_signed_network.py`

```bash
python build_signed_network.py --dirs /path/to/Twibot-20/ [TweetNLP Model]
```

### 4. Hypergraph walk embedding — `SignedGAE/generate_hyper_emb.py`

```bash
python SignedGAE/generate_hyper_emb.py --dirs /path/to/Twibot-20/
```

When **`hyperedge_dict.pickle`** is missing, builds it (and **`hyperedge.npy`**) from **`interaction_with_label.csv`** + **`node_with_label.csv`**, then runs **`hyperwalk_10.npy`**.

---

## BotPool (coordination motifs)

- **`BotPool/main.py`** — training entry (check `ReadConfig` / `config.yaml` for dataset and paths).  
- **`BotPool/Evaluation.py`** — evaluation utilities.

Ensure **`retweet_matrix_label.pickle`**, **`mention_matrix_label.pickle`**, and related `.npy` / `.pickle` files exist under the dataset folder configured in **`BotPool/config.yaml`**.

---

## SignedGAE

- **`SignedGAE/train.py`** — train the signed graph autoencoder.  

Point **`SignedGAE/ReadConfig.py`** at your **`SignedGAE/config.yaml`** and set data paths there.

---

## Citation

**Twibot-20**: cite the dataset paper from the [official repository](https://github.com/bunsenfeng/twibot-20). 
**SGCN** cite the dataset paper from the [official repository](https://github.com/benedekrozemberczki/SGCN). 

