from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_lines(path: Path | str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    lines = [line.strip() for line in p.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def tokenize_whitespace(text: str) -> List[str]:
    return [tok for tok in text.split() if tok]


def read_corpus(path: Path | str) -> List[List[str]]:
    return [tokenize_whitespace(line) for line in read_lines(path)]


def read_metadata(path: Path | str) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _label_from_item(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ["topic", "category", "label", "class", "tag"]:
            if key in item and item[key] is not None:
                return str(item[key])
    return None


def extract_topic_labels(metadata: Any, n_docs: int) -> List[str]:
    labels: List[str] = []

    if isinstance(metadata, list):
        labels = [_label_from_item(item) or "unknown" for item in metadata]
    elif isinstance(metadata, dict):
        for candidate in ["labels", "topics", "categories", "data", "articles", "items"]:
            if candidate in metadata and isinstance(metadata[candidate], list):
                labels = [_label_from_item(item) or "unknown" for item in metadata[candidate]]
                break
        if not labels:
            # Fall back to dictionary values if they look list-like.
            for value in metadata.values():
                if isinstance(value, list) and value:
                    labels = [_label_from_item(item) or "unknown" for item in value]
                    if labels:
                        break

    if not labels:
        labels = ["unknown"] * n_docs

    if len(labels) < n_docs:
        labels = labels + [labels[-1] if labels else "unknown"] * (n_docs - len(labels))
    return labels[:n_docs]


def build_vocab(
    tokenized_docs: Sequence[Sequence[str]],
    max_vocab_size: int = 10000,
    min_freq: int = 1,
    include_cls: bool = True,
) -> Tuple[Dict[str, int], List[str], Counter]:
    freq = Counter(tok for doc in tokenized_docs for tok in doc)
    sorted_tokens = [tok for tok, c in freq.most_common() if c >= min_freq]

    specials = [PAD_TOKEN, UNK_TOKEN]
    if include_cls:
        specials.append(CLS_TOKEN)

    cap = max(0, max_vocab_size - len(specials))
    kept = sorted_tokens[:cap]
    idx2word = specials + kept
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return word2idx, idx2word, freq


def encode_docs(tokenized_docs: Sequence[Sequence[str]], word2idx: Dict[str, int]) -> List[List[int]]:
    unk_id = word2idx[UNK_TOKEN]
    return [[word2idx.get(tok, unk_id) for tok in doc] for doc in tokenized_docs]


def save_json(path: Path | str, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def stratified_split_indices(
    labels: Sequence[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("Split ratios must sum to 1")

    rng = random.Random(seed)
    label_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        label_to_indices[label].append(i)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for indices in label_to_indices.values():
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val

        # Keep all sets non-negative while preserving total.
        if n_test < 0:
            n_test = 0
            n_val = n - n_train
        if n_val < 0:
            n_val = 0
            n_train = n

        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train : n_train + n_val])
        test_idx.extend(indices[n_train + n_val : n_train + n_val + n_test])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def cosine_similarity_matrix(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    y = x if y is None else y
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    y_norm = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
    return x_norm @ y_norm.T


def top_k_cosine_neighbors(
    vectors: np.ndarray,
    idx2word: Sequence[str],
    query_words: Sequence[str],
    word2idx: Dict[str, int],
    k: int = 10,
) -> Dict[str, List[Tuple[str, float]]]:
    sims = cosine_similarity_matrix(vectors)
    out: Dict[str, List[Tuple[str, float]]] = {}
    for qw in query_words:
        if qw not in word2idx:
            out[qw] = []
            continue
        qi = word2idx[qw]
        order = np.argsort(-sims[qi])
        neighbors: List[Tuple[str, float]] = []
        for j in order:
            if j == qi:
                continue
            neighbors.append((idx2word[j], float(sims[qi, j])))
            if len(neighbors) >= k:
                break
        out[qw] = neighbors
    return out


def compute_mrr(candidates: Sequence[Sequence[str]], gold_targets: Sequence[str]) -> float:
    if len(candidates) != len(gold_targets):
        raise ValueError("candidates and gold_targets lengths differ")
    reciprocal_ranks = []
    for cand_list, gold in zip(candidates, gold_targets):
        rank = 0
        for i, cand in enumerate(cand_list, start=1):
            if cand == gold:
                rank = i
                break
        reciprocal_ranks.append(1.0 / rank if rank > 0 else 0.0)
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
