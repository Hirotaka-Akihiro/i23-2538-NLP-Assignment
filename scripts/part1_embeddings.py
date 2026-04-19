from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import (
    PAD_TOKEN,
    UNK_TOKEN,
    build_vocab,
    compute_mrr,
    cosine_similarity_matrix,
    ensure_dir,
    encode_docs,
    extract_topic_labels,
    read_corpus,
    read_metadata,
    save_json,
    set_seed,
    top_k_cosine_neighbors,
)


def build_tfidf(
    encoded_docs: Sequence[Sequence[int]], vocab_size: int, pad_id: int
) -> Tuple[np.ndarray, np.ndarray]:
    n_docs = len(encoded_docs)
    tf = np.zeros((n_docs, vocab_size), dtype=np.float32)
    df = np.zeros(vocab_size, dtype=np.float32)

    for i, doc in enumerate(encoded_docs):
        counts = Counter(doc)
        for wid, c in counts.items():
            if wid == pad_id:
                continue
            tf[i, wid] = float(c)
        for wid in counts.keys():
            if wid == pad_id:
                continue
            df[wid] += 1.0

    idf = np.zeros(vocab_size, dtype=np.float32)
    non_pad = np.arange(vocab_size) != pad_id
    idf[non_pad] = np.log(float(n_docs) / (1.0 + df[non_pad]))
    tfidf = tf * idf.reshape(1, -1)
    return tfidf, idf


def top_discriminative_words_per_topic(
    tfidf: np.ndarray,
    labels: Sequence[str],
    idx2word: Sequence[str],
    top_k: int = 10,
) -> Dict[str, List[Tuple[str, float]]]:
    topic_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, topic in enumerate(labels):
        topic_to_indices[topic].append(i)

    out: Dict[str, List[Tuple[str, float]]] = {}
    for topic, indices in topic_to_indices.items():
        if not indices:
            out[topic] = []
            continue
        mean_scores = tfidf[indices].mean(axis=0)
        order = np.argsort(-mean_scores)
        best: List[Tuple[str, float]] = []
        for wid in order:
            word = idx2word[wid]
            if word in {PAD_TOKEN, UNK_TOKEN, "<CLS>"}:
                continue
            best.append((word, float(mean_scores[wid])))
            if len(best) >= top_k:
                break
        out[topic] = best
    return out


def build_cooccurrence(encoded_docs: Sequence[Sequence[int]], vocab_size: int, window: int = 5) -> np.ndarray:
    cooc = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    for doc in encoded_docs:
        n = len(doc)
        for i, center in enumerate(doc):
            left = max(0, i - window)
            right = min(n, i + window + 1)
            for j in range(left, right):
                if i == j:
                    continue
                context = doc[j]
                cooc[center, context] += 1.0
    return cooc


def to_ppmi(cooc: np.ndarray) -> np.ndarray:
    total = float(cooc.sum())
    if total <= 0:
        return np.zeros_like(cooc)

    row = cooc.sum(axis=1, keepdims=True)
    col = cooc.sum(axis=0, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        numer = cooc * total
        denom = row * col + 1e-12
        pmi = np.log2((numer / denom) + 1e-12)
    pmi[~np.isfinite(pmi)] = 0.0
    pmi[pmi < 0] = 0.0
    pmi[cooc <= 0] = 0.0
    return pmi.astype(np.float32)


def semantic_bucket(token: str) -> str:
    politics = {"hukumat", "wazir", "election", "parliament", "adalat", "fauj", "pakistan"}
    sports = {"cricket", "match", "team", "player", "score", "goal"}
    economy = {"maeeshat", "bank", "budget", "inflation", "trade", "gdp"}
    geography = {"lahore", "karachi", "islamabad", "punjab", "sindh", "balochistan"}
    health_society = {"sehat", "hospital", "vaccine", "taleem", "aabadi", "disease"}

    lw = token.lower()
    if lw in politics:
        return "politics"
    if lw in sports:
        return "sports"
    if lw in economy:
        return "economy"
    if lw in geography:
        return "geography"
    if lw in health_society:
        return "health_society"
    return "other"


def plot_tsne(ppmi: np.ndarray, idx2word: Sequence[str], freq: Counter, out_path: Path) -> None:
    candidates = [w for w in idx2word if w not in {PAD_TOKEN, UNK_TOKEN, "<CLS>"}]
    top_words = sorted(candidates, key=lambda w: freq.get(w, 0), reverse=True)[:200]
    if len(top_words) < 5:
        return

    indices = [idx2word.index(w) for w in top_words]
    vecs = ppmi[indices]

    perplexity = max(5, min(30, len(indices) - 1))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca", learning_rate="auto")
    points = tsne.fit_transform(vecs)

    categories = [semantic_bucket(w) for w in top_words]
    unique = sorted(set(categories))
    color_map = {cat: plt.cm.tab10(i % 10) for i, cat in enumerate(unique)}

    plt.figure(figsize=(12, 9))
    for cat in unique:
        idxs = [i for i, c in enumerate(categories) if c == cat]
        xy = points[idxs]
        plt.scatter(xy[:, 0], xy[:, 1], s=18, alpha=0.8, color=color_map[cat], label=cat)

    for i, token in enumerate(top_words[:40]):
        plt.annotate(token, (points[i, 0], points[i, 1]), fontsize=8, alpha=0.7)

    plt.title("Part 1: t-SNE of Top-200 Tokens from PPMI Vectors")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


class SkipGramPairsDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_docs: Sequence[Sequence[int]], window: int = 5):
        self.pairs: List[Tuple[int, int]] = []
        for doc in encoded_docs:
            n = len(doc)
            for i, center in enumerate(doc):
                left = max(0, i - window)
                right = min(n, i + window + 1)
                for j in range(left, right):
                    if i == j:
                        continue
                    self.pairs.append((center, doc[j]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[int, int]:
        return self.pairs[idx]


class NegativeSampler:
    def __init__(self, token_freq: Counter, word2idx: Dict[str, int], power: float = 0.75):
        size = len(word2idx)
        probs = np.zeros(size, dtype=np.float64)
        for w, i in word2idx.items():
            probs[i] = float(token_freq.get(w, 0)) ** power

        probs[0] = 0.0  # <PAD>
        total = probs.sum()
        if total <= 0:
            probs += 1.0
            probs[0] = 0.0
            total = probs.sum()
        self.probs = probs / total
        self.ids = np.arange(size)

    def sample(self, batch_size: int, k: int) -> torch.Tensor:
        draws = np.random.choice(self.ids, size=(batch_size, k), p=self.probs)
        return torch.from_numpy(draws).long()


class SkipGramWord2Vec(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, dim)
        self.context_embeddings = nn.Embedding(vocab_size, dim)

        nn.init.uniform_(self.center_embeddings.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.context_embeddings.weight)

    def forward(self, center_ids: torch.Tensor, pos_context_ids: torch.Tensor, neg_context_ids: torch.Tensor) -> torch.Tensor:
        v = self.center_embeddings(center_ids)  # [B, D]
        u_pos = self.context_embeddings(pos_context_ids)  # [B, D]
        u_neg = self.context_embeddings(neg_context_ids)  # [B, K, D]

        pos_logits = torch.sum(v * u_pos, dim=1)  # [B]
        neg_logits = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)  # [B, K]

        loss = -(F.logsigmoid(pos_logits) + torch.sum(F.logsigmoid(-neg_logits), dim=1)).mean()
        return loss

    def export_embeddings(self) -> np.ndarray:
        v = self.center_embeddings.weight.detach().cpu().numpy()
        u = self.context_embeddings.weight.detach().cpu().numpy()
        return 0.5 * (v + u)


def train_skipgram(
    encoded_docs: Sequence[Sequence[int]],
    token_freq: Counter,
    word2idx: Dict[str, int],
    dim: int = 100,
    window: int = 5,
    negative_k: int = 10,
    epochs: int = 5,
    batch_size: int = 1024,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Tuple[np.ndarray, List[float]]:
    dataset = SkipGramPairsDataset(encoded_docs, window=window)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model = SkipGramWord2Vec(vocab_size=len(word2idx), dim=dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sampler = NegativeSampler(token_freq=token_freq, word2idx=word2idx)

    history: List[float] = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for centers, contexts in loader:
            centers = centers.to(device)
            contexts = contexts.to(device)
            negatives = sampler.sample(batch_size=centers.shape[0], k=negative_k).to(device)

            optimizer.zero_grad()
            loss = model(centers, contexts, negatives)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            batches += 1

        mean_loss = epoch_loss / max(1, batches)
        history.append(mean_loss)
        print(f"[SkipGram] epoch={epoch + 1}/{epochs} loss={mean_loss:.4f}")

    return model.export_embeddings(), history


def save_loss_curve(losses: Sequence[float], out_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(losses) + 1), losses, marker="o")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def solve_analogy(
    vectors: np.ndarray,
    word2idx: Dict[str, int],
    idx2word: Sequence[str],
    a: str,
    b: str,
    c: str,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    for w in (a, b, c):
        if w not in word2idx:
            return []

    target = vectors[word2idx[b]] - vectors[word2idx[a]] + vectors[word2idx[c]]
    norms = np.linalg.norm(vectors, axis=1) + 1e-12
    sims = (vectors @ target) / (norms * (np.linalg.norm(target) + 1e-12))
    banned = {word2idx[a], word2idx[b], word2idx[c]}

    order = np.argsort(-sims)
    result: List[Tuple[str, float]] = []
    for idx in order:
        if idx in banned:
            continue
        result.append((idx2word[idx], float(sims[idx])))
        if len(result) >= top_k:
            break
    return result


def default_analogy_tests() -> List[Tuple[str, str, str]]:
    return [
        ("pakistan", "hukumat", "adalat"),
        ("sehat", "hospital", "taleem"),
        ("team", "match", "player"),
        ("bank", "maeeshat", "budget"),
        ("lahore", "punjab", "karachi"),
        ("wazir", "hukumat", "player"),
        ("vaccine", "hospital", "disease"),
        ("fauj", "hukumat", "adalat"),
        ("cricket", "sports", "economy"),
        ("inflation", "economy", "trade"),
    ]


def default_analogy_expected() -> Dict[str, set]:
    return {
        "pakistan:hukumat::adalat:?": {"verdict", "judge", "qanoon"},
        "sehat:hospital::taleem:?": {"education", "school", "university", "taleem"},
        "team:match::player:?": {"coach", "captain", "team"},
        "bank:maeeshat::budget:?": {"inflation", "trade", "tax", "budget"},
        "lahore:punjab::karachi:?": {"sindh"},
        "wazir:hukumat::player:?": {"team", "coach", "captain"},
        "vaccine:hospital::disease:?": {"malaria", "dengue", "covid", "disease"},
        "fauj:hukumat::adalat:?": {"verdict", "judge", "qanoon"},
        "cricket:sports::economy:?": {"trade", "budget", "inflation", "maeeshat"},
        "inflation:economy::trade:?": {"budget", "tax", "imports", "exports"},
    }


def run_condition_comparison(
    cleaned_docs: Sequence[Sequence[str]],
    raw_docs: Sequence[Sequence[str]],
    query_words: Sequence[str],
    output_dir: Path,
    device: str,
    epochs: int,
    manual_pairs_path: Path | None,
) -> Dict[str, object]:
    results: Dict[str, object] = {}

    # C1: PPMI baseline on cleaned corpus.
    w2i_c1, i2w_c1, freq_c1 = build_vocab(cleaned_docs, max_vocab_size=5000, include_cls=False)
    enc_c1 = encode_docs(cleaned_docs, w2i_c1)
    ppmi_c1 = to_ppmi(build_cooccurrence(enc_c1, vocab_size=len(i2w_c1), window=5))
    nn_c1 = top_k_cosine_neighbors(ppmi_c1, i2w_c1, query_words[:5], w2i_c1, k=5)
    results["C1"] = {"description": "PPMI baseline", "neighbors": nn_c1}

    # C2: Skip-gram on raw.
    w2i_c2, i2w_c2, freq_c2 = build_vocab(raw_docs, max_vocab_size=10000, include_cls=False)
    enc_c2 = encode_docs(raw_docs, w2i_c2)
    emb_c2, _ = train_skipgram(
        enc_c2,
        freq_c2,
        w2i_c2,
        dim=100,
        window=5,
        negative_k=10,
        epochs=epochs,
        batch_size=1024,
        lr=1e-3,
        device=device,
    )
    nn_c2 = top_k_cosine_neighbors(emb_c2, i2w_c2, query_words[:5], w2i_c2, k=5)
    results["C2"] = {"description": "Skip-gram on raw", "neighbors": nn_c2}

    # C3: Skip-gram on cleaned.
    w2i_c3, i2w_c3, freq_c3 = build_vocab(cleaned_docs, max_vocab_size=10000, include_cls=False)
    enc_c3 = encode_docs(cleaned_docs, w2i_c3)
    emb_c3, _ = train_skipgram(
        enc_c3,
        freq_c3,
        w2i_c3,
        dim=100,
        window=5,
        negative_k=10,
        epochs=epochs,
        batch_size=1024,
        lr=1e-3,
        device=device,
    )
    nn_c3 = top_k_cosine_neighbors(emb_c3, i2w_c3, query_words[:5], w2i_c3, k=5)
    results["C3"] = {"description": "Skip-gram on cleaned", "neighbors": nn_c3}

    # C4: Skip-gram on cleaned with d=200.
    emb_c4, _ = train_skipgram(
        enc_c3,
        freq_c3,
        w2i_c3,
        dim=200,
        window=5,
        negative_k=10,
        epochs=epochs,
        batch_size=1024,
        lr=1e-3,
        device=device,
    )
    nn_c4 = top_k_cosine_neighbors(emb_c4, i2w_c3, query_words[:5], w2i_c3, k=5)
    results["C4"] = {"description": "Skip-gram on cleaned, d=200", "neighbors": nn_c4}

    # MRR on manual pairs where available.
    mrr_scores: Dict[str, float] = {}
    pairs = []
    pair_source = "manual"
    if manual_pairs_path and manual_pairs_path.exists():
        with manual_pairs_path.open("r", encoding="utf-8") as f:
            pairs = json.load(f)

    if not pairs:
        pair_source = "auto_from_c1_neighbors"
        for query in query_words[:5]:
            nns = nn_c1.get(query, [])
            if nns:
                pairs.append({"query": query, "gold": nns[0][0]})

    if pairs:
        for key, vectors, w2i, i2w in [
            ("C1", ppmi_c1, w2i_c1, i2w_c1),
            ("C2", emb_c2, w2i_c2, i2w_c2),
            ("C3", emb_c3, w2i_c3, i2w_c3),
            ("C4", emb_c4, w2i_c3, i2w_c3),
        ]:
            cand_lists: List[List[str]] = []
            gold: List[str] = []
            for item in pairs:
                query = item.get("query")
                target = item.get("gold")
                if query not in w2i or target not in w2i:
                    continue
                nns = top_k_cosine_neighbors(vectors, i2w, [query], w2i, k=20).get(query, [])
                cand_lists.append([w for w, _ in nns])
                gold.append(target)
            mrr_scores[key] = compute_mrr(cand_lists, gold) if gold else 0.0
    else:
        mrr_scores = {"C1": 0.0, "C2": 0.0, "C3": 0.0, "C4": 0.0}

    results["mrr"] = mrr_scores
    for cid in ["C1", "C2", "C3", "C4"]:
        if cid in results:
            results[cid]["mrr"] = float(mrr_scores.get(cid, 0.0))
    results["mrr_pair_source"] = pair_source
    save_json(output_dir / "part1_condition_comparison.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Assignment 2 - Part 1 Embeddings")
    parser.add_argument("--cleaned", type=Path, default=Path("cleaned.txt"))
    parser.add_argument("--raw", type=Path, default=Path("raw.txt"))
    parser.add_argument("--metadata", type=Path, default=Path("Metadata.json"))
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--max-vocab", type=int, default=10000)
    parser.add_argument("--ppmi-vocab", type=int, default=5000)
    parser.add_argument("--w2v-dim", type=int, default=100)
    parser.add_argument("--w2v-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-comparison", action="store_true")
    parser.add_argument("--comparison-epochs", type=int, default=2)
    parser.add_argument("--manual-pairs", type=Path, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    cleaned_docs = read_corpus(args.cleaned)
    raw_docs = read_corpus(args.raw) if args.raw.exists() else cleaned_docs
    metadata = read_metadata(args.metadata) if args.metadata.exists() else ["unknown"] * len(cleaned_docs)
    topic_labels = extract_topic_labels(metadata, len(cleaned_docs))

    embeddings_dir = ensure_dir(args.output_root / "embeddings")
    reports_dir = ensure_dir(args.output_root / "reports")

    # TF-IDF
    word2idx, idx2word, freq = build_vocab(cleaned_docs, max_vocab_size=args.max_vocab, include_cls=False)
    encoded = encode_docs(cleaned_docs, word2idx)
    tfidf, _ = build_tfidf(encoded, vocab_size=len(idx2word), pad_id=word2idx[PAD_TOKEN])
    np.save(embeddings_dir / "tfidf_matrix.npy", tfidf)

    top_disc = top_discriminative_words_per_topic(tfidf, topic_labels, idx2word, top_k=10)
    save_json(reports_dir / "part1_tfidf_top_words.json", top_disc)

    # PPMI
    word2idx_ppmi, idx2word_ppmi, freq_ppmi = build_vocab(cleaned_docs, max_vocab_size=args.ppmi_vocab, include_cls=False)
    encoded_ppmi = encode_docs(cleaned_docs, word2idx_ppmi)
    cooc = build_cooccurrence(encoded_ppmi, vocab_size=len(idx2word_ppmi), window=5)
    ppmi = to_ppmi(cooc)
    np.save(embeddings_dir / "ppmi_matrix.npy", ppmi)

    plot_tsne(ppmi, idx2word_ppmi, freq_ppmi, reports_dir / "part1_ppmi_tsne.png")

    query_words = ["Pakistan", "Hukumat", "Adalat", "Maeeshat", "Fauj", "Sehat", "Taleem", "Aabadi"]
    query_words_lower = [q.lower() for q in query_words]

    ppmi_neighbors = top_k_cosine_neighbors(ppmi, idx2word_ppmi, query_words_lower[:10], word2idx_ppmi, k=5)
    save_json(reports_dir / "part1_ppmi_neighbors.json", ppmi_neighbors)

    # Skip-gram
    embeddings, losses = train_skipgram(
        encoded_docs=encoded,
        token_freq=freq,
        word2idx=word2idx,
        dim=args.w2v_dim,
        window=5,
        negative_k=10,
        epochs=max(5, args.w2v_epochs),
        batch_size=max(512, args.batch_size),
        lr=1e-3,
        device=device,
    )
    np.save(embeddings_dir / "embeddings_w2v.npy", embeddings)
    save_json(embeddings_dir / "word2idx.json", word2idx)
    save_loss_curve(losses, reports_dir / "part1_skipgram_loss.png", "Skip-gram Training Loss")

    # Evaluation
    nn_results = top_k_cosine_neighbors(embeddings, idx2word, query_words_lower, word2idx, k=10)

    analogies = {}
    for a, b, c in default_analogy_tests():
        analogies[f"{a}:{b}::{c}:?"] = solve_analogy(embeddings, word2idx, idx2word, a, b, c, top_k=3)

    expected_answers = default_analogy_expected()
    analogies_with_flags = {}
    for query, preds in analogies.items():
        expected = expected_answers.get(query, set())
        analogies_with_flags[query] = [
            {
                "candidate": w,
                "score": float(s),
                "is_correct": w in expected,
            }
            for w, s in preds
        ]

    eval_payload = {
        "nearest_neighbors": nn_results,
        "analogies": analogies,
        "analogy_top3_with_correctness": analogies_with_flags,
        "analogy_expected_answers": {k: sorted(v) for k, v in expected_answers.items()},
        "semantic_assessment": (
            "The learned vectors show local semantic grouping for frequent words, but quality depends strongly on corpus size and noise. "
            "Nearest-neighbour coherence is typically stronger in cleaned-corpus training than raw-corpus training."
        ),
    }
    save_json(reports_dir / "part1_w2v_eval.json", eval_payload)

    if args.run_comparison:
        run_condition_comparison(
            cleaned_docs=cleaned_docs,
            raw_docs=raw_docs,
            query_words=query_words_lower,
            output_dir=reports_dir,
            device=device,
            epochs=args.comparison_epochs,
            manual_pairs_path=args.manual_pairs,
        )

    print("Part 1 completed.")


if __name__ == "__main__":
    main()
