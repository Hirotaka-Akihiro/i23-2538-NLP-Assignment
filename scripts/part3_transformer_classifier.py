from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import (
    CLS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    build_vocab,
    ensure_dir,
    extract_topic_labels,
    read_corpus,
    read_metadata,
    save_json,
    set_seed,
    stratified_split_indices,
)


CATEGORY_MAP = {
    0: "Politics",
    1: "Sports",
    2: "Economy",
    3: "International",
    4: "Health & Society",
}

CATEGORY_KEYWORDS = {
    0: {"election", "government", "minister", "parliament", "hukumat"},
    1: {"cricket", "match", "team", "player", "score"},
    2: {"inflation", "trade", "bank", "gdp", "budget", "maeeshat"},
    3: {"un", "treaty", "foreign", "bilateral", "conflict"},
    4: {"hospital", "disease", "vaccine", "flood", "education", "sehat", "taleem"},
}


class TopicDataset(torch.utils.data.Dataset):
    def __init__(self, sequences: Sequence[List[int]], labels: Sequence[int], max_len: int = 256, pad_id: int = 0):
        self.max_len = max_len
        self.pad_id = pad_id
        self.labels = labels

        arr = np.full((len(sequences), max_len), pad_id, dtype=np.int64)
        mask = np.zeros((len(sequences), max_len), dtype=np.bool_)
        for i, seq in enumerate(sequences):
            trunc = seq[:max_len]
            arr[i, : len(trunc)] = np.array(trunc, dtype=np.int64)
            mask[i, : len(trunc)] = True

        self.inputs = arr
        self.masks = mask

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "input_ids": torch.tensor(self.inputs[idx], dtype=torch.long),
            "mask": torch.tensor(self.masks[idx], dtype=torch.bool),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.scale = math.sqrt(d_k)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_mask: torch.Tensor | None = None):
        # q, k, v: [B, L, D]
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale
        if key_mask is not None:
            # key_mask: [B, L], True where token is valid.
            mask = key_mask.unsqueeze(1).expand(-1, scores.size(1), -1)
            scores = scores.masked_fill(~mask, -1e9)

        attn = torch.softmax(scores, dim=-1)
        out = torch.bmm(attn, v)
        return out, attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int = 128, num_heads: int = 4, d_k: int = 32, d_v: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.ModuleList([nn.Linear(d_model, d_k) for _ in range(num_heads)])
        self.k_proj = nn.ModuleList([nn.Linear(d_model, d_k) for _ in range(num_heads)])
        self.v_proj = nn.ModuleList([nn.Linear(d_model, d_v) for _ in range(num_heads)])
        self.attn = ScaledDotProductAttention(d_k=d_k)
        self.out_proj = nn.Linear(num_heads * d_v, d_model)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None = None):
        head_outs = []
        head_weights = []

        for h in range(self.num_heads):
            q = self.q_proj[h](x)
            k = self.k_proj[h](x)
            v = self.v_proj[h](x)
            out, attn_weights = self.attn(q, k, v, key_mask=key_mask)
            head_outs.append(out)
            head_weights.append(attn_weights)

        concat = torch.cat(head_outs, dim=-1)
        out = self.out_proj(concat)
        weights = torch.stack(head_weights, dim=1)  # [B, H, L, L]
        return out, weights


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int = 128, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 128, max_len: int = 300):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int = 128, num_heads: int = 4, d_k: int = 32, d_v: int = 32, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None = None):
        y, attn = self.mha(self.ln1(x), key_mask=key_mask)
        x = x + self.dropout1(y)
        x = x + self.dropout2(self.ffn(self.ln2(x)))
        return x, attn


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 5,
        d_model: int = 128,
        num_heads: int = 4,
        d_k: int = 32,
        d_v: int = 32,
        d_ff: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=260)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_k=d_k,
                    d_v=d_v,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor):
        bsz = input_ids.size(0)
        x = self.token_embedding(input_ids)

        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.ones((bsz, 1), dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)

        x = self.pos_encoding(x)
        x = self.dropout(x)

        attn_stack = []
        for layer in self.layers:
            x, attn = layer(x, key_mask=full_mask)
            attn_stack.append(attn)

        cls_repr = x[:, 0]
        logits = self.classifier(cls_repr)
        return logits, attn_stack, full_mask


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, hidden: int = 128, pad_id: int = 0, num_classes: int = 5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.lstm = nn.LSTM(d_model, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor):
        x = self.embedding(input_ids)
        h, _ = self.lstm(x)
        mask_f = mask.unsqueeze(-1).float()
        pooled = (h * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-12)
        return self.classifier(pooled)


def infer_category(text_tokens: Sequence[str], metadata_label: str | None = None) -> int:
    if metadata_label:
        ml = metadata_label.lower()
        if "polit" in ml:
            return 0
        if "sport" in ml:
            return 1
        if "econom" in ml:
            return 2
        if "intern" in ml or "foreign" in ml:
            return 3
        if "health" in ml or "society" in ml or "education" in ml:
            return 4

    token_set = set(t.lower() for t in text_tokens)
    scores = {cid: len(token_set & kws) for cid, kws in CATEGORY_KEYWORDS.items()}
    return max(scores, key=scores.get)


def build_topic_dataset(docs: Sequence[Sequence[str]], topic_labels: Sequence[str], word2idx: Dict[str, int], max_len: int = 256):
    sequences = []
    labels = []
    for tokens, tlabel in zip(docs, topic_labels):
        label = infer_category(tokens, tlabel)
        ids = [word2idx.get(tok.lower(), word2idx[UNK_TOKEN]) for tok in tokens]
        sequences.append(ids[:max_len])
        labels.append(label)
    return sequences, labels


def make_loader(dataset, batch_size: int, shuffle: bool):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch_transformer(model, loader, optimizer, scheduler, device: str, training: bool):
    criterion = nn.CrossEntropyLoss()
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    labels_all = []
    preds_all = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        if training:
            optimizer.zero_grad()

        logits, _, _ = model(input_ids, mask)
        loss = criterion(logits, labels)

        if training:
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += float(loss.item())
        preds = logits.argmax(dim=-1)
        labels_all.extend(labels.cpu().tolist())
        preds_all.extend(preds.cpu().tolist())

    acc = float(accuracy_score(labels_all, preds_all)) if labels_all else 0.0
    return total_loss / max(1, len(loader)), acc


def evaluate_transformer(model, loader, device: str):
    model.eval()
    labels_all, preds_all = [], []
    sample_batches = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].to(device)
            logits, attn_stack, full_mask = model(input_ids, mask)
            preds = logits.argmax(dim=-1)

            labels_all.extend(labels.cpu().tolist())
            preds_all.extend(preds.cpu().tolist())
            sample_batches.append((batch, preds.cpu(), attn_stack, full_mask.cpu()))

    acc = float(accuracy_score(labels_all, preds_all)) if labels_all else 0.0
    macro_f1 = float(f1_score(labels_all, preds_all, average="macro", zero_division=0)) if labels_all else 0.0
    cm = confusion_matrix(labels_all, preds_all, labels=[0, 1, 2, 3, 4]).tolist()
    return {"accuracy": acc, "macro_f1": macro_f1, "confusion_matrix": cm, "samples": sample_batches}


def train_bilstm_baseline(
    model,
    train_loader,
    val_loader,
    epochs: int,
    device: str,
    lr: float = 1e-3,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "epoch_time": []}
    best_acc = -1.0
    best_state = None

    model.to(device)
    for _ in range(epochs):
        start = time.perf_counter()
        model.train()

        train_loss = 0.0
        y_true, y_pred = [], []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(dim=-1).cpu().tolist())

        train_acc = float(accuracy_score(y_true, y_pred)) if y_true else 0.0

        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                mask = batch["mask"].to(device)
                labels = batch["label"].to(device)

                logits = model(input_ids, mask)
                loss = criterion(logits, labels)
                val_loss += float(loss.item())
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(logits.argmax(dim=-1).cpu().tolist())

        val_acc = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
        epoch_time = time.perf_counter() - start

        history["train_loss"].append(train_loss / max(1, len(train_loader)))
        history["val_loss"].append(val_loss / max(1, len(val_loader)))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def eval_bilstm(model, loader, device: str):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, mask)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
    acc = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0
    return {"accuracy": acc, "macro_f1": macro_f1}


def plot_training_curves(history: Dict[str, List[float]], path: Path, title: str):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.title(f"{title} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.title(f"{title} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_attention_heatmaps(
    samples,
    idx2word: Sequence[str],
    out_dir: Path,
    max_articles: int = 3,
    heads: Sequence[int] = (0, 1),
):
    saved = 0
    for batch, preds, attn_stack, full_mask in samples:
        labels = batch["label"]
        input_ids = batch["input_ids"]
        for i in range(input_ids.size(0)):
            if int(preds[i].item()) != int(labels[i].item()):
                continue

            valid = int(full_mask[i].sum().item())
            tokens = ["[CLS]"] + [idx2word[t.item()] for t in input_ids[i][: valid - 1]]
            tokens = tokens[:30]
            n = len(tokens)

            final_attn = attn_stack[-1][i].cpu().numpy()  # [H, L, L]
            for h in heads:
                if h >= final_attn.shape[0]:
                    continue
                mat = final_attn[h, :n, :n]
                plt.figure(figsize=(8, 6))
                plt.imshow(mat, aspect="auto", cmap="viridis")
                plt.colorbar()
                plt.xticks(range(n), tokens, rotation=90, fontsize=7)
                plt.yticks(range(n), tokens, fontsize=7)
                plt.title(f"Final-layer Attention: article={saved + 1}, head={h}")
                plt.tight_layout()
                plt.savefig(out_dir / f"part3_attention_article{saved + 1}_head{h}.png", dpi=160)
                plt.close()

            saved += 1
            if saved >= max_articles:
                return


def build_comparison_text(
    transformer_metrics: Dict[str, float],
    bilstm_metrics: Dict[str, float],
    transformer_history: Dict[str, List[float]],
    bilstm_history: Dict[str, List[float]],
) -> str:
    t_acc = transformer_metrics["accuracy"]
    b_acc = bilstm_metrics["accuracy"]
    acc_gap = t_acc - b_acc

    t_best_epoch = int(np.argmax(transformer_history["val_acc"])) + 1
    b_best_epoch = int(np.argmax(bilstm_history["val_acc"])) + 1

    t_epoch_time = float(np.mean(transformer_history["epoch_time"])) if transformer_history["epoch_time"] else 0.0
    b_epoch_time = float(np.mean(bilstm_history["epoch_time"])) if bilstm_history["epoch_time"] else 0.0

    lines = [
        "The Transformer and BiLSTM were evaluated on the same 5-class split.",
        f"The Transformer reached test accuracy {t_acc:.4f}, while the BiLSTM reached {b_acc:.4f}.",
        f"The absolute accuracy difference is {acc_gap:.4f} in favour of {'Transformer' if acc_gap >= 0 else 'BiLSTM'}.",
        f"Transformer validation accuracy peaked at epoch {t_best_epoch}.",
        f"BiLSTM validation accuracy peaked at epoch {b_best_epoch}.",
        (
            "The model with fewer epochs to peak validation accuracy appears to converge faster; "
            f"in this run that was {'Transformer' if t_best_epoch <= b_best_epoch else 'BiLSTM'}."
        ),
        f"Average Transformer training time per epoch was {t_epoch_time:.2f} seconds.",
        f"Average BiLSTM training time per epoch was {b_epoch_time:.2f} seconds.",
        (
            "Per-epoch runtime differs because multi-head attention performs dense token-token interactions, "
            "while recurrent models process steps sequentially and can be cheaper for short sequences."
        ),
        (
            "Attention heatmaps in the final encoder layer highlight which tokens influence the [CLS] representation, "
            "and they can show focused patterns around topic-indicative words."
        ),
        (
            "When the dataset is very small (around 200-300 articles), a BiLSTM can be more data-efficient and less prone to overfitting."
        ),
        (
            "A Transformer may still win if regularisation and learning-rate scheduling are tuned well, "
            "but it usually needs stronger supervision to stabilise."
        ),
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assignment 2 - Part 3 Transformer Topic Classification")
    parser.add_argument("--cleaned", type=Path, default=Path("cleaned.txt"))
    parser.add_argument("--metadata", type=Path, default=Path("Metadata.json"))
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    docs = read_corpus(args.cleaned)
    metadata = read_metadata(args.metadata) if args.metadata.exists() else ["unknown"] * len(docs)
    topic_labels = extract_topic_labels(metadata, len(docs))

    word2idx, idx2word, _ = build_vocab(docs, max_vocab_size=10000, include_cls=True)
    sequences, labels = build_topic_dataset(docs, topic_labels, word2idx, max_len=args.max_len)

    tr_idx, va_idx, te_idx = stratified_split_indices([str(y) for y in labels], seed=args.seed)
    tr_seqs = [sequences[i] for i in tr_idx]
    va_seqs = [sequences[i] for i in va_idx]
    te_seqs = [sequences[i] for i in te_idx]
    tr_labels = [labels[i] for i in tr_idx]
    va_labels = [labels[i] for i in va_idx]
    te_labels = [labels[i] for i in te_idx]

    train_ds = TopicDataset(tr_seqs, tr_labels, max_len=args.max_len, pad_id=word2idx[PAD_TOKEN])
    val_ds = TopicDataset(va_seqs, va_labels, max_len=args.max_len, pad_id=word2idx[PAD_TOKEN])
    test_ds = TopicDataset(te_seqs, te_labels, max_len=args.max_len, pad_id=word2idx[PAD_TOKEN])

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False)
    test_loader = make_loader(test_ds, args.batch_size, shuffle=False)

    output_root = ensure_dir(args.output_root)
    models_dir = ensure_dir(output_root / "models")
    reports_dir = ensure_dir(output_root / "reports")

    # Transformer model.
    transformer = TransformerClassifier(
        vocab_size=len(word2idx),
        num_classes=5,
        d_model=128,
        num_heads=4,
        d_k=32,
        d_v=32,
        d_ff=512,
        num_layers=4,
        dropout=0.1,
        pad_id=word2idx[PAD_TOKEN],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=5e-4, weight_decay=0.01)

    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = 50

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    t_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "epoch_time": []}

    for epoch in range(args.epochs):
        start = time.perf_counter()
        transformer.train()
        train_loss = 0.0
        y_true, y_pred = [], []

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)
            labels_batch = batch["label"].to(device)

            optimizer.zero_grad()
            logits, _, _ = transformer(input_ids, mask)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += float(loss.item())
            y_true.extend(labels_batch.cpu().tolist())
            y_pred.extend(logits.argmax(dim=-1).cpu().tolist())

        train_acc = float(accuracy_score(y_true, y_pred)) if y_true else 0.0

        val_loss, val_acc = run_epoch_transformer(transformer, val_loader, optimizer=None, scheduler=None, device=device, training=False)

        t_history["train_loss"].append(train_loss / max(1, len(train_loader)))
        t_history["val_loss"].append(val_loss)
        t_history["train_acc"].append(train_acc)
        t_history["val_acc"].append(val_acc)
        t_history["epoch_time"].append(time.perf_counter() - start)

    plot_training_curves(t_history, reports_dir / "part3_transformer_curves.png", "Transformer")

    transformer_eval = evaluate_transformer(transformer, test_loader, device=device)
    save_attention_heatmaps(transformer_eval["samples"], idx2word, reports_dir, max_articles=3, heads=(0, 1))

    torch.save(transformer.state_dict(), models_dir / "transformer_cls.pt")

    # BiLSTM baseline for comparison.
    bilstm = BiLSTMClassifier(vocab_size=len(word2idx), d_model=128, hidden=128, pad_id=word2idx[PAD_TOKEN], num_classes=5)
    bilstm, b_history = train_bilstm_baseline(bilstm, train_loader, val_loader, epochs=args.epochs, device=device)
    plot_training_curves(b_history, reports_dir / "part3_bilstm_curves.png", "BiLSTM")
    bilstm_eval = eval_bilstm(bilstm.to(device), test_loader, device)

    comparison_text = build_comparison_text(
        transformer_metrics={"accuracy": transformer_eval["accuracy"]},
        bilstm_metrics={"accuracy": bilstm_eval["accuracy"]},
        transformer_history=t_history,
        bilstm_history=b_history,
    )
    (reports_dir / "part3_bilstm_vs_transformer.txt").write_text(comparison_text, encoding="utf-8")

    class_dist = Counter(labels)
    summary = {
        "dataset": {
            "class_distribution": {CATEGORY_MAP[k]: int(v) for k, v in class_dist.items()},
            "split_sizes": {
                "train": len(tr_labels),
                "val": len(va_labels),
                "test": len(te_labels),
            },
        },
        "transformer": {
            "test_accuracy": transformer_eval["accuracy"],
            "test_macro_f1": transformer_eval["macro_f1"],
            "confusion_matrix": transformer_eval["confusion_matrix"],
            "mean_epoch_time_sec": float(np.mean(t_history["epoch_time"])) if t_history["epoch_time"] else 0.0,
        },
        "bilstm_baseline": {
            "test_accuracy": bilstm_eval["accuracy"],
            "test_macro_f1": bilstm_eval["macro_f1"],
            "mean_epoch_time_sec": float(np.mean(b_history["epoch_time"])) if b_history["epoch_time"] else 0.0,
        },
    }
    save_json(reports_dir / "part3_results.json", summary)
    print("Part 3 completed.")


if __name__ == "__main__":
    main()
