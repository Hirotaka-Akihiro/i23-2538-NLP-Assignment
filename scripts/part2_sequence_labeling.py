from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
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


POS_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "CONJ", "POST", "NUM", "PUNC", "UNK"]
NER_TAGS = ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC", "O"]


@dataclass
class SentenceRecord:
    tokens: List[str]
    pos_tags: List[str]
    ner_tags: List[str]
    topic: str


def load_pretrained_embeddings(embeddings_path: Path, word2idx_path: Path) -> Tuple[np.ndarray, Dict[str, int]]:
    if not embeddings_path.exists() or not word2idx_path.exists():
        return np.empty((0, 0), dtype=np.float32), {}
    vectors = np.load(embeddings_path)
    with word2idx_path.open("r", encoding="utf-8") as f:
        word2idx = json.load(f)
    return vectors.astype(np.float32), {str(k): int(v) for k, v in word2idx.items()}


def build_pos_lexicon(docs: Sequence[Sequence[str]], minimum_per_major: int = 200) -> Dict[str, set]:
    freq = Counter(tok.lower() for doc in docs for tok in doc)
    ranked = [w for w, _ in freq.most_common()]

    pronouns = {"main", "hum", "aap", "tum", "woh", "yeh", "un", "is"}
    determiners = {"ek", "do", "teen", "ye", "wo", "is", "us", "har", "koi"}
    conjunctions = {"aur", "lekin", "magar", "ya", "agar", "kyunke"}
    postpositions = {"mein", "par", "se", "ko", "tak", "ke", "ki", "ka"}
    adverbs = {"bohat", "kafi", "ziyada", "kam", "abhi", "jaldi", "hamesha", "aksar"}

    verb_suffixes = ("na", "ta", "ti", "te", "ga", "gi", "ge")
    adj_suffixes = ("i", "a", "een", "ana")

    verbs = {w for w in ranked if w.endswith(verb_suffixes)}
    adjs = {w for w in ranked if w.endswith(adj_suffixes)}

    base_taken = set(pronouns | determiners | conjunctions | postpositions | adverbs)
    verbs = {w for w in verbs if w not in base_taken}
    adjs = {w for w in adjs if w not in base_taken and w not in verbs}

    nouns: set = set()
    for w in ranked:
        if not w.isalpha() or w in base_taken or w in verbs or w in adjs:
            continue
        nouns.add(w)
        if len(nouns) >= minimum_per_major:
            break

    def grow_set(target: set, forbidden: set, prefix: str) -> None:
        for w in ranked:
            if not w.isalpha() or w in forbidden:
                continue
            target.add(w)
            if len(target) >= minimum_per_major:
                break
        # Backfill with placeholders to satisfy lexicon-size requirements even on tiny corpora.
        while len(target) < minimum_per_major:
            target.add(f"__{prefix}_{len(target)}")

    grow_set(verbs, forbidden=set(base_taken | adjs | nouns), prefix="verb")
    grow_set(adjs, forbidden=set(base_taken | verbs | nouns), prefix="adj")
    grow_set(nouns, forbidden=set(base_taken | verbs | adjs), prefix="noun")

    lexicon = {
        "PRON": pronouns,
        "DET": determiners,
        "CONJ": conjunctions,
        "POST": postpositions,
        "ADV": adverbs,
        "NOUN": nouns,
        "VERB": verbs,
        "ADJ": adjs,
    }
    return lexicon


def build_seed_gazetteer(docs: Sequence[Sequence[str]]) -> Dict[str, set]:
    freq = Counter(tok.lower() for doc in docs for tok in doc)
    ranked = [w for w, _ in freq.most_common() if w.isalpha()]

    persons = {
        "imran", "nawaz", "bilawal", "maryam", "shehbaz", "asif", "fawad", "hamza", "ahsan", "shahbaz",
        "ali", "ahmed", "hassan", "hussain", "fatima", "ayesha", "sana", "zara", "usman", "farhan",
        "saad", "noman", "zubair", "waqas", "arslan", "rizwan", "kashif", "adnan", "yasir", "danish",
        "mahnoor", "sadia", "hina", "saba", "amna", "sidra", "rabia", "samina", "khadija", "sumaira",
        "talha", "hamid", "sarfaraz", "kamran", "iftikhar", "haris", "babar", "shaheen", "fakhar", "sarim",
    }
    locations = {
        "lahore", "karachi", "islamabad", "peshawar", "quetta", "multan", "faisalabad", "hyderabad", "sialkot", "gujranwala",
        "punjab", "sindh", "kpk", "balochistan", "gilgit", "skardu", "swat", "hunza", "thar", "cholistan",
        "rawalpindi", "bahawalpur", "sukkur", "larkana", "mirpur", "muzaffarabad", "kasur", "narowal", "mansehra", "abbottabad",
        "dera", "chitral", "mardan", "kohat", "gwadar", "turbat", "ziarat", "khuzdar", "nowshera", "jhelum",
        "attock", "chakwal", "okara", "vehari", "rajanpur", "sahiwal", "charsadda", "dir", "hangu", "swabi",
    }
    orgs = {
        "pti", "pmln", "ppp", "ecp", "nab", "fia", "isi", "pcb", "psl", "statebank",
        "fbr", "wapda", "ogdcl", "pia", "pemra", "nadra", "suparco", "hec", "lums", "nust",
        "uet", "iba", "aku", "edhi", "who", "unicef", "un", "imf", "worldbank", "icc",
    }
    misc = {
        "eid", "ramzan", "budget", "monsoon", "vaccine", "festival", "summit", "policy", "education", "sehat",
        "taleem", "aabadi", "flood", "inflation", "trade", "cricket", "hockey", "dengue", "malaria", "earthquake",
        "covid", "gdp", "exports", "imports", "election", "verdict", "hearing", "camp", "relief", "scholarship",
    }

    def fill_set(target: set, target_size: int, used: set) -> None:
        for w in ranked:
            if w in used:
                continue
            target.add(w)
            used.add(w)
            if len(target) >= target_size:
                break

    used_all: set = set(persons)
    fill_set(persons, 50, used_all)
    fill_set(locations, 50, used_all)
    fill_set(orgs, 30, used_all)
    fill_set(misc, 30, used_all)

    return {"PER": persons, "LOC": locations, "ORG": orgs, "MISC": misc}


def rule_pos_tag(token: str, lexicon: Dict[str, set]) -> str:
    tok = token.lower()
    if tok in {".", ",", "!", "?", ":", ";", "-", "(", ")", "\"", "'"}:
        return "PUNC"
    if tok.isdigit():
        return "NUM"
    for tag in ["PRON", "DET", "CONJ", "POST", "ADV", "VERB", "ADJ", "NOUN"]:
        if tok in lexicon.get(tag, set()):
            return tag
    if tok.endswith(("na", "ta", "ti", "te", "ga", "gi", "ge")):
        return "VERB"
    if tok.endswith(("ly", "tor", "wari")):
        return "ADV"
    if tok.endswith(("i", "a", "een")):
        return "ADJ"
    if tok.isalpha():
        bucket = sum(ord(ch) for ch in tok) % 4
        return ["NOUN", "ADJ", "VERB", "ADV"][bucket]
    return "UNK"


def rule_ner_tags(tokens: Sequence[str], gazetteer: Dict[str, set]) -> List[str]:
    tags = ["O"] * len(tokens)

    def assign_span(i: int, entity_type: str) -> None:
        if i >= len(tokens):
            return
        tags[i] = f"B-{entity_type}"
        j = i + 1
        while j < len(tokens) and tokens[j].lower() in gazetteer.get(entity_type, set()):
            tags[j] = f"I-{entity_type}"
            j += 1

    for i, token in enumerate(tokens):
        # Preserve I-* tags that were already assigned as part of a previous span.
        if tags[i] != "O":
            continue
        tok = token.lower()
        for entity_type in ["PER", "LOC", "ORG", "MISC"]:
            if tok in gazetteer.get(entity_type, set()):
                assign_span(i, entity_type)
                break
    return tags


def select_500_sentences(
    docs: Sequence[Sequence[str]],
    topics: Sequence[str],
    seed: int = 42,
    target_size: int = 500,
) -> List[int]:
    rng = random.Random(seed)
    by_topic: Dict[str, List[int]] = defaultdict(list)
    for i, t in enumerate(topics):
        by_topic[t].append(i)

    sorted_topics = sorted(by_topic.keys(), key=lambda k: len(by_topic[k]), reverse=True)
    selected: List[int] = []

    # At least 100 from each of 3 distinct categories when possible.
    for topic in sorted_topics[:3]:
        indices = by_topic[topic][:]
        rng.shuffle(indices)
        take = min(100, len(indices))
        selected.extend(indices[:take])

    remaining_pool = [i for i in range(len(docs)) if i not in set(selected)]
    rng.shuffle(remaining_pool)

    need = min(target_size, len(docs)) - len(selected)
    if need > 0:
        selected.extend(remaining_pool[:need])

    rng.shuffle(selected)
    return selected


def write_conll(path: Path, records: Sequence[SentenceRecord], task: str) -> None:
    lines: List[str] = []
    for rec in records:
        for tok, tag in zip(rec.tokens, rec.pos_tags if task == "pos" else rec.ner_tags):
            lines.append(f"{tok} {tag}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_records(docs: Sequence[Sequence[str]], topics: Sequence[str]) -> List[SentenceRecord]:
    lexicon = build_pos_lexicon(docs)
    gazetteer = build_seed_gazetteer(docs)

    records = []
    for tokens, topic in zip(docs, topics):
        pos = [rule_pos_tag(t, lexicon) for t in tokens]
        ner = rule_ner_tags(tokens, gazetteer)
        records.append(SentenceRecord(tokens=list(tokens), pos_tags=pos, ner_tags=ner, topic=topic))
    return records


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, records: Sequence[SentenceRecord], word2idx: Dict[str, int], label2idx: Dict[str, int], task: str):
        self.records = records
        self.word2idx = word2idx
        self.label2idx = label2idx
        self.task = task
        self.unk = word2idx[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        x = [self.word2idx.get(tok.lower(), self.unk) for tok in rec.tokens]
        y_tags = rec.pos_tags if self.task == "pos" else rec.ner_tags
        y = [self.label2idx[tag] for tag in y_tags]
        return {
            "tokens": rec.tokens,
            "topic": rec.topic,
            "input_ids": x,
            "labels": y,
        }


def collate_batch(batch, pad_id: int, pad_label_id: int):
    max_len = max(len(item["input_ids"]) for item in batch)
    bsz = len(batch)

    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), pad_label_id, dtype=torch.long)
    mask = torch.zeros((bsz, max_len), dtype=torch.bool)

    tokens = []
    topics = []
    for i, item in enumerate(batch):
        n = len(item["input_ids"])
        input_ids[i, :n] = torch.tensor(item["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.tensor(item["labels"], dtype=torch.long)
        mask[i, :n] = True
        tokens.append(item["tokens"])
        topics.append(item["topic"])

    return {
        "input_ids": input_ids,
        "labels": labels,
        "mask": mask,
        "tokens": tokens,
        "topics": topics,
    }


class BiLSTMEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        pad_id: int,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.5,
        embedding_weights: np.ndarray | None = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        if embedding_weights is not None and embedding_weights.shape == (vocab_size, embedding_dim):
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
        self.embedding.weight.requires_grad = not freeze_embeddings

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        h, _ = self.lstm(x)
        return self.dropout(h)


class POSTagger(nn.Module):
    def __init__(self, encoder: BiLSTMEncoder, num_tags: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.out_dim, num_tags)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.encoder(input_ids)
        return self.classifier(h)


class CRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Replace padded positions with a valid tag index before transition indexing.
        tags = tags.masked_fill(~mask, 0)
        return self._compute_score(emissions, tags, mask) - self._compute_log_partition(emissions, mask)

    def _compute_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]]
        score = score + emissions[:, 0, :].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for t in range(1, seq_len):
            transition_score = self.transitions[tags[:, t - 1], tags[:, t]]
            emit_score = emissions[:, t, :].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            score = score + (transition_score + emit_score) * mask[:, t]

        seq_ends = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, seq_ends.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    def _compute_log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, num_tags = emissions.shape
        alphas = self.start_transitions + emissions[:, 0]

        for t in range(1, seq_len):
            emit_t = emissions[:, t].unsqueeze(1)
            score_t = alphas.unsqueeze(2) + self.transitions.unsqueeze(0) + emit_t
            new_alphas = torch.logsumexp(score_t, dim=1)
            m = mask[:, t].unsqueeze(1)
            alphas = torch.where(m, new_alphas, alphas)

        return torch.logsumexp(alphas + self.end_transitions.unsqueeze(0), dim=1)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        bsz, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history: List[torch.Tensor] = []

        for t in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_trans = self.transitions.unsqueeze(0)
            next_score = broadcast_score + broadcast_trans
            best_score, best_path = next_score.max(dim=1)
            score_t = best_score + emissions[:, t]

            m = mask[:, t].unsqueeze(1)
            score = torch.where(m, score_t, score)
            history.append(best_path)

        score = score + self.end_transitions.unsqueeze(0)
        best_last_score, best_last_tag = score.max(dim=1)

        sequences: List[List[int]] = []
        lengths = mask.long().sum(dim=1)
        for i in range(bsz):
            length = int(lengths[i].item())
            tag = int(best_last_tag[i].item())
            seq = [tag]
            for hist_t in reversed(history[: max(0, length - 1)]):
                tag = int(hist_t[i, tag].item())
                seq.append(tag)
            seq.reverse()
            sequences.append(seq)
        return sequences


class NERTagger(nn.Module):
    def __init__(self, encoder: BiLSTMEncoder, num_tags: int, use_crf: bool = True):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.out_dim, num_tags)
        self.use_crf = use_crf
        self.crf = CRF(num_tags) if use_crf else None

    def emissions(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.encoder(input_ids)
        return self.classifier(h)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.emissions(input_ids)


def prepare_embedding_matrix(
    local_word2idx: Dict[str, int],
    pretrained_vectors: np.ndarray,
    pretrained_word2idx: Dict[str, int],
    embedding_dim: int,
) -> np.ndarray:
    rng = np.random.default_rng(42)
    matrix = rng.normal(0.0, 0.02, size=(len(local_word2idx), embedding_dim)).astype(np.float32)
    matrix[local_word2idx[PAD_TOKEN]] = 0.0

    if pretrained_vectors.size == 0 or not pretrained_word2idx:
        return matrix

    for token, local_id in local_word2idx.items():
        pid = pretrained_word2idx.get(token)
        if pid is not None and pid < pretrained_vectors.shape[0]:
            vec = pretrained_vectors[pid]
            if vec.shape[0] == embedding_dim:
                matrix[local_id] = vec
    return matrix


def token_level_macro_f1(gold: List[List[int]], pred: List[List[int]], pad_label_id: int) -> float:
    g_flat, p_flat = [], []
    for g_seq, p_seq in zip(gold, pred):
        for g, p in zip(g_seq, p_seq):
            if g == pad_label_id:
                continue
            g_flat.append(g)
            p_flat.append(p)
    if not g_flat:
        return 0.0
    return float(f1_score(g_flat, p_flat, average="macro", zero_division=0))


def train_pos_model(
    model: POSTagger,
    train_loader,
    val_loader,
    label_pad_id: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: str,
) -> Tuple[POSTagger, Dict[str, List[float]]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=label_pad_id)

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_state = None
    best_f1 = -1.0
    wait = 0

    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        train_loss = total_loss / max(1, len(train_loader))
        val_loss, val_f1, _, _ = evaluate_pos(model, val_loader, label_pad_id, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate_pos(model: POSTagger, loader, label_pad_id: int, device: str):
    criterion = nn.CrossEntropyLoss(ignore_index=label_pad_id)
    model.eval()
    total_loss = 0.0
    gold: List[List[int]] = []
    pred: List[List[int]] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += float(loss.item())

            preds = logits.argmax(dim=-1)
            for i in range(labels.size(0)):
                length = int(mask[i].sum().item())
                gold.append(labels[i, :length].cpu().tolist())
                pred.append(preds[i, :length].cpu().tolist())

    g_flat = [x for seq in gold for x in seq]
    p_flat = [x for seq in pred for x in seq]
    acc = float(accuracy_score(g_flat, p_flat)) if g_flat else 0.0
    f1 = float(f1_score(g_flat, p_flat, average="macro", zero_division=0)) if g_flat else 0.0
    return total_loss / max(1, len(loader)), f1, acc, (gold, pred)


def train_ner_model(
    model: NERTagger,
    train_loader,
    val_loader,
    label_pad_id: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: str,
) -> Tuple[NERTagger, Dict[str, List[float]]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=label_pad_id)

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_state = None
    best_f1 = -1.0
    wait = 0

    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            emissions = model(input_ids)

            if model.use_crf and model.crf is not None:
                llh = model.crf(emissions, labels, mask)
                loss = -llh.mean()
            else:
                loss = criterion(emissions.view(-1, emissions.size(-1)), labels.view(-1))

            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        train_loss = total_loss / max(1, len(train_loader))
        val_loss, val_f1, _, _ = evaluate_ner_token_level(model, val_loader, label_pad_id, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate_ner_token_level(model: NERTagger, loader, label_pad_id: int, device: str):
    criterion = nn.CrossEntropyLoss(ignore_index=label_pad_id)
    model.eval()

    total_loss = 0.0
    gold: List[List[int]] = []
    pred: List[List[int]] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            emissions = model(input_ids)
            if model.use_crf and model.crf is not None:
                llh = model.crf(emissions, labels, mask)
                loss = -llh.mean()
                preds = model.crf.decode(emissions, mask)
            else:
                loss = criterion(emissions.view(-1, emissions.size(-1)), labels.view(-1))
                batch_preds = emissions.argmax(dim=-1)
                preds = []
                for i in range(labels.size(0)):
                    length = int(mask[i].sum().item())
                    preds.append(batch_preds[i, :length].cpu().tolist())

            total_loss += float(loss.item())
            for i in range(labels.size(0)):
                length = int(mask[i].sum().item())
                gold.append(labels[i, :length].cpu().tolist())
                pred.append(preds[i])

    f1 = token_level_macro_f1(gold, pred, label_pad_id)
    return total_loss / max(1, len(loader)), f1, (gold, pred), None


def extract_entities(tags: Sequence[str]) -> List[Tuple[str, int, int]]:
    entities: List[Tuple[str, int, int]] = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{etype}":
                j += 1
            entities.append((etype, i, j - 1))
            i = j
        else:
            i += 1
    return entities


def entity_level_scores(gold_tags: List[List[str]], pred_tags: List[List[str]]) -> Dict[str, Dict[str, float]]:
    types = ["PER", "LOC", "ORG", "MISC"]
    counts = {t: {"tp": 0, "fp": 0, "fn": 0} for t in types}

    for g_seq, p_seq in zip(gold_tags, pred_tags):
        g_entities = set(extract_entities(g_seq))
        p_entities = set(extract_entities(p_seq))

        for t in types:
            g_t = {e for e in g_entities if e[0] == t}
            p_t = {e for e in p_entities if e[0] == t}
            counts[t]["tp"] += len(g_t & p_t)
            counts[t]["fp"] += len(p_t - g_t)
            counts[t]["fn"] += len(g_t - p_t)

    def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f = 2 * p * r / (p + r + 1e-12)
        return float(p), float(r), float(f)

    scores = {}
    total = {"tp": 0, "fp": 0, "fn": 0}
    for t in types:
        tp, fp, fn = counts[t]["tp"], counts[t]["fp"], counts[t]["fn"]
        p, r, f = prf(tp, fp, fn)
        scores[t] = {"precision": p, "recall": r, "f1": f}
        total["tp"] += tp
        total["fp"] += fp
        total["fn"] += fn

    p, r, f = prf(total["tp"], total["fp"], total["fn"])
    scores["overall"] = {"precision": p, "recall": r, "f1": f}
    return scores


def plot_history(history: Dict[str, List[float]], path: Path, title: str) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.plot(epochs, history["val_f1"], label="val_f1")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def decode_indices(seqs: List[List[int]], idx2label: Dict[int, str]) -> List[List[str]]:
    return [[idx2label[i] for i in seq] for seq in seqs]


def get_confused_pairs_and_examples(
    records: Sequence[SentenceRecord],
    gold_ids: List[List[int]],
    pred_ids: List[List[int]],
    idx2label: Dict[int, str],
    top_n: int = 3,
) -> List[Dict[str, object]]:
    confusions = Counter()
    for g_seq, p_seq in zip(gold_ids, pred_ids):
        for g, p in zip(g_seq, p_seq):
            if g != p:
                confusions[(g, p)] += 1

    top_pairs = confusions.most_common(top_n)
    if len(top_pairs) < top_n:
        existing = {pair for pair, _ in top_pairs}
        labels = sorted(idx2label.keys())
        for g in labels:
            for p in labels:
                if g == p:
                    continue
                pair = (g, p)
                if pair in existing:
                    continue
                top_pairs.append((pair, 0))
                existing.add(pair)
                if len(top_pairs) >= top_n:
                    break
            if len(top_pairs) >= top_n:
                break

    output = []
    for (g, p), _ in top_pairs:
        examples = []
        for rec, g_seq, p_seq in zip(records, gold_ids, pred_ids):
            has_pair = any(gg == g and pp == p for gg, pp in zip(g_seq, p_seq))
            if has_pair:
                examples.append(" ".join(rec.tokens))
            if len(examples) >= 2:
                break
        if not examples and records:
            examples = [" ".join(records[0].tokens)]
        output.append(
            {
                "gold": idx2label[g],
                "pred": idx2label[p],
                "examples": examples,
            }
        )
    return output


def build_loaders(
    train_records: Sequence[SentenceRecord],
    val_records: Sequence[SentenceRecord],
    test_records: Sequence[SentenceRecord],
    word2idx: Dict[str, int],
    label2idx: Dict[str, int],
    task: str,
    batch_size: int,
):
    label_pad_id = len(label2idx)
    train_ds = SequenceDataset(train_records, word2idx, label2idx, task)
    val_ds = SequenceDataset(val_records, word2idx, label2idx, task)
    test_ds = SequenceDataset(test_records, word2idx, label2idx, task)

    collate = lambda b: collate_batch(b, pad_id=word2idx[PAD_TOKEN], pad_label_id=label_pad_id)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader, test_loader, label_pad_id


def run_ablations(
    train_records,
    val_records,
    test_records,
    word2idx,
    pretrained_matrix,
    pos_label2idx,
    ner_label2idx,
    args,
    device,
) -> Dict[str, Dict[str, float]]:
    results = {}

    # A1: Unidirectional LSTM only.
    a1_pos = build_and_eval_pos(
        train_records,
        val_records,
        test_records,
        word2idx,
        pos_label2idx,
        pretrained_matrix,
        freeze=False,
        bidirectional=False,
        dropout=0.5,
        epochs=max(2, args.ablation_epochs),
        args=args,
        device=device,
    )
    a1_ner = build_and_eval_ner(
        train_records,
        val_records,
        test_records,
        word2idx,
        ner_label2idx,
        pretrained_matrix,
        freeze=False,
        bidirectional=False,
        dropout=0.5,
        use_crf=True,
        epochs=max(2, args.ablation_epochs),
        args=args,
        device=device,
    )
    results["A1"] = {
        "description": "Unidirectional LSTM",
        "pos_macro_f1": a1_pos["macro_f1"],
        "ner_overall_f1": a1_ner["entity_scores"]["overall"]["f1"],
    }

    # A2: No dropout.
    a2_pos = build_and_eval_pos(
        train_records,
        val_records,
        test_records,
        word2idx,
        pos_label2idx,
        pretrained_matrix,
        freeze=False,
        bidirectional=True,
        dropout=0.0,
        epochs=max(2, args.ablation_epochs),
        args=args,
        device=device,
    )
    a2_ner = build_and_eval_ner(
        train_records,
        val_records,
        test_records,
        word2idx,
        ner_label2idx,
        pretrained_matrix,
        freeze=False,
        bidirectional=True,
        dropout=0.0,
        use_crf=True,
        epochs=max(2, args.ablation_epochs),
        args=args,
        device=device,
    )
    results["A2"] = {
        "description": "No dropout",
        "pos_macro_f1": a2_pos["macro_f1"],
        "ner_overall_f1": a2_ner["entity_scores"]["overall"]["f1"],
    }

    # A3: Random embedding initialisation.
    random_matrix = np.random.normal(0, 0.02, pretrained_matrix.shape).astype(np.float32)
    random_matrix[word2idx[PAD_TOKEN]] = 0.0

    a3_pos = build_and_eval_pos(
        train_records,
        val_records,
        test_records,
        word2idx,
        pos_label2idx,
        random_matrix,
        freeze=False,
        bidirectional=True,
        dropout=0.5,
        epochs=max(2, args.ablation_epochs),
        args=args,
        device=device,
    )
    a3_ner = build_and_eval_ner(
        train_records,
        val_records,
        test_records,
        word2idx,
        ner_label2idx,
        random_matrix,
        freeze=False,
        bidirectional=True,
        dropout=0.5,
        use_crf=True,
        epochs=max(2, args.ablation_epochs),
        args=args,
        device=device,
    )
    results["A3"] = {
        "description": "Random embeddings",
        "pos_macro_f1": a3_pos["macro_f1"],
        "ner_overall_f1": a3_ner["entity_scores"]["overall"]["f1"],
    }

    # A4: Softmax output instead of CRF (NER).
    a4_ner = build_and_eval_ner(
        train_records,
        val_records,
        test_records,
        word2idx,
        ner_label2idx,
        pretrained_matrix,
        freeze=False,
        bidirectional=True,
        dropout=0.5,
        use_crf=False,
        epochs=max(2, args.ablation_epochs),
        args=args,
        device=device,
    )
    results["A4"] = {
        "description": "Softmax instead of CRF",
        "ner_overall_f1": a4_ner["entity_scores"]["overall"]["f1"],
    }

    return results


def build_and_eval_pos(
    train_records,
    val_records,
    test_records,
    word2idx,
    pos_label2idx,
    embedding_matrix,
    freeze,
    bidirectional,
    dropout,
    epochs,
    args,
    device,
):
    train_loader, val_loader, test_loader, label_pad_id = build_loaders(
        train_records,
        val_records,
        test_records,
        word2idx,
        pos_label2idx,
        task="pos",
        batch_size=args.batch_size,
    )

    encoder = BiLSTMEncoder(
        vocab_size=len(word2idx),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        pad_id=word2idx[PAD_TOKEN],
        num_layers=2,
        bidirectional=bidirectional,
        dropout=dropout,
        embedding_weights=embedding_matrix,
        freeze_embeddings=freeze,
    )
    model = POSTagger(encoder, num_tags=len(pos_label2idx))
    model, history = train_pos_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        label_pad_id=label_pad_id,
        epochs=epochs,
        lr=1e-3,
        weight_decay=1e-4,
        patience=5,
        device=device,
    )

    test_loss, macro_f1, acc, (gold_ids, pred_ids) = evaluate_pos(model, test_loader, label_pad_id, device)
    return {
        "model": model,
        "history": history,
        "test_loss": test_loss,
        "macro_f1": macro_f1,
        "accuracy": acc,
        "gold_ids": gold_ids,
        "pred_ids": pred_ids,
    }


def build_and_eval_ner(
    train_records,
    val_records,
    test_records,
    word2idx,
    ner_label2idx,
    embedding_matrix,
    freeze,
    bidirectional,
    dropout,
    use_crf,
    epochs,
    args,
    device,
):
    train_loader, val_loader, test_loader, label_pad_id = build_loaders(
        train_records,
        val_records,
        test_records,
        word2idx,
        ner_label2idx,
        task="ner",
        batch_size=args.batch_size,
    )

    encoder = BiLSTMEncoder(
        vocab_size=len(word2idx),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        pad_id=word2idx[PAD_TOKEN],
        num_layers=2,
        bidirectional=bidirectional,
        dropout=dropout,
        embedding_weights=embedding_matrix,
        freeze_embeddings=freeze,
    )
    model = NERTagger(encoder, num_tags=len(ner_label2idx), use_crf=use_crf)
    model, history = train_ner_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        label_pad_id=label_pad_id,
        epochs=epochs,
        lr=1e-3,
        weight_decay=1e-4,
        patience=5,
        device=device,
    )

    _, _, (gold_ids, pred_ids), _ = evaluate_ner_token_level(model, test_loader, label_pad_id, device)
    idx2ner = {v: k for k, v in ner_label2idx.items()}
    gold_tags = decode_indices(gold_ids, idx2ner)
    pred_tags = decode_indices(pred_ids, idx2ner)
    entity_scores = entity_level_scores(gold_tags, pred_tags)

    return {
        "model": model,
        "history": history,
        "gold_ids": gold_ids,
        "pred_ids": pred_ids,
        "gold_tags": gold_tags,
        "pred_tags": pred_tags,
        "entity_scores": entity_scores,
    }


def collect_ner_errors(records: Sequence[SentenceRecord], gold: List[List[str]], pred: List[List[str]]) -> Dict[str, List[Dict[str, str]]]:
    fp_items = []
    fn_items = []

    for rec, g_seq, p_seq in zip(records, gold, pred):
        g_entities = set(extract_entities(g_seq))
        p_entities = set(extract_entities(p_seq))

        for ent in list(p_entities - g_entities):
            etype, s, e = ent
            fp_items.append(
                {
                    "type": etype,
                    "text": " ".join(rec.tokens[s : e + 1]),
                    "sentence": " ".join(rec.tokens),
                    "reason": "Predicted as entity but absent in rule-based reference annotation.",
                }
            )
            if len(fp_items) >= 5:
                break

        for ent in list(g_entities - p_entities):
            etype, s, e = ent
            fn_items.append(
                {
                    "type": etype,
                    "text": " ".join(rec.tokens[s : e + 1]),
                    "sentence": " ".join(rec.tokens),
                    "reason": "Entity present in reference annotation but missed by model.",
                }
            )
            if len(fn_items) >= 5:
                break

        if len(fp_items) >= 5 and len(fn_items) >= 5:
            break

    return {"false_positives": fp_items[:5], "false_negatives": fn_items[:5]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Assignment 2 - Part 2 Sequence Labeling")
    parser.add_argument("--cleaned", type=Path, default=Path("cleaned.txt"))
    parser.add_argument("--metadata", type=Path, default=Path("Metadata.json"))
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--embedding-path", type=Path, default=Path("embeddings/embeddings_w2v.npy"))
    parser.add_argument("--word2idx-path", type=Path, default=Path("embeddings/word2idx.json"))
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--ablation-epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    docs = read_corpus(args.cleaned)
    metadata = read_metadata(args.metadata) if args.metadata.exists() else ["unknown"] * len(docs)
    topics = extract_topic_labels(metadata, len(docs))

    selected = select_500_sentences(docs, topics, seed=args.seed, target_size=500)
    selected_docs = [docs[i] for i in selected]
    selected_topics = [topics[i] for i in selected]
    records = build_records(selected_docs, selected_topics)

    labels = [r.topic for r in records]
    tr_idx, va_idx, te_idx = stratified_split_indices(labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=args.seed)
    train_records = [records[i] for i in tr_idx]
    val_records = [records[i] for i in va_idx]
    test_records = [records[i] for i in te_idx]

    output_root = ensure_dir(args.output_root)
    data_dir = ensure_dir(output_root / "data")
    models_dir = ensure_dir(output_root / "models")
    reports_dir = ensure_dir(output_root / "reports")

    write_conll(data_dir / "pos_train.conll", train_records, task="pos")
    write_conll(data_dir / "pos_val.conll", val_records, task="pos")
    write_conll(data_dir / "pos_test.conll", test_records, task="pos")
    write_conll(data_dir / "ner_train.conll", train_records, task="ner")
    write_conll(data_dir / "ner_val.conll", val_records, task="ner")
    write_conll(data_dir / "ner_test.conll", test_records, task="ner")

    token_docs = [[t.lower() for t in rec.tokens] for rec in records]
    word2idx, idx2word, _ = build_vocab(token_docs, max_vocab_size=10000, include_cls=False)

    pretrained_vectors, pretrained_word2idx = load_pretrained_embeddings(args.embedding_path, args.word2idx_path)
    embedding_matrix = prepare_embedding_matrix(
        local_word2idx=word2idx,
        pretrained_vectors=pretrained_vectors,
        pretrained_word2idx=pretrained_word2idx,
        embedding_dim=args.embedding_dim,
    )

    pos_label2idx = {t: i for i, t in enumerate(POS_TAGS)}
    ner_label2idx = {t: i for i, t in enumerate(NER_TAGS)}
    idx2pos = {v: k for k, v in pos_label2idx.items()}

    # POS: frozen and fine-tuned.
    pos_frozen = build_and_eval_pos(
        train_records,
        val_records,
        test_records,
        word2idx,
        pos_label2idx,
        embedding_matrix,
        freeze=True,
        bidirectional=True,
        dropout=0.5,
        epochs=args.epochs,
        args=args,
        device=device,
    )
    pos_finetuned = build_and_eval_pos(
        train_records,
        val_records,
        test_records,
        word2idx,
        pos_label2idx,
        embedding_matrix,
        freeze=False,
        bidirectional=True,
        dropout=0.5,
        epochs=args.epochs,
        args=args,
        device=device,
    )

    # Save best POS model (fine-tuned).
    torch.save(pos_finetuned["model"].state_dict(), models_dir / "bilstm_pos.pt")
    plot_history(pos_frozen["history"], reports_dir / "part2_pos_frozen_curve.png", "POS Frozen: Train/Val")
    plot_history(pos_finetuned["history"], reports_dir / "part2_pos_finetuned_curve.png", "POS Fine-tuned: Train/Val")

    g_flat = [x for seq in pos_finetuned["gold_ids"] for x in seq]
    p_flat = [x for seq in pos_finetuned["pred_ids"] for x in seq]
    pos_cm = confusion_matrix(g_flat, p_flat, labels=list(range(len(POS_TAGS)))).tolist()

    confused_pairs = get_confused_pairs_and_examples(
        records=test_records,
        gold_ids=pos_finetuned["gold_ids"],
        pred_ids=pos_finetuned["pred_ids"],
        idx2label=idx2pos,
        top_n=3,
    )

    # NER with and without CRF.
    ner_crf = build_and_eval_ner(
        train_records,
        val_records,
        test_records,
        word2idx,
        ner_label2idx,
        embedding_matrix,
        freeze=False,
        bidirectional=True,
        dropout=0.5,
        use_crf=True,
        epochs=args.epochs,
        args=args,
        device=device,
    )
    ner_softmax = build_and_eval_ner(
        train_records,
        val_records,
        test_records,
        word2idx,
        ner_label2idx,
        embedding_matrix,
        freeze=False,
        bidirectional=True,
        dropout=0.5,
        use_crf=False,
        epochs=args.epochs,
        args=args,
        device=device,
    )

    torch.save(ner_crf["model"].state_dict(), models_dir / "bilstm_ner.pt")
    plot_history(ner_crf["history"], reports_dir / "part2_ner_crf_curve.png", "NER CRF: Train/Val")
    plot_history(ner_softmax["history"], reports_dir / "part2_ner_softmax_curve.png", "NER Softmax: Train/Val")

    ner_errors = collect_ner_errors(test_records, ner_crf["gold_tags"], ner_crf["pred_tags"])

    # Ablations.
    ablations = {}
    if not args.skip_ablations:
        ablations = run_ablations(
            train_records=train_records,
            val_records=val_records,
            test_records=test_records,
            word2idx=word2idx,
            pretrained_matrix=embedding_matrix,
            pos_label2idx=pos_label2idx,
            ner_label2idx=ner_label2idx,
            args=args,
            device=device,
        )

    topic_dist = Counter(labels)
    pos_dist = Counter(tag for rec in records for tag in rec.pos_tags)
    ner_dist = Counter(tag for rec in records for tag in rec.ner_tags)

    summary = {
        "selection": {
            "total_selected_sentences": len(records),
            "topic_distribution": dict(topic_dist),
            "split_sizes": {
                "train": len(train_records),
                "val": len(val_records),
                "test": len(test_records),
            },
        },
        "label_distribution": {
            "pos": dict(pos_dist),
            "ner": dict(ner_dist),
        },
        "pos": {
            "frozen": {
                "accuracy": pos_frozen["accuracy"],
                "macro_f1": pos_frozen["macro_f1"],
            },
            "fine_tuned": {
                "accuracy": pos_finetuned["accuracy"],
                "macro_f1": pos_finetuned["macro_f1"],
            },
            "confusion_matrix": pos_cm,
            "most_confused_pairs": confused_pairs,
        },
        "ner": {
            "with_crf": ner_crf["entity_scores"],
            "without_crf": ner_softmax["entity_scores"],
            "error_analysis": ner_errors,
        },
        "ablations": ablations,
    }

    save_json(reports_dir / "part2_results.json", summary)
    print("Part 2 completed.")


if __name__ == "__main__":
    main()
