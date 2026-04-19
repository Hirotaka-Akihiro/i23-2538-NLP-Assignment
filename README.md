# CS-4063 NLP Assignment 2

This repository contains a from-scratch PyTorch implementation for:
- Part 1: TF-IDF, PPMI, Skip-gram Word2Vec
- Part 2: BiLSTM POS/NER (with CRF + Viterbi for NER)
- Part 3: Custom Transformer encoder for 5-class topic classification

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Required input files

Place these files in the project root:
- cleaned.txt
- raw.txt
- Metadata.json

## Run full assignment

```bash
python scripts/run_assignment.py --output-root . --device cpu
```

Quick mode for faster smoke run:

```bash
python scripts/run_assignment.py --output-root . --device cpu --quick
```

## Run individual parts

```bash
python scripts/part1_embeddings.py --cleaned cleaned.txt --raw raw.txt --metadata Metadata.json --output-root .
python scripts/part2_sequence_labeling.py --cleaned cleaned.txt --metadata Metadata.json --output-root . --embedding-path embeddings/embeddings_w2v.npy --word2idx-path embeddings/word2idx.json
python scripts/part3_transformer_classifier.py --cleaned cleaned.txt --metadata Metadata.json --output-root .
```

## Temporary scripts

Generate temporary synthetic data for smoke testing:

```bash
python temp/temp_generate_synthetic_data.py
```

Temporary self-grader:

```bash
python temp/temp_self_grade.py --output-root . --save reports/self_grade.json
```

## Expected outputs

- embeddings/
  - tfidf_matrix.npy
  - ppmi_matrix.npy
  - embeddings_w2v.npy
  - word2idx.json
- models/
  - bilstm_pos.pt
  - bilstm_ner.pt
  - transformer_cls.pt
- data/
  - pos_train.conll
  - pos_test.conll
  - ner_train.conll
  - ner_test.conll
- reports/
  - JSON summaries and plot artifacts for all parts
