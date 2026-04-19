from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd):
    print("Running:", " ".join(str(x) for x in cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {' '.join(str(x) for x in cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all Assignment 2 parts")
    parser.add_argument("--cleaned", type=Path, default=Path("cleaned.txt"))
    parser.add_argument("--raw", type=Path, default=Path("raw.txt"))
    parser.add_argument("--metadata", type=Path, default=Path("Metadata.json"))
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    py = sys.executable

    part1_cmd = [
        py,
        str(Path("scripts") / "part1_embeddings.py"),
        "--cleaned",
        str(args.cleaned),
        "--raw",
        str(args.raw),
        "--metadata",
        str(args.metadata),
        "--output-root",
        str(args.output_root),
        "--device",
        args.device,
    ]
    if args.quick:
        part1_cmd.extend(["--w2v-epochs", "5", "--ppmi-vocab", "1500"])

    part2_cmd = [
        py,
        str(Path("scripts") / "part2_sequence_labeling.py"),
        "--cleaned",
        str(args.cleaned),
        "--metadata",
        str(args.metadata),
        "--output-root",
        str(args.output_root),
        "--embedding-path",
        str(args.output_root / "embeddings" / "embeddings_w2v.npy"),
        "--word2idx-path",
        str(args.output_root / "embeddings" / "word2idx.json"),
        "--device",
        args.device,
    ]
    if args.quick:
        part2_cmd.extend(["--epochs", "6", "--ablation-epochs", "3"])

    part3_cmd = [
        py,
        str(Path("scripts") / "part3_transformer_classifier.py"),
        "--cleaned",
        str(args.cleaned),
        "--metadata",
        str(args.metadata),
        "--output-root",
        str(args.output_root),
        "--device",
        args.device,
    ]
    if args.quick:
        part3_cmd.extend(["--epochs", "6"])

    run_step(part1_cmd)
    run_step(part2_cmd)
    run_step(part3_cmd)

    print("All parts finished.")


if __name__ == "__main__":
    main()
