"""Select stratified test samples from the training dataset.

Usage:
    python -m scripts.select_test_samples \
        --data_path outputs/runs/evidence_audit_full/train_dataset.jsonl \
        --output_path outputs/eval/test_200.jsonl \
        --n_samples 200 --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dcqg.utils.jsonl import read_jsonl, write_jsonl


DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select stratified test samples from training dataset.",
    )
    p.add_argument(
        "--data_path",
        type=str,
        default="outputs/runs/evidence_audit_full/train_dataset.jsonl",
        help="Path to the full training JSONL.",
    )
    p.add_argument(
        "--output_path",
        type=str,
        default="outputs/eval/test_200.jsonl",
        help="Output path for selected test samples.",
    )
    p.add_argument("--n_samples", type=int, default=200, help="Total number of samples to select.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


def stratified_sample(
    records: list[dict],
    n_total: int,
    seed: int,
) -> list[dict]:
    """Sample n_total items stratified by difficulty (Easy/Medium/Hard).

    Allocates floor(n_total / 3) to the first two levels and the remainder
    to the third, e.g. 200 -> 67 + 67 + 66.
    """
    rng = random.Random(seed)

    by_level: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        diff = rec.get("difficulty_label") or rec.get("difficulty") or rec.get("label")
        if diff in DIFFICULTY_LEVELS:
            by_level[diff].append(rec)

    # Determine per-level quota: first two get ceil, last gets remainder
    base = n_total // len(DIFFICULTY_LEVELS)
    remainder = n_total - base * len(DIFFICULTY_LEVELS)
    quotas = {level: base for level in DIFFICULTY_LEVELS}
    # Distribute remainder across levels in order
    for i in range(remainder):
        quotas[DIFFICULTY_LEVELS[i]] += 1
    # e.g. n=200 -> Easy=67, Medium=67, Hard=66

    selected: list[dict] = []
    for level in DIFFICULTY_LEVELS:
        pool = by_level[level]
        quota = quotas[level]
        if len(pool) < quota:
            print(
                f"  WARNING: {level} has only {len(pool)} items, "
                f"requested {quota}. Taking all."
            )
            sampled = pool[:]
        else:
            sampled = rng.sample(pool, quota)
        selected.extend(sampled)
        print(f"  {level}: {len(sampled)} / {len(pool)} available")

    rng.shuffle(selected)
    return selected


def main() -> None:
    args = parse_args()

    print(f"Loading data from: {args.data_path}")
    records = read_jsonl(args.data_path)
    print(f"Loaded {len(records)} records")

    selected = stratified_sample(records, args.n_samples, args.seed)
    print(f"\nSelected {len(selected)} stratified samples (seed={args.seed})")

    write_jsonl(args.output_path, selected)
    print(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    main()
