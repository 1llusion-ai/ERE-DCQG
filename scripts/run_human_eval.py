"""Prepare human evaluation samples from reranked and K=1 outputs.

Samples items stratified by difficulty and method, outputs a blinded CSV
for human annotators (no source column).

Usage:
    python -m scripts.run_human_eval \
        --reranked_path outputs/runs/reranking_eval/reranked.jsonl \
        --k1_path outputs/runs/reranking_eval/k1.jsonl \
        --output_path outputs/eval/human_eval_samples.csv \
        --n_reranked 50 --n_k1 50 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dcqg.utils.jsonl import read_jsonl

DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]


# ═══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare human evaluation samples (blinded).",
    )
    p.add_argument(
        "--reranked_path",
        type=str,
        default="outputs/runs/reranking_eval/reranked.jsonl",
        help="Path to reranked output JSONL.",
    )
    p.add_argument(
        "--k1_path",
        type=str,
        default="outputs/runs/reranking_eval/k1.jsonl",
        help="Path to K=1 (no reranking) output JSONL.",
    )
    p.add_argument(
        "--output_path",
        type=str,
        default="outputs/eval/human_eval_samples.csv",
        help="Output CSV for human evaluation.",
    )
    p.add_argument("--n_reranked", type=int, default=50, help="Number of reranked samples.")
    p.add_argument("--n_k1", type=int, default=50, help="Number of K=1 samples.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Stratified sampling
# ═══════════════════════════════════════════════════════════════════════════════

def stratified_sample(
    records: list[dict],
    n_total: int,
    seed: int,
) -> list[dict]:
    """Sample n_total items stratified by difficulty level."""
    rng = random.Random(seed)

    by_level: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        diff = (
            rec.get("difficulty_target")
            or rec.get("difficulty_label")
            or rec.get("difficulty")
            or "Medium"
        )
        if diff in DIFFICULTY_LEVELS:
            by_level[diff].append(rec)
        else:
            by_level["Medium"].append(rec)

    # Allocate quota per level
    base = n_total // len(DIFFICULTY_LEVELS)
    remainder = n_total - base * len(DIFFICULTY_LEVELS)
    quotas = {level: base for level in DIFFICULTY_LEVELS}
    for i in range(remainder):
        quotas[DIFFICULTY_LEVELS[i]] += 1

    selected: list[dict] = []
    for level in DIFFICULTY_LEVELS:
        pool = by_level[level]
        quota = quotas[level]
        if len(pool) < quota:
            print(f"  WARNING: {level} has only {len(pool)} items, requested {quota}.")
            sampled = pool[:]
        else:
            sampled = rng.sample(pool, quota)
        selected.extend(sampled)
        print(f"    {level}: {len(sampled)} / {len(pool)}")

    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    # Load data
    print(f"Loading reranked: {args.reranked_path}")
    reranked_records = read_jsonl(args.reranked_path)
    print(f"  {len(reranked_records)} records")

    print(f"Loading K=1: {args.k1_path}")
    k1_records = read_jsonl(args.k1_path)
    print(f"  {len(k1_records)} records")

    # Stratified sampling
    print(f"\nSampling {args.n_reranked} reranked items:")
    reranked_samples = stratified_sample(reranked_records, args.n_reranked, args.seed)

    print(f"\nSampling {args.n_k1} K=1 items:")
    k1_samples = stratified_sample(k1_records, args.n_k1, args.seed + 1)

    # Tag source internally (for metadata, NOT included in output CSV)
    metadata: list[dict] = []
    all_samples: list[dict] = []

    for rec in reranked_samples:
        rec["_source"] = "reranked"
        all_samples.append(rec)

    for rec in k1_samples:
        rec["_source"] = "k1"
        all_samples.append(rec)

    # Shuffle to blind ordering
    rng.shuffle(all_samples)

    # Assign sample IDs
    for i, rec in enumerate(all_samples):
        rec["_sample_id"] = f"S{i + 1:03d}"

    # Build CSV rows (blinded: no source column)
    csv_rows: list[dict] = []
    for rec in all_samples:
        story = (
            rec.get("story_section")
            or rec.get("context")
            or rec.get("story", "")
        )
        question = rec.get("question") or rec.get("generated_question", "")
        answer = (
            rec.get("answer")
            or rec.get("gold_answer_phrase")
            or rec.get("gold_answer_trigger", "")
        )
        difficulty_target = (
            rec.get("difficulty_target")
            or rec.get("difficulty_label")
            or rec.get("difficulty", "")
        )

        csv_rows.append({
            "sample_id": rec["_sample_id"],
            "story": story,
            "question": question,
            "answer": answer,
            "difficulty_target": difficulty_target,
        })

        metadata.append({
            "sample_id": rec["_sample_id"],
            "source": rec["_source"],
            "story_name": rec.get("story_name") or rec.get("doc_id", ""),
            "difficulty_target": difficulty_target,
        })

    # Write CSV
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["sample_id", "story", "question", "answer", "difficulty_target"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nSaved blinded CSV: {output_path}")
    print(f"  {len(csv_rows)} rows ({args.n_reranked} reranked + {args.n_k1} K=1)")

    # Write metadata (internal, for later unblinding)
    meta_path = output_path.parent / "human_eval_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved metadata (internal): {meta_path}")


if __name__ == "__main__":
    main()
