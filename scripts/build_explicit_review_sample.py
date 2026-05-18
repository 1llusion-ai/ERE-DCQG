"""Build a section-stratified FairytaleQA explicit review sample.

The explicit split is much larger than the implicit split and is dominated by
direct lookup questions.  This script samples explicit QA pairs for human review
while avoiding many near-duplicate rows from the same story section.

Usage:
    python -m scripts.build_explicit_review_sample \
        --split train \
        --output_dir outputs/runs/explicit_600_section_stratified_seed42 \
        --n 600 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dcqg.datasets.fairytaleqa_loader import load_fairytaleqa
from dcqg.path.self_consistency import build_explicit_labels
from dcqg.utils.jsonl import write_jsonl
from scripts.export_evidence_labels_for_review import export_review_csv
from scripts.export_label_studio_tasks import read_csv, to_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an explicit FairytaleQA review sample.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_per_story",
        type=int,
        default=5,
        help="Maximum sampled explicit rows per story_name.",
    )
    return parser.parse_args()


def _largest_remainder_quotas(counts: Counter[str], n: int) -> dict[str, int]:
    total = sum(counts.values())
    if total <= 0 or n <= 0:
        return {}

    raw = {
        key: (value * n / total)
        for key, value in counts.items()
    }
    quotas = {key: int(value) for key, value in raw.items()}
    remaining = n - sum(quotas.values())
    order = sorted(
        raw,
        key=lambda key: (raw[key] - quotas[key], counts[key], key),
        reverse=True,
    )
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def _section_key(record: dict) -> tuple[str, str]:
    return (record.get("story_name", ""), record.get("story_section", ""))


def sample_explicit_records(
    records: list[dict],
    n: int,
    seed: int,
    max_per_story: int,
) -> list[dict]:
    """Sample explicit rows with attribute stratification and section caps."""
    rng = random.Random(seed)
    explicit = [
        dict(record, sample_original_index=idx, sample_mode="section_stratified",
             sample_seed=seed)
        for idx, record in enumerate(records)
        if record.get("ex_or_im", "").lower() == "explicit"
    ]
    if n >= len(explicit):
        return explicit

    quotas = _largest_remainder_quotas(
        Counter(record.get("attribute", "") for record in explicit),
        n,
    )
    by_attribute: dict[str, list[dict]] = {}
    for record in explicit:
        by_attribute.setdefault(record.get("attribute", ""), []).append(record)
    for pool in by_attribute.values():
        rng.shuffle(pool)

    selected: list[dict] = []
    used_sections: set[tuple[str, str]] = set()
    story_counts: Counter[str] = Counter()

    def can_take(record: dict, relax_story_cap: bool = False) -> bool:
        section = _section_key(record)
        story = record.get("story_name", "")
        if section in used_sections:
            return False
        if not relax_story_cap and story_counts[story] >= max_per_story:
            return False
        return True

    def take(record: dict) -> None:
        selected.append(record)
        used_sections.add(_section_key(record))
        story_counts[record.get("story_name", "")] += 1

    for attribute, quota in quotas.items():
        pool = by_attribute.get(attribute, [])
        taken = 0
        for record in pool:
            if taken >= quota:
                break
            if can_take(record):
                take(record)
                taken += 1

    if len(selected) < n:
        leftovers = [
            record for record in explicit
            if can_take(record)
        ]
        rng.shuffle(leftovers)
        for record in leftovers:
            if len(selected) >= n:
                break
            take(record)

    if len(selected) < n:
        relaxed_leftovers = [
            record for record in explicit
            if can_take(record, relax_story_cap=True)
        ]
        rng.shuffle(relaxed_leftovers)
        for record in relaxed_leftovers:
            if len(selected) >= n:
                break
            take(record)

    rng.shuffle(selected)
    return selected[:n]


def _write_label_studio_tasks(csv_path: Path, output_path: Path) -> None:
    rows = read_csv(csv_path)
    tasks = [to_task(row) for row in rows]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)


def _write_summary(path: Path, records: list[dict], labels_path: Path,
                   csv_path: Path, tasks_path: Path, total_explicit: int) -> None:
    summary = {
        "total_explicit_records": total_explicit,
        "sample_count": len(records),
        "unique_stories": len({record.get("story_name", "") for record in records}),
        "unique_sections": len({_section_key(record) for record in records}),
        "attribute_distribution": dict(Counter(record.get("attribute", "") for record in records)),
        "scope_distribution": dict(Counter(record.get("local_or_sum", "") for record in records)),
        "top_story_counts": Counter(
            record.get("story_name", "") for record in records
        ).most_common(20),
        "labels_path": str(labels_path),
        "review_csv_path": str(csv_path),
        "label_studio_tasks_path": str(tasks_path),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = load_fairytaleqa(split=args.split, limit=None)
    total_explicit = sum(
        1 for record in all_records
        if record.get("ex_or_im", "").lower() == "explicit"
    )
    sampled = sample_explicit_records(
        all_records,
        n=args.n,
        seed=args.seed,
        max_per_story=args.max_per_story,
    )

    sample_records_path = output_dir / f"explicit_sample_{len(sampled)}_records.jsonl"
    labels_path = output_dir / f"labels_explicit_sample_{len(sampled)}.jsonl"
    review_csv_path = output_dir / f"human_review_{len(sampled)}_explicit.csv"
    tasks_path = output_dir / f"label_studio_tasks_{len(sampled)}_explicit.json"
    summary_path = output_dir / "sample_summary.json"

    write_jsonl(sample_records_path, sampled)
    build_explicit_labels(sampled, labels_path)
    export_review_csv(
        labels_path,
        review_csv_path,
        n=len(sampled),
        sample_mode="first",
        seed=args.seed,
    )
    _write_label_studio_tasks(review_csv_path, tasks_path)
    _write_summary(
        summary_path,
        sampled,
        labels_path,
        review_csv_path,
        tasks_path,
        total_explicit,
    )

    with review_csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded explicit records: {total_explicit}")
    print(f"Sampled records: {len(sampled)}")
    print(f"Unique stories: {len({r.get('story_name', '') for r in sampled})}")
    print(f"Unique sections: {len({_section_key(r) for r in sampled})}")
    print(f"Attribute dist: {dict(Counter(r.get('attribute', '') for r in sampled))}")
    print(f"Scope dist: {dict(Counter(r.get('local_or_sum', '') for r in sampled))}")
    print(f"Review rows: {len(rows)}")
    print(f"Labels: {labels_path}")
    print(f"Review CSV: {review_csv_path}")
    print(f"Label Studio tasks: {tasks_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
