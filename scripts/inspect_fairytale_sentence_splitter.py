"""Inspect FairytaleQA story sentence splitter — old vs new comparison.

Samples *limit* implicit QA pairs from FairytaleQA *split* and writes a
side-by-side Markdown report of the old regex-based split and the new
speech-attribution-aware split.

Usage:
    python -m scripts.inspect_fairytale_sentence_splitter \
        --split train --limit 30 --seed 42 \
        --output outputs/debug/fairytale_sentence_splitter_30.md
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dcqg.datasets.fairytaleqa_loader import load_fairytaleqa
from dcqg.path.fairytale_evidence_audit import _split_story_sentences


def _old_split_sentences(text: str) -> list[str]:
    """Original regex-based sentence splitter (kept for comparison)."""
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if s.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect FairytaleQA sentence splitter (old vs new).",
    )
    parser.add_argument("--split", default="train", help="FairytaleQA split.")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/debug/fairytale_sentence_splitter_30.md")
    return parser.parse_args()


def render_sentence_block(sentences: list[str]) -> str:
    return "\n".join(f"[S{i}] {sent}" for i, sent in enumerate(sentences))


def main() -> None:
    args = parse_args()

    # Load
    all_records = load_fairytaleqa(split=args.split)
    implicit = [r for r in all_records if r.get("ex_or_im") == "implicit"]
    print(f"Loaded {len(all_records)} records ({len(implicit)} implicit) from {args.split}")

    # Sample
    rng = random.Random(args.seed)
    sample = implicit[:]  # shallow copy
    rng.shuffle(sample)
    sample = sample[: args.limit]
    print(f"Sampled {len(sample)} implicit QA pairs (seed={args.seed})")

    # Build Markdown
    lines: list[str] = []
    lines.append("# FairytaleQA Sentence Splitter Inspection")
    lines.append("")
    lines.append(f"**Split:** {args.split}  ")
    lines.append(f"**Limit:** {args.limit}  ")
    lines.append(f"**Seed:** {args.seed}  ")
    lines.append(f"**Filter:** implicit (`ex_or_im == 'implicit'`)  ")
    lines.append("")

    total_old = 0
    total_new = 0
    unchanged = 0
    changed = 0

    for idx, rec in enumerate(sample, start=1):
        story_section = rec.get("story_section", "")
        old_sents = _old_split_sentences(story_section)
        new_sents = _split_story_sentences(story_section)

        n_old = len(old_sents)
        n_new = len(new_sents)
        total_old += n_old
        total_new += n_new
        if n_old == n_new:
            unchanged += 1
        else:
            changed += 1

        lines.append(f"## Sample {idx}")
        lines.append("")
        lines.append(f"**story_name:** {rec.get('story_name', '')}  ")
        lines.append(f"**question:** {rec.get('question', '')}  ")
        lines.append(f"**answer:** {rec.get('answer1', '')}  ")
        lines.append("")
        lines.append(f"**old sentence count:** {n_old}  ")
        lines.append(f"**new sentence count:** {n_new}  ")
        lines.append("")

        lines.append("### Old Split")
        lines.append("")
        lines.append(render_sentence_block(old_sents))
        lines.append("")

        lines.append("### New Split")
        lines.append("")
        lines.append(render_sentence_block(new_sents))
        lines.append("")

    # Summary
    lines.insert(4, f"**Samples where count changed:** {changed} / {len(sample)}  ")
    lines.insert(5, f"**Samples unchanged:** {unchanged} / {len(sample)}  ")
    lines.insert(6, f"**Total old sentences:** {total_old}  ")
    lines.insert(7, f"**Total new sentences:** {total_new}  ")
    lines.insert(8, "")

    # Write
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote inspection report to {output_path}")
    print(f"Summary: {changed}/{len(sample)} samples changed, "
          f"{total_old} -> {total_new} sentences")


if __name__ == "__main__":
    main()
