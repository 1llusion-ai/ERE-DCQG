"""Convert evidence-review CSV files into Label Studio JSON tasks.

Usage:
    python -m scripts.export_label_studio_tasks \
        --input outputs/runs/no_vote_full_implicit_storysplit_qwen3_32b_think_fewshot_v2/human_review_1968_full_implicit.csv \
        --output outputs/runs/no_vote_full_implicit_storysplit_qwen3_32b_think_fewshot_v2/label_studio_tasks.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATA_FIELDS = [
    "sample_id",
    "annotation_priority",
    "story_name",
    "source_index",
    "attribute",
    "fairytale_scope",
    "fairytale_explicitness",
    "suggested_difficulty_label",
    "suggested_answer_directly_found",
    "required_evidence_ids",
    "num_required_sentences",
    "blind_sufficient",
    "blind_reason",
    "selector_difficulty",
    "selector_evidence_ids",
    "selector_answer_directly_found",
    "selector_reason",
    "full_context_qa",
    "full_context_numbered",
    "full_context_zh",
    "evidence_context",
    "qa",
    "evidence_context_zh",
    "qa_zh",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert evidence human-review CSV into Label Studio tasks.",
    )
    parser.add_argument("--input", required=True, help="Input human-review CSV.")
    parser.add_argument("--output", required=True, help="Output Label Studio JSON.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _fmt_model_suggestion(row: dict[str, str]) -> str:
    parts = [
        f"sample_id: {row.get('sample_id', '')}",
        f"story_name: {row.get('story_name', '')}",
        f"priority: {row.get('annotation_priority', '')}",
        f"suggested difficulty: {row.get('suggested_difficulty_label', '')}",
        f"suggested direct answer: {row.get('suggested_answer_directly_found', '')}",
        f"suggested evidence ids: {row.get('required_evidence_ids', '')}",
        f"blind sufficient: {row.get('blind_sufficient', '')}",
    ]
    reason = row.get("selector_reason") or row.get("blind_reason")
    if reason:
        parts.append(f"model reason: {reason}")
    return "\n".join(parts)


def to_task(row: dict[str, str]) -> dict:
    data = {field: row.get(field, "") for field in DATA_FIELDS}
    data["model_suggestion"] = _fmt_model_suggestion(row)
    return {"data": data}


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = read_csv(input_path)
    tasks = [to_task(row) for row in rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"Loaded {len(rows)} rows from {input_path}")
    print(f"Wrote {len(tasks)} Label Studio tasks to {output_path}")


if __name__ == "__main__":
    main()
