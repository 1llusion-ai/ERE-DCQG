"""Export evidence-label JSONL into a human-review CSV.

Usage:
    python -m scripts.export_evidence_labels_for_review \
        --input outputs/runs/no_vote_500_newdef_qwen3_32b/annotation_candidates.jsonl \
        --output outputs/runs/no_vote_500_newdef_qwen3_32b/human_review_100.csv \
        --n 100 --sample_mode balanced --seed 42
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

from dcqg.path.fairytale_evidence_audit import _split_story_sentences
from dcqg.utils.jsonl import read_jsonl


DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]
BACK_COLUMNS = ["evidence_context", "qa"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export verified evidence labels for human review.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL, usually annotation_candidates.jsonl.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of verified examples to export.",
    )
    parser.add_argument(
        "--sample_mode",
        choices=["balanced", "proportional", "first"],
        default="balanced",
        help="Sampling mode over difficulty labels.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include_invalid",
        action="store_true",
        help="Also allow discard/invalid examples. Default excludes them.",
    )
    parser.add_argument(
        "--priority",
        choices=["all", "high", "repair"],
        default="all",
        help="Which annotation-priority rows to export.",
    )
    return parser.parse_args()


def _safe_ids(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    ids: list[int] = []
    for item in value:
        try:
            ids.append(int(item))
        except (TypeError, ValueError):
            continue
    return ids


def _fmt_sentence_block(sentences: list[str], ids: list[int]) -> str:
    lines = []
    for sid in ids:
        if 0 <= sid < len(sentences):
            lines.append(f"[S{sid}] {sentences[sid]}")
    return "\n".join(lines)


def _fmt_full_context(sentences: list[str]) -> str:
    return "\n".join(f"[S{i}] {sent}" for i, sent in enumerate(sentences))


def _fmt_full_context_qa(sentences: list[str], question: str, answer: str) -> str:
    return f"Full context:\n{_fmt_full_context(sentences)}\n\nQA:\nQ: {question}\nA: {answer}"


def _fmt_removal_summary(checks: object) -> str:
    if not isinstance(checks, list):
        return ""
    parts = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        sid = check.get("sentence_id", "")
        decision = check.get("decision", "")
        can_still = check.get("can_still_answer", "")
        reason = check.get("reasoning", "")
        parts.append(
            f"S{sid}: decision={decision}, can_still_answer={can_still}, reason={reason}"
        )
    return "\n".join(parts)


def _difficulty_counts(records: list[dict]) -> dict[str, int]:
    counts = {level: 0 for level in DIFFICULTY_LEVELS}
    for rec in records:
        label = _suggested_difficulty(rec)
        if label in counts:
            counts[label] += 1
    return counts


def _map_difficulty(sentence_ids: list[int], answer_directly_found: object) -> str:
    direct = ""
    if isinstance(answer_directly_found, str):
        direct = answer_directly_found.strip().lower()
    if direct not in {"yes", "no"}:
        direct = "yes" if len(sentence_ids) == 1 else "no"

    if len(sentence_ids) <= 0:
        return "Invalid"
    if direct == "yes" and len(sentence_ids) == 1:
        return "Easy"
    if direct == "no" and len(sentence_ids) == 1:
        return "Medium"
    if direct == "yes" and len(sentence_ids) >= 2:
        return "Medium"
    if direct == "no" and len(sentence_ids) >= 2:
        return "Hard"
    return "Invalid"


def _annotation_priority(rec: dict) -> str:
    priority = rec.get("annotation_priority")
    if priority in {"high", "repair", "discard"}:
        return priority
    if rec.get("selector_status") != "ok":
        return "discard"
    if rec.get("section_sufficient") != "yes":
        return "discard"
    if not _safe_ids(rec.get("selected_evidence_sentences")):
        return "discard"
    if rec.get("evidence_set_sufficient") == "yes":
        return "high"
    return "repair"


def _suggested_evidence_ids(rec: dict) -> list[int]:
    selected = _safe_ids(rec.get("selected_evidence_sentences"))
    if selected:
        return selected
    return _safe_ids(rec.get("required_evidence_sentences"))


def _suggested_directness(rec: dict) -> str:
    if rec.get("evidence_set_sufficient") == "yes":
        value = rec.get("verified_answer_directly_found")
        if isinstance(value, str) and value.strip().lower() in {"yes", "no"}:
            return value.strip().lower()
    value = rec.get("final_answer_directly_found") or rec.get("answer_directly_found")
    if isinstance(value, str) and value.strip().lower() in {"yes", "no"}:
        return value.strip().lower()
    return ""


def _suggested_difficulty(rec: dict) -> str:
    for key in ("suggested_difficulty_label", "difficulty_label", "selector_difficulty"):
        value = rec.get(key)
        if value in DIFFICULTY_LEVELS:
            return value
    return _map_difficulty(_suggested_evidence_ids(rec), _suggested_directness(rec))


def _balanced_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_level: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        label = _suggested_difficulty(rec)
        if label in DIFFICULTY_LEVELS:
            by_level[label].append(rec)

    base = n // len(DIFFICULTY_LEVELS)
    remainder = n - base * len(DIFFICULTY_LEVELS)
    quotas = {level: base for level in DIFFICULTY_LEVELS}
    for level in DIFFICULTY_LEVELS[:remainder]:
        quotas[level] += 1

    selected: list[dict] = []
    leftovers: list[dict] = []
    for level in DIFFICULTY_LEVELS:
        pool = by_level[level][:]
        rng.shuffle(pool)
        quota = quotas[level]
        selected.extend(pool[:quota])
        leftovers.extend(pool[quota:])

    if len(selected) < n:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: n - len(selected)])

    rng.shuffle(selected)
    return selected[:n]


def sample_records(records: list[dict], n: int, sample_mode: str, seed: int) -> list[dict]:
    if sample_mode == "first":
        return records[:n]
    if sample_mode == "proportional":
        rng = random.Random(seed)
        pool = records[:]
        rng.shuffle(pool)
        return pool[:n]
    return _balanced_sample(records, n, seed)


def to_review_row(rec: dict, sample_id: str, source_index: int) -> dict:
    sentences = _split_story_sentences(rec.get("story_section", ""))
    required_ids = _suggested_evidence_ids(rec)
    selector_ids = _safe_ids(rec.get("selected_evidence_sentences"))
    question = rec.get("question", "")
    answer = rec.get("answer", "")
    priority = _annotation_priority(rec)
    suggested_direct = _suggested_directness(rec)
    suggested_label = _suggested_difficulty(rec)

    return {
        "sample_id": sample_id,
        "annotation_priority": priority,
        "suggested_difficulty_label": suggested_label,
        "full_context_qa": _fmt_full_context_qa(sentences, question, answer),
        "required_evidence_ids": json.dumps(required_ids),
        "evidence_context": _fmt_sentence_block(sentences, required_ids),
        "qa": f"Q: {question}\nA: {answer}",
        "human_valid": "",
        "human_difficulty_label": "",
        "human_answer_directly_found": "",
        "human_evidence_ids": "",
        "human_notes": "",
        "num_required_sentences": len(required_ids),
        "suggested_answer_directly_found": suggested_direct,
        "blind_sufficient": rec.get("evidence_set_sufficient", ""),
        "blind_reason": rec.get("sufficiency_reason", ""),
        "story_name": rec.get("story_name", ""),
        "source_index": source_index,
        "attribute": rec.get("attribute", ""),
        "fairytale_scope": rec.get("local_or_sum", ""),
        "fairytale_explicitness": rec.get("ex_or_im", ""),
        "full_context_numbered": _fmt_full_context(sentences),
        "removal_summary": _fmt_removal_summary(rec.get("removal_checks")),
        "selector_difficulty": rec.get("selector_difficulty", ""),
        "selector_evidence_ids": json.dumps(selector_ids),
        "selector_answer_directly_found": rec.get("answer_directly_found", ""),
        "selector_reason": rec.get("evidence_reason", ""),
    }


def export_review_csv(
    input_path: str | Path,
    output_path: str | Path,
    n: int = 100,
    sample_mode: str = "balanced",
    seed: int = 42,
    include_invalid: bool = False,
    priority: str = "all",
) -> list[dict]:
    records = read_jsonl(input_path)
    indexed = list(enumerate(records))
    if not include_invalid:
        indexed = [
            (idx, rec)
            for idx, rec in indexed
            if _annotation_priority(rec) != "discard"
            and _suggested_difficulty(rec) in DIFFICULTY_LEVELS
        ]
    if priority != "all":
        indexed = [
            (idx, rec)
            for idx, rec in indexed
            if _annotation_priority(rec) == priority
        ]

    sampled_records = sample_records([rec for _, rec in indexed], n, sample_mode, seed)
    id_by_object = {id(rec): idx for idx, rec in indexed}

    rows = [
        to_review_row(rec, f"EV{row_idx:03d}", id_by_object.get(id(rec), -1))
        for row_idx, rec in enumerate(sampled_records, start=1)
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    back = [field for field in BACK_COLUMNS if field in fieldnames]
    fieldnames = [field for field in fieldnames if field not in back] + back
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    verified = [
        rec for rec in records
        if _annotation_priority(rec) != "discard"
        and _suggested_difficulty(rec) in DIFFICULTY_LEVELS
    ]
    print(f"Loaded {len(records)} records from {args.input}")
    print(f"Usable annotation records: {len(verified)}")
    print(f"Suggested distribution: {_difficulty_counts(verified)}")

    rows = export_review_csv(
        input_path=args.input,
        output_path=args.output,
        n=args.n,
        sample_mode=args.sample_mode,
        seed=args.seed,
        include_invalid=args.include_invalid,
        priority=args.priority,
    )
    print(f"Exported {len(rows)} rows to {args.output}")
    print(f"Exported distribution: {_difficulty_counts(rows)}")


if __name__ == "__main__":
    main()
