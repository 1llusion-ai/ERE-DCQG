"""Run a no-vote FairytaleQA evidence-label pilot.

Pipeline:
  1. Single full-context selector.
  2. Blind verifier using selected evidence only.
  3. Default: annotation-assist output. Blind verifier is an auxiliary
     priority signal, not a hard filter. Legacy auto-label mode can still run
     leave-one-out checks.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dcqg.datasets.fairytaleqa_loader import load_fairytaleqa
from dcqg.path.no_vote_evidence import NoVoteEvidenceAuditor
from dcqg.utils.config import get_api_config
from dcqg.utils.jsonl import read_jsonl, write_jsonl


def _timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _reset(path):
    if os.path.exists(path):
        os.remove(path)


def _count(path):
    if not os.path.exists(path):
        return 0
    return len(read_jsonl(path))


def parse_args():
    parser = argparse.ArgumentParser(
        description="No-vote evidence selection + blind verification pilot"
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--implicit_limit", type=int, default=100)
    parser.add_argument("--sample_mode", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--selector_enable_thinking",
        action="store_true",
        help="Pass enable_thinking=true for selector API calls.",
    )
    parser.add_argument(
        "--verifier_disable_thinking",
        action="store_true",
        help="Pass enable_thinking=false for Blind/Removal verifier API calls.",
    )
    parser.add_argument(
        "--thinking_budget",
        type=int,
        default=None,
        help="Optional thinking_budget for selector API calls when thinking is enabled.",
    )
    parser.add_argument(
        "--selector_only",
        action="store_true",
        help="Run only the selector stage and skip Blind/Removal verification.",
    )
    parser.add_argument(
        "--mode",
        choices=["annotation_assist", "auto_labels"],
        default="annotation_assist",
        help=(
            "annotation_assist keeps selector-valid rows for human review and "
            "uses Blind Verifier only as priority; auto_labels reproduces the "
            "old blind-filter + removal-verifier behavior."
        ),
    )
    return parser.parse_args()


def select_implicit_records(records, limit, sample_mode, seed):
    """Select implicit records while keeping sampling metadata in each row."""
    selected = list(enumerate(records))
    total = len(selected)

    if limit and total > limit:
        if sample_mode == "random":
            rng = random.Random(seed)
            selected = rng.sample(selected, limit)
        else:
            selected = selected[:limit]

    output = []
    for original_index, record in selected:
        item = dict(record)
        item["sample_original_index"] = original_index
        item["sample_mode"] = sample_mode
        item["sample_seed"] = seed if sample_mode == "random" else None
        output.append(item)
    return output, total


def run_selector(records, auditor, output_path, batch_size, resume):
    start = _count(output_path) if resume else 0
    if not resume:
        _reset(output_path)
    if start >= len(records):
        print(f"Selector: already complete ({start}/{len(records)})")
        return read_jsonl(output_path)

    print(f"Selector: {len(records)} items, batch_size={batch_size}, start={start}")
    for batch_start in range(start, len(records), batch_size):
        batch = records[batch_start:batch_start + batch_size]
        t0 = time.time()
        candidates = auditor.select_batch(batch)
        elapsed = time.time() - t0
        for candidate in candidates:
            _append_jsonl(output_path, candidate)
        dist = Counter(c.get("selector_difficulty", "Invalid") for c in candidates)
        print(
            f"  Selector: {batch_start + len(batch)}/{len(records)} "
            f"{dict(dist)} ({elapsed:.1f}s)"
        )
        time.sleep(0.2)
    return read_jsonl(output_path)


def run_verifier(candidates, auditor, output_path, resume):
    start = _count(output_path) if resume else 0
    if not resume:
        _reset(output_path)
    if start >= len(candidates):
        print(f"Verifier: already complete ({start}/{len(candidates)})")
        return read_jsonl(output_path)

    print(f"Verifier: {len(candidates)} items, start={start}")
    for idx, candidate in enumerate(candidates[start:], start=start):
        t0 = time.time()
        result = auditor.verify_candidate(candidate)
        elapsed = time.time() - t0
        _append_jsonl(output_path, result)
        if (idx + 1) % 10 == 0 or idx + 1 == len(candidates):
            print(
                f"  Verifier: {idx + 1}/{len(candidates)} "
                f"label={result.get('difficulty_label')} "
                f"status={result.get('verification_status')} "
                f"({elapsed:.1f}s)"
            )
        time.sleep(0.1)
    return read_jsonl(output_path)


def write_summary(output_dir, selector_path, verification_path, labels_path,
                  auditor, records, total_implicit_records, sample_mode,
                  sample_seed, mode, selector_only=False):
    selector_rows = read_jsonl(selector_path) if os.path.exists(selector_path) else []
    verification_rows = (
        read_jsonl(verification_path) if os.path.exists(verification_path) else []
    )
    if selector_only:
        labels = []
    elif mode == "annotation_assist":
        labels = [
            r for r in verification_rows
            if r.get("annotation_priority") != "discard"
            and r.get("suggested_difficulty_label", r.get("difficulty_label"))
            != "Invalid"
        ]
    else:
        labels = [
            r for r in verification_rows
            if r.get("difficulty_label") != "Invalid"
        ]
    if selector_only:
        if os.path.exists(labels_path):
            os.remove(labels_path)
    else:
        write_jsonl(labels_path, labels)

    api_calls = auditor.call_counts()
    selector_raws = {
        row.get("selector_raw", "")
        for row in selector_rows
        if row.get("selector_raw")
    }
    inferred_selector_calls = len(selector_raws)
    inferred_sufficiency_calls = sum(
        1 for row in verification_rows if row.get("sufficiency_raw")
    )
    inferred_removal_calls = sum(
        len(row.get("removal_checks") or []) for row in verification_rows
    )
    api_calls = {
        "selector_calls": max(
            api_calls.get("selector_calls", 0), inferred_selector_calls
        ),
        "sufficiency_calls": max(
            api_calls.get("sufficiency_calls", 0), inferred_sufficiency_calls
        ),
        "removal_calls": max(
            api_calls.get("removal_calls", 0), inferred_removal_calls
        ),
    }
    api_calls["total_calls"] = sum(api_calls.values())

    summary = {
        "timestamp": _timestamp(),
        "mode": mode,
        "selector_only": selector_only,
        "output_dir": output_dir,
        "num_input_records": len(records),
        "total_implicit_records": total_implicit_records,
        "sample_mode": sample_mode,
        "sample_seed": sample_seed if sample_mode == "random" else None,
        "selector_path": selector_path,
        "verification_path": verification_path,
        "labels_path": labels_path,
        "selector_count": len(selector_rows),
        "verified_count": len(verification_rows),
        "label_count": len(labels),
        "annotation_priority": dict(
            Counter(r.get("annotation_priority", "?") for r in verification_rows)
        ),
        "selector_difficulty": dict(
            Counter(r.get("selector_difficulty", "Invalid") for r in selector_rows)
        ),
        "blind_sufficiency": dict(
            Counter(r.get("evidence_set_sufficient", "?") for r in verification_rows)
        ),
        "verification_status": dict(
            Counter(r.get("verification_status", "?") for r in verification_rows)
        ),
        "final_difficulty": dict(
            Counter(
                r.get("suggested_difficulty_label", r.get("difficulty_label", "Invalid"))
                for r in labels
            )
        ),
        "invalid_count": (
            sum(
                1 for r in selector_rows
                if r.get("selector_difficulty", "Invalid") == "Invalid"
            )
            if selector_only else
            sum(
                1 for r in verification_rows
                if r.get("annotation_priority") == "discard"
                or r.get("difficulty_label") == "Invalid"
            )
        ),
        "thinking_config": {
            "selector_enable_thinking": auditor.selector_enable_thinking,
            "verifier_enable_thinking": auditor.verifier_enable_thinking,
            "thinking_budget": auditor.thinking_budget,
        },
        "api_calls": api_calls,
    }

    summary_path = os.path.join(output_dir, "pipeline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("NO-VOTE PILOT SUMMARY")
    print("=" * 70)
    print(f"Output dir:       {output_dir}")
    print(f"Input records:    {summary['num_input_records']}")
    print(f"Selector dist:    {summary['selector_difficulty']}")
    print(f"Statuses:         {summary['verification_status']}")
    label_name = (
        "Annotation rows" if mode == "annotation_assist" else "Final labels"
    )
    print(f"{label_name}: {summary['label_count']} {summary['final_difficulty']}")
    print(f"Priority:         {summary['annotation_priority']}")
    print(f"Invalid:          {summary['invalid_count']}")
    print(f"Thinking config:  {summary['thinking_config']}")
    print(f"API calls:        {summary['api_calls']}")
    print(f"Summary saved:    {summary_path}")
    print("=" * 70)
    return summary_path


def main():
    args = parse_args()
    cfg = get_api_config()
    model = args.model or cfg.get("JUDGE_MODEL", "Qwen/Qwen2.5-32B-Instruct")

    print("=" * 70)
    print("No-Vote Evidence Pilot")
    print("=" * 70)
    print(f"Split:          {args.split}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Model:          {model}")
    print(f"Implicit limit: {args.implicit_limit}")
    print(f"Sample mode:    {args.sample_mode}")
    print(f"Sample seed:    {args.seed if args.sample_mode == 'random' else 'n/a'}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Timeout:        {args.timeout}s")
    print(f"Resume:         {args.resume}")
    print(f"Mode:           {args.mode}")
    print(f"Selector only:  {args.selector_only}")
    print(
        "Thinking:       "
        f"selector={True if args.selector_enable_thinking else None}, "
        f"verifier={False if args.verifier_disable_thinking else None}, "
        f"budget={args.thinking_budget}"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    all_records = load_fairytaleqa(split=args.split, limit=None)
    implicit = [
        record for record in all_records
        if record.get("ex_or_im", "").lower() == "implicit"
    ]
    implicit, total_implicit_records = select_implicit_records(
        implicit,
        args.implicit_limit,
        args.sample_mode,
        args.seed,
    )
    print(
        f"Loaded implicit records: {len(implicit)} "
        f"(total available: {total_implicit_records})"
    )

    auditor = NoVoteEvidenceAuditor(
        batch_size=args.batch_size,
        model=model,
        timeout=args.timeout,
        use_removal_verifier=(args.mode == "auto_labels"),
        blind_filter=(args.mode == "auto_labels"),
        selector_enable_thinking=True if args.selector_enable_thinking else None,
        verifier_enable_thinking=False if args.verifier_disable_thinking else None,
        thinking_budget=args.thinking_budget,
    )

    selector_path = os.path.join(args.output_dir, "selector_candidates.jsonl")
    verification_path = os.path.join(args.output_dir, "blind_verification.jsonl")
    labels_name = (
        "annotation_candidates.jsonl"
        if args.mode == "annotation_assist"
        else "labels_implicit.jsonl"
    )
    labels_path = os.path.join(args.output_dir, labels_name)

    candidates = run_selector(
        implicit, auditor, selector_path, args.batch_size, args.resume
    )
    if args.selector_only:
        if not args.resume and os.path.exists(verification_path):
            os.remove(verification_path)
    else:
        verified = run_verifier(candidates, auditor, verification_path, args.resume)
        if len(verified) != len(candidates):
            print(f"WARNING: verified {len(verified)} of {len(candidates)} candidates")

    write_summary(
        args.output_dir, selector_path, verification_path, labels_path,
        auditor, implicit, total_implicit_records, args.sample_mode, args.seed,
        args.mode, selector_only=args.selector_only
    )


if __name__ == "__main__":
    main()
