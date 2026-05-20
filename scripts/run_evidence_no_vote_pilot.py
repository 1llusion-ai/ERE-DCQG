"""Run a no-vote FairytaleQA evidence-label pilot.

Pipeline:
  1. Single full-context selector.
  2. Blind verifier using selected evidence only.
  3. Stateless leave-one-out checks with no voting.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dcqg.datasets.fairytaleqa_loader import load_fairytaleqa
from dcqg.path.no_vote_evidence import NoVoteEvidenceAuditor, _difficulty_from_ids
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
    parser.add_argument("--split", default="train",
                        help="Dataset split, or 'all' to load train+val+test")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--record_type", choices=["implicit", "explicit", "all"],
                        default="implicit",
                        help="Which record type to process")
    parser.add_argument("--implicit_limit", type=int, default=0,
                        help="Max implicit records (0=all)")
    parser.add_argument("--per_story_limit", type=int, default=0,
                        help="Max records per story (0=no limit, use with explicit)")
    parser.add_argument("--sample_mode", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_verification", action="store_true",
                        help="Skip blind verifier and removal checks; use selector output directly as labels.")
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


def select_per_story_records(records, per_story_limit, sample_mode, seed):
    """Select up to N records per story."""
    by_story = defaultdict(list)
    for r in records:
        by_story[r["story_name"]].append(r)

    rng = random.Random(seed)
    output = []
    for story in sorted(by_story.keys()):
        story_records = by_story[story]
        if per_story_limit and len(story_records) > per_story_limit:
            if sample_mode == "random":
                picked = rng.sample(story_records, per_story_limit)
            else:
                picked = story_records[:per_story_limit]
        else:
            picked = story_records
        for r in picked:
            output.append(dict(r))
    return output


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
                  sample_seed, skip_verification=False):
    selector_rows = read_jsonl(selector_path) if os.path.exists(selector_path) else []
    if skip_verification:
        labels = read_jsonl(labels_path) if os.path.exists(labels_path) else []
        verification_rows = []
    else:
        verification_rows = (
            read_jsonl(verification_path) if os.path.exists(verification_path) else []
        )
        labels = [r for r in verification_rows if r.get("difficulty_label") != "Invalid"]
        write_jsonl(labels_path, labels)

    summary = {
        "timestamp": _timestamp(),
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
        "selector_difficulty": dict(
            Counter(r.get("selector_difficulty", "Invalid") for r in selector_rows)
        ),
        "verification_status": dict(
            Counter(r.get("verification_status", "?") for r in verification_rows)
        ),
        "final_difficulty": dict(
            Counter(r.get("difficulty_label", "Invalid") for r in labels)
        ),
        "invalid_count": sum(
            1 for r in verification_rows if r.get("difficulty_label") == "Invalid"
        ),
        "api_calls": auditor.call_counts(),
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
    print(f"Final labels:     {summary['label_count']} {summary['final_difficulty']}")
    print(f"Invalid:          {summary['invalid_count']}")
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
    print(f"Record type:    {args.record_type}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Model:          {model}")
    print(f"Implicit limit: {args.implicit_limit}")
    print(f"Per-story limit:{args.per_story_limit}")
    print(f"Sample mode:    {args.sample_mode}")
    print(f"Sample seed:    {args.seed if args.sample_mode == 'random' else 'n/a'}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Timeout:        {args.timeout}s")
    print(f"Resume:         {args.resume}")
    print(f"Skip verify:    {args.skip_verification}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load records
    if args.split == "all":
        all_records = []
        for s in ["train", "validation", "test"]:
            all_records.extend(load_fairytaleqa(split=s, limit=None))
    else:
        all_records = load_fairytaleqa(split=args.split, limit=None)

    # Filter by record type
    if args.record_type == "implicit":
        records = [
            r for r in all_records
            if r.get("ex_or_im", "").lower() == "implicit"
        ]
        total_available = len(records)
        if args.implicit_limit:
            records, total_available = select_implicit_records(
                records, args.implicit_limit, args.sample_mode, args.seed
            )
    elif args.record_type == "explicit":
        records = [
            r for r in all_records
            if r.get("ex_or_im", "").lower() == "explicit"
        ]
        total_available = len(records)
        if args.per_story_limit:
            records = select_per_story_records(
                records, args.per_story_limit, args.sample_mode, args.seed
            )
    else:  # all
        records = all_records
        total_available = len(records)
        if args.implicit_limit:
            records, total_available = select_implicit_records(
                records, args.implicit_limit, args.sample_mode, args.seed
            )

    print(f"Loaded {args.record_type} records: {len(records)} (total available: {total_available})")

    auditor = NoVoteEvidenceAuditor(
        batch_size=args.batch_size,
        model=model,
        timeout=args.timeout,
    )

    selector_path = os.path.join(args.output_dir, "selector_candidates.jsonl")
    verification_path = os.path.join(args.output_dir, "blind_verification.jsonl")
    labels_path = os.path.join(args.output_dir, f"labels_{args.record_type}.jsonl")

    candidates = run_selector(
        records, auditor, selector_path, args.batch_size, args.resume
    )

    if args.skip_verification:
        print("Skipping blind verification; using selector output as labels.")
        labels = []
        for c in candidates:
            row = dict(c)
            if row.get("section_sufficient") == "yes" and row.get("selected_evidence_sentences"):
                sel = row["selected_evidence_sentences"]
                direct = row.get("answer_directly_found", "no")
                row["difficulty_label"] = _difficulty_from_ids(sel, direct)
                row["required_evidence_sentences"] = sel
                row["num_required_sentences"] = len(sel)
                row["final_answer_directly_found"] = direct
                row["final_reasoning_level"] = row.get("reasoning_level", "unknown")
                row["verification_status"] = "skipped"
            else:
                row["difficulty_label"] = "Invalid"
                row["verification_status"] = "selector_invalid_or_empty"
            labels.append(row)
        write_jsonl(labels_path, labels)
        valid = [r for r in labels if r.get("difficulty_label") != "Invalid"]
        print(f"Labels: {len(valid)} valid / {len(labels)} total")
        dist = Counter(r.get("difficulty_label") for r in labels)
        print(f"Distribution: {dict(dist)}")
    else:
        verified = run_verifier(candidates, auditor, verification_path, args.resume)
        if len(verified) != len(candidates):
            print(f"WARNING: verified {len(verified)} of {len(candidates)} candidates")

    write_summary(
        args.output_dir, selector_path, verification_path, labels_path,
        auditor, records, total_available, args.sample_mode, args.seed,
        skip_verification=args.skip_verification,
    )


if __name__ == "__main__":
    main()
