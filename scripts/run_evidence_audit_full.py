"""Full 3-stage evidence audit pipeline orchestrator.

Orchestrates:
  Stage A: Run FairytaleEvidenceAuditor N times on implicit items
  Stage B: Run counterfactual verification on union of evidence sentences
  Stage C: Aggregate with self-consistency -> labels_implicit.jsonl
  Explicit: Audit explicit items -> labels_explicit.jsonl
  Merge: labels_implicit + labels_explicit -> train_dataset.jsonl

Usage:
    python -m scripts.run_evidence_audit_full --split train --output_dir outputs/runs/evidence_audit_full/
    python -m scripts.run_evidence_audit_full --split train --stage_a_runs r1.jsonl r2.jsonl r3.jsonl --skip_stage_b
    python -m scripts.run_evidence_audit_full --split train --resume --implicit_limit 2166
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dcqg.datasets.fairytaleqa_loader import load_fairytaleqa
from dcqg.path.fairytale_evidence_audit import FairytaleEvidenceAuditor, classify_difficulty
from dcqg.path.counterfactual_verify import verify_candidate
from dcqg.utils.jsonl import read_jsonl, write_jsonl
from dcqg.utils.config import get_api_config


# ── Helpers ──────────────────────────────────────────────────────────

def _timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_existing_jsonl(path):
    """Load JSONL if it exists, else return empty list."""
    if os.path.exists(path):
        return read_jsonl(path)
    return []


def _count_done(path):
    """Count lines in a JSONL file without loading all into memory."""
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# ── Stage A: Evidence audit runs ─────────────────────────────────────

def run_stage_a(records, output_dir, n_runs, batch_size, model, resume,
                existing_run_paths, timeout=300):
    """Run FairytaleEvidenceAuditor n_runs times on the given records.

    Args:
        records: list of QA pair dicts to audit.
        output_dir: directory for stage_a_run_{i}.jsonl files.
        n_runs: total number of runs.
        batch_size: batch size per LLM call.
        model: LLM model string.
        resume: if True, skip runs whose output already has the right count.
        existing_run_paths: list of paths provided via --stage_a_runs.
            If path i is provided and non-empty, skip run i.

    Returns:
        list of paths to the n_runs JSONL files.
    """
    run_paths = []

    for run_i in range(1, n_runs + 1):
        # Check if user provided an existing run for this index
        ext_idx = run_i - 1
        if ext_idx < len(existing_run_paths) and existing_run_paths[ext_idx]:
            ext_path = existing_run_paths[ext_idx]
            if os.path.exists(ext_path) and _count_done(ext_path) > 0:
                print(f"Stage A run {run_i}: using existing file {ext_path} "
                      f"({_count_done(ext_path)} items)")
                run_paths.append(ext_path)
                continue

        run_path = os.path.join(output_dir, f"stage_a_run_{run_i}.jsonl")

        # Resume check
        if resume and os.path.exists(run_path):
            existing_count = _count_done(run_path)
            if existing_count >= len(records):
                print(f"Stage A run {run_i}: already complete "
                      f"({existing_count}/{len(records)} items)")
                run_paths.append(run_path)
                continue
            else:
                print(f"Stage A run {run_i}: resuming from {existing_count}/{len(records)}")
        else:
            existing_count = 0

        print(f"\nStage A run {run_i}/{n_runs}: auditing {len(records)} items "
              f"(batch_size={batch_size}, model={model})")

        auditor = FairytaleEvidenceAuditor(
            batch_size=batch_size,
            model=model,
            timeout=timeout,
        )

        mode = "a" if (resume and existing_count > 0) else "w"
        out_f = open(run_path, mode, encoding="utf-8")

        # Determine which items to skip when resuming
        start_idx = existing_count if (resume and existing_count > 0) else 0
        remaining = records[start_idx:]

        try:
            for batch_start in range(0, len(remaining), batch_size):
                batch = remaining[batch_start:batch_start + batch_size]
                abs_start = start_idx + batch_start
                abs_end = abs_start + len(batch)

                t0 = time.time()
                try:
                    candidates = auditor.audit_batch(batch)
                except Exception as e:
                    print(f"  Stage A run {run_i}: batch error at "
                          f"{abs_start}-{abs_end}: {e}")
                    # Write error placeholders so count stays aligned
                    for rec in batch:
                        error_item = {
                            "story_name": rec.get("story_name", ""),
                            "question": rec.get("question", ""),
                            "assessment_status": "batch_error",
                            "evidence_difficulty": "Easy",
                            "error": str(e),
                        }
                        out_f.write(json.dumps(error_item, ensure_ascii=False) + "\n")
                    out_f.flush()
                    continue
                elapsed = time.time() - t0

                for c in candidates:
                    out_f.write(json.dumps(c, ensure_ascii=False) + "\n")
                out_f.flush()

                diff_counts = Counter(c.get("evidence_difficulty", "?")
                                      for c in candidates)
                print(f"  Stage A run {run_i}: {abs_end}/{len(records)} items "
                      f"(E={diff_counts.get('Easy', 0)} "
                      f"M={diff_counts.get('Medium', 0)} "
                      f"H={diff_counts.get('Hard', 0)}) "
                      f"{elapsed:.1f}s")

                time.sleep(0.2)
        finally:
            out_f.close()

        run_paths.append(run_path)
        print(f"Stage A run {run_i}: saved to {run_path}")

    return run_paths


# ── Stage B: Counterfactual verification ─────────────────────────────

def run_stage_b(stage_a_paths, output_dir, model, resume):
    """Run counterfactual verification on Stage A results.

    Loads all Stage A runs, takes the union of evidence candidates,
    and runs counterfactual checks on each.

    Returns path to stage_b_verification.jsonl.
    """
    output_path = os.path.join(output_dir, "stage_b_verification.jsonl")

    # Load all Stage A results: use the first run as the base, since
    # all runs audit the same items in the same order
    base_candidates = read_jsonl(stage_a_paths[0])
    print(f"\nStage B: verifying {len(base_candidates)} items "
          f"from {len(stage_a_paths)} Stage A runs")

    cfg = get_api_config()
    api_url = cfg["SILICONFLOW_API_URL"]
    api_key = cfg["SILICONFLOW_API_KEY"]
    verify_model = model or cfg["JUDGE_MODEL"]

    # Resume check
    existing_count = 0
    if resume and os.path.exists(output_path):
        existing_count = _count_done(output_path)
        if existing_count >= len(base_candidates):
            print(f"Stage B: already complete ({existing_count} items)")
            return output_path
        print(f"Stage B: resuming from {existing_count}/{len(base_candidates)}")

    mode = "a" if (resume and existing_count > 0) else "w"
    out_f = open(output_path, mode, encoding="utf-8")
    start_idx = existing_count if (resume and existing_count > 0) else 0

    try:
        for i, candidate in enumerate(base_candidates[start_idx:], start=start_idx):
            t0 = time.time()
            try:
                verification = verify_candidate(
                    candidate,
                    api_url=api_url,
                    api_key=api_key,
                    model=verify_model,
                    n_runs=3,
                )
            except Exception as e:
                verification = {
                    "verified_evidence_sentences": candidate.get(
                        "required_evidence_sentences", []),
                    "dropped_evidence_sentences": [],
                    "verification_details": [],
                    "verified_num_required": candidate.get(
                        "num_required_sentences", 0),
                    "verified_difficulty": candidate.get(
                        "evidence_difficulty", "Easy"),
                    "verification_error": str(e),
                }
            elapsed = time.time() - t0

            # Merge candidate + verification
            result = dict(candidate)
            result.update(verification)
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

            if (i + 1) % 20 == 0 or (i + 1) == len(base_candidates):
                out_f.flush()
                print(f"  Stage B: {i + 1}/{len(base_candidates)} verified "
                      f"({elapsed:.1f}s/item)")

            time.sleep(0.3)
    finally:
        out_f.close()

    print(f"Stage B: saved to {output_path}")
    return output_path


# ── Stage C: Self-consistency aggregation ────────────────────────────

def run_stage_c(stage_a_paths, stage_b_path, output_dir, skip_stage_b):
    """Aggregate Stage A (+ optional Stage B) into labels_implicit.jsonl.

    Uses majority voting across Stage A runs. If Stage B was run,
    the verified difficulty overrides disagreements.

    Returns path to labels_implicit.jsonl.
    """
    output_path = os.path.join(output_dir, "labels_implicit.jsonl")

    # Try importing the self_consistency module; fall back to inline logic
    try:
        from dcqg.path.self_consistency import (
            aggregate_audit_runs, build_explicit_labels, merge_label_files,
        )
        has_self_consistency = True
    except ImportError:
        has_self_consistency = False

    if has_self_consistency:
        n_runs = len(stage_a_paths)
        threshold = max(1, (n_runs // 2) + 1)  # majority: 1→1, 2→2, 3→2
        print(f"\nStage C: aggregating with dcqg.path.self_consistency "
              f"(n_runs={n_runs}, threshold={threshold})")
        summary = aggregate_audit_runs(
            stage_a_paths,
            stage_b_path=stage_b_path if not skip_stage_b else None,
            output_path=output_path,
            agreement_threshold=threshold,
        )
        print(f"Stage C: saved {summary['total']} items to {output_path}")
        print(f"  Coverage: {summary['coverage']:.1%}")
        print(f"  Difficulty: {summary['difficulty_distribution']}")
        return output_path

    # Inline self-consistency: majority vote on evidence_difficulty
    print("\nStage C: aggregating with inline majority vote "
          "(dcqg.path.self_consistency not available)")

    all_runs = [read_jsonl(p) for p in stage_a_paths]
    n_items = len(all_runs[0])
    n_runs = len(all_runs)

    # Load Stage B results if available
    stage_b_items = {}
    if not skip_stage_b and stage_b_path and os.path.exists(stage_b_path):
        for item in read_jsonl(stage_b_path):
            key = (item.get("story_name", ""), item.get("question", ""))
            stage_b_items[key] = item

    aggregated = []
    for idx in range(n_items):
        # Collect votes from all runs
        votes = []
        items_for_idx = []
        for run in all_runs:
            if idx < len(run):
                item = run[idx]
                votes.append(item.get("evidence_difficulty", "Easy"))
                items_for_idx.append(item)

        if not items_for_idx:
            continue

        # Majority vote
        vote_counts = Counter(votes)
        majority_diff, majority_count = vote_counts.most_common(1)[0]
        agreement = majority_count / len(votes) if votes else 0.0

        # Use the first run's item as the base record
        base = dict(items_for_idx[0])

        # Check Stage B override
        key = (base.get("story_name", ""), base.get("question", ""))
        if key in stage_b_items:
            b_item = stage_b_items[key]
            verified_diff = b_item.get("verified_difficulty")
            if verified_diff:
                base["verified_difficulty"] = verified_diff
                base["verified_evidence_sentences"] = b_item.get(
                    "verified_evidence_sentences", [])
                base["dropped_evidence_sentences"] = b_item.get(
                    "dropped_evidence_sentences", [])
                # Use verified difficulty as the final label
                majority_diff = verified_diff

        base["final_difficulty"] = majority_diff
        base["difficulty_votes"] = dict(vote_counts)
        base["difficulty_agreement"] = round(agreement, 3)
        base["n_audit_runs"] = n_runs

        aggregated.append(base)

    write_jsonl(output_path, aggregated)
    print(f"Stage C: saved {len(aggregated)} items to {output_path}")
    return output_path


# ── Explicit items ───────────────────────────────────────────────────

def run_explicit_audit(records, output_dir, batch_size, model, resume, timeout=300):
    """Audit explicit items and produce labels_explicit.jsonl.

    Explicit items are typically Easy (answer directly stated in text),
    so we run a single audit pass.

    Returns path to labels_explicit.jsonl.
    """
    output_path = os.path.join(output_dir, "labels_explicit.jsonl")

    # Try the dedicated helper first
    try:
        from dcqg.path.self_consistency import build_explicit_labels
        has_helper = True
    except ImportError:
        has_helper = False

    if has_helper:
        print(f"\nExplicit audit: using build_explicit_labels on {len(records)} items")
        summary = build_explicit_labels(records, output_path)
        print(f"Explicit audit: saved {summary['total']} items to {output_path}")
        return output_path

    # Fallback: single audit pass
    print(f"\nExplicit audit: auditing {len(records)} items "
          f"(batch_size={batch_size}, model={model})")

    # Resume check
    existing_count = 0
    if resume and os.path.exists(output_path):
        existing_count = _count_done(output_path)
        if existing_count >= len(records):
            print(f"Explicit audit: already complete ({existing_count} items)")
            return output_path
        print(f"Explicit audit: resuming from {existing_count}/{len(records)}")

    auditor = FairytaleEvidenceAuditor(
        batch_size=batch_size,
        model=model,
        timeout=timeout,
    )

    mode = "a" if (resume and existing_count > 0) else "w"
    out_f = open(output_path, mode, encoding="utf-8")
    start_idx = existing_count if (resume and existing_count > 0) else 0
    remaining = records[start_idx:]

    try:
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start:batch_start + batch_size]
            abs_end = start_idx + batch_start + len(batch)

            t0 = time.time()
            try:
                candidates = auditor.audit_batch(batch)
            except Exception as e:
                print(f"  Explicit audit: batch error at {abs_end}: {e}")
                for rec in batch:
                    error_item = {
                        "story_name": rec.get("story_name", ""),
                        "question": rec.get("question", ""),
                        "assessment_status": "batch_error",
                        "evidence_difficulty": "Easy",
                        "final_difficulty": "Easy",
                        "ex_or_im": "explicit",
                        "error": str(e),
                    }
                    out_f.write(json.dumps(error_item, ensure_ascii=False) + "\n")
                out_f.flush()
                continue
            elapsed = time.time() - t0

            for c in candidates:
                c["final_difficulty"] = c.get("evidence_difficulty", "Easy")
                out_f.write(json.dumps(c, ensure_ascii=False) + "\n")
            out_f.flush()

            print(f"  Explicit audit: {abs_end}/{len(records)} items "
                  f"({elapsed:.1f}s)")

            time.sleep(0.2)
    finally:
        out_f.close()

    print(f"Explicit audit: saved to {output_path}")
    return output_path


# ── Merge ────────────────────────────────────────────────────────────

def merge_labels(implicit_path, explicit_path, output_dir):
    """Merge implicit and explicit labels into train_dataset.jsonl.

    Returns path to the merged file.
    """
    output_path = os.path.join(output_dir, "train_dataset.jsonl")

    # Try the dedicated helper first
    try:
        from dcqg.path.self_consistency import merge_label_files
        has_helper = True
    except ImportError:
        has_helper = False

    if has_helper:
        print(f"\nMerge: using merge_label_files")
        summary = merge_label_files(implicit_path, explicit_path, output_path)
        print(f"Merge: saved {summary['total']} items to {output_path}")
        return output_path

    # Inline merge
    implicit = read_jsonl(implicit_path) if os.path.exists(implicit_path) else []
    explicit = read_jsonl(explicit_path) if os.path.exists(explicit_path) else []

    merged = implicit + explicit
    write_jsonl(output_path, merged)
    print(f"\nMerge: {len(implicit)} implicit + {len(explicit)} explicit "
          f"= {len(merged)} total -> {output_path}")
    return output_path


# ── Summary report ───────────────────────────────────────────────────

def write_summary(output_dir, implicit_path, explicit_path, merged_path,
                  stage_a_paths, stage_b_path, skip_stage_b, n_runs):
    """Print and save pipeline summary."""
    summary = {
        "timestamp": _timestamp(),
        "output_dir": output_dir,
        "stage_a_runs": len(stage_a_paths),
        "stage_a_paths": stage_a_paths,
        "stage_b_path": stage_b_path if not skip_stage_b else None,
        "skip_stage_b": skip_stage_b,
        "implicit_path": implicit_path,
        "explicit_path": explicit_path,
        "merged_path": merged_path,
    }

    # Count items and difficulty distributions
    for label, path in [("implicit", implicit_path),
                        ("explicit", explicit_path),
                        ("merged", merged_path)]:
        if os.path.exists(path):
            items = read_jsonl(path)
            diff_key = "final_difficulty" if any(
                "final_difficulty" in it for it in items[:5]
            ) else "evidence_difficulty"
            dists = Counter(it.get(diff_key, "Easy") for it in items)
            summary[f"{label}_count"] = len(items)
            summary[f"{label}_difficulty"] = dict(dists)
        else:
            summary[f"{label}_count"] = 0
            summary[f"{label}_difficulty"] = {}

    # Agreement stats from implicit labels
    if os.path.exists(implicit_path):
        implicit_items = read_jsonl(implicit_path)
        agreements = [it.get("difficulty_agreement", 1.0) for it in implicit_items]
        if agreements:
            summary["implicit_mean_agreement"] = round(
                sum(agreements) / len(agreements), 3)
            summary["implicit_unanimous"] = sum(
                1 for a in agreements if a >= 1.0)
            summary["implicit_majority"] = sum(
                1 for a in agreements if 0.5 < a < 1.0)
            summary["implicit_split"] = sum(
                1 for a in agreements if a <= 0.5)

    # Save summary JSON
    summary_path = os.path.join(output_dir, "pipeline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Timestamp:      {summary['timestamp']}")
    print(f"Output dir:     {output_dir}")
    print(f"Stage A runs:   {summary['stage_a_runs']}")
    print(f"Stage B:        {'skipped' if skip_stage_b else 'completed'}")
    print()
    print(f"Implicit items: {summary.get('implicit_count', 0)}")
    if summary.get("implicit_difficulty"):
        for diff in ["Easy", "Medium", "Hard"]:
            cnt = summary["implicit_difficulty"].get(diff, 0)
            total = summary.get("implicit_count", 1)
            pct = 100 * cnt / total if total else 0
            print(f"  {diff:8s}: {cnt:5d} ({pct:5.1f}%)")
    if summary.get("implicit_mean_agreement") is not None:
        print(f"  Agreement:  mean={summary['implicit_mean_agreement']:.3f}, "
              f"unanimous={summary.get('implicit_unanimous', 0)}, "
              f"majority={summary.get('implicit_majority', 0)}, "
              f"split={summary.get('implicit_split', 0)}")
    print()
    print(f"Explicit items: {summary.get('explicit_count', 0)}")
    if summary.get("explicit_difficulty"):
        for diff in ["Easy", "Medium", "Hard"]:
            cnt = summary["explicit_difficulty"].get(diff, 0)
            total = summary.get("explicit_count", 1)
            pct = 100 * cnt / total if total else 0
            print(f"  {diff:8s}: {cnt:5d} ({pct:5.1f}%)")
    print()
    print(f"Merged total:   {summary.get('merged_count', 0)}")
    if summary.get("merged_difficulty"):
        for diff in ["Easy", "Medium", "Hard"]:
            cnt = summary["merged_difficulty"].get(diff, 0)
            total = summary.get("merged_count", 1)
            pct = 100 * cnt / total if total else 0
            print(f"  {diff:8s}: {cnt:5d} ({pct:5.1f}%)")
    print()
    print(f"Summary saved:  {summary_path}")
    print("=" * 70)

    return summary_path


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Full 3-stage evidence audit pipeline orchestrator")
    p.add_argument("--split", default="train",
                   help="FairytaleQA split (default: train)")
    p.add_argument("--output_dir", default="outputs/runs/evidence_audit_full/",
                   help="Output directory")
    p.add_argument("--implicit_limit", type=int, default=2166,
                   help="Max implicit items to audit (default: 2166)")
    p.add_argument("--explicit_limit", type=int, default=2000,
                   help="Max explicit items (default: 2000)")
    p.add_argument("--batch_size", type=int, default=10,
                   help="Audit batch size (default: 10)")
    p.add_argument("--model", default=None,
                   help="LLM model for audit (default: JUDGE_MODEL from .env)")
    p.add_argument("--stage_a_runs", nargs="*", default=[],
                   help="Paths to existing Stage A JSONL files (skip re-running)")
    p.add_argument("--skip_stage_b", action="store_true",
                   help="Skip counterfactual verification (use Stage A labels)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from where we left off")
    p.add_argument("--n_runs_stage_a", type=int, default=3,
                   help="Number of Stage A audit runs (default: 3)")
    p.add_argument("--timeout", type=int, default=300,
                   help="LLM call timeout in seconds (default: 300)")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    cfg = get_api_config()
    model = args.model or cfg.get("JUDGE_MODEL", "Qwen/Qwen2.5-32B-Instruct")

    print("=" * 70)
    print("Full Evidence Audit Pipeline")
    print("=" * 70)
    print(f"Split:            {args.split}")
    print(f"Output dir:       {args.output_dir}")
    print(f"Model:            {model}")
    print(f"Implicit limit:   {args.implicit_limit}")
    print(f"Explicit limit:   {args.explicit_limit}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Stage A runs:     {args.n_runs_stage_a}")
    print(f"Timeout:          {args.timeout}s")
    print(f"Skip Stage B:     {args.skip_stage_b}")
    print(f"Resume:           {args.resume}")
    if args.stage_a_runs:
        print(f"Existing runs:    {args.stage_a_runs}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load dataset ──
    print("Loading FairytaleQA dataset...")
    try:
        all_records = load_fairytaleqa(split=args.split, limit=None)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Loaded {len(all_records)} total QA pairs from split={args.split}")

    # Split into implicit and explicit
    implicit_records = [r for r in all_records
                        if r.get("ex_or_im", "").lower() == "implicit"]
    explicit_records = [r for r in all_records
                        if r.get("ex_or_im", "").lower() == "explicit"]
    print(f"  Implicit: {len(implicit_records)}")
    print(f"  Explicit: {len(explicit_records)}")

    # Apply limits
    if args.implicit_limit and len(implicit_records) > args.implicit_limit:
        implicit_records = implicit_records[:args.implicit_limit]
    if args.explicit_limit and len(explicit_records) > args.explicit_limit:
        explicit_records = explicit_records[:args.explicit_limit]
    print(f"After limits: implicit={len(implicit_records)}, "
          f"explicit={len(explicit_records)}")
    print()

    # ── Stage A: Evidence audit runs on implicit items ──
    print("-" * 70)
    print("STAGE A: Evidence Audit (implicit items)")
    print("-" * 70)

    stage_a_paths = run_stage_a(
        records=implicit_records,
        output_dir=args.output_dir,
        n_runs=args.n_runs_stage_a,
        batch_size=args.batch_size,
        model=model,
        resume=args.resume,
        existing_run_paths=args.stage_a_runs,
        timeout=args.timeout,
    )

    # ── Stage B: Counterfactual verification ──
    stage_b_path = None
    if not args.skip_stage_b:
        print()
        print("-" * 70)
        print("STAGE B: Counterfactual Verification")
        print("-" * 70)

        stage_b_path = run_stage_b(
            stage_a_paths=stage_a_paths,
            output_dir=args.output_dir,
            model=model,
            resume=args.resume,
        )
    else:
        print("\nStage B: SKIPPED (--skip_stage_b)")

    # ── Stage C: Self-consistency aggregation ──
    print()
    print("-" * 70)
    print("STAGE C: Self-Consistency Aggregation")
    print("-" * 70)

    implicit_path = run_stage_c(
        stage_a_paths=stage_a_paths,
        stage_b_path=stage_b_path,
        output_dir=args.output_dir,
        skip_stage_b=args.skip_stage_b,
    )

    # ── Explicit items ──
    print()
    print("-" * 70)
    print("EXPLICIT ITEMS AUDIT")
    print("-" * 70)

    explicit_path = run_explicit_audit(
        records=explicit_records,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        model=model,
        resume=args.resume,
        timeout=args.timeout,
    )

    # ── Merge ──
    print()
    print("-" * 70)
    print("MERGE")
    print("-" * 70)

    merged_path = merge_labels(
        implicit_path=implicit_path,
        explicit_path=explicit_path,
        output_dir=args.output_dir,
    )

    # ── Summary ──
    write_summary(
        output_dir=args.output_dir,
        implicit_path=implicit_path,
        explicit_path=explicit_path,
        merged_path=merged_path,
        stage_a_paths=stage_a_paths,
        stage_b_path=stage_b_path,
        skip_stage_b=args.skip_stage_b,
        n_runs=args.n_runs_stage_a,
    )


if __name__ == "__main__":
    main()
