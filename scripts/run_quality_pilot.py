"""Quality Pilot: N-item quality-first pilot with comprehensive filtering.

Runs N Easy + N Medium + N Hard through PathQG-HardAware generation
and the full quality filter pipeline.

Usage:
    python scripts/run_quality_pilot.py --n_per_level 10 --skip_llm_filters
"""
import argparse
import json
import time
import random
from pathlib import Path
from collections import defaultdict, Counter

from dcqg.path.answer_extraction import enrich_path_item, is_valid_final_event
from dcqg.generation.generator import generate_with_retry_hardaware
from dcqg.question_filter.pipeline import quality_filter_pipeline
from dcqg.utils.jsonl import read_jsonl, write_jsonl


def load_and_sample_paths(paths_file, n_per_level, seed):
    """Load sampled paths, enrich, filter invalid, select N/level."""
    all_paths = read_jsonl(paths_file)
    enriched = [enrich_path_item(p) for p in all_paths]

    valid = []
    for p in enriched:
        ok, _ = is_valid_final_event(p)
        if ok:
            valid.append(p)
    print(f"  Valid final events: {len(valid)}/{len(enriched)}")

    by_level = defaultdict(list)
    for p in valid:
        by_level[p["difficulty"]].append(p)

    random.seed(seed)
    sampled = []
    for level in ["Easy", "Medium", "Hard"]:
        pool = by_level[level]
        n = min(n_per_level, len(pool))
        selected = random.sample(pool, n)
        sampled.extend(selected)
        print(f"  {level}: sampled {n} from {len(pool)} valid paths")

    random.shuffle(sampled)
    return sampled


def generate_questions(sampled_items):
    results = []
    for i, item in enumerate(sampled_items):
        item["_item_id"] = i
        r, _ = generate_with_retry_hardaware(item, max_attempts=3)
        results.append(r)
        if (i + 1) % 10 == 0:
            n_pass = sum(1 for r in results if r.get("grammar_pass", False))
            print(f"  [{i+1}/{len(sampled_items)}] grammar_pass={n_pass}", flush=True)
        time.sleep(0.1)
    return results


def run_quality_filters(results, skip_llm=False):
    for i, r in enumerate(results):
        is_gen_error = r.get("generation_error", False)
        r = quality_filter_pipeline(r, skip_llm=(skip_llm or is_gen_error))
        if (i + 1) % 10 == 0:
            n_pass = sum(1 for r in results if r.get("final_filter_pass", False))
            print(f"  [{i+1}/{len(results)}] final_pass={n_pass}", flush=True)
        if not skip_llm and not is_gen_error:
            time.sleep(0.1)
    return results


def generate_report(results, output_dir):
    n_total = len(results)
    n_passed = sum(1 for r in results if r.get("final_filter_pass", False))
    overall_pass_rate = n_passed / n_total if n_total else 0

    by_level = defaultdict(list)
    for r in results:
        by_level[r["difficulty"]].append(r)

    level_stats = {}
    for level in ["Easy", "Medium", "Hard"]:
        items = by_level[level]
        n = len(items)
        p = sum(1 for r in items if r.get("final_filter_pass", False))
        level_stats[level] = {"total": n, "passed": p, "pass_rate": p / n if n else 0}

    grammar_fails = Counter()
    for r in results:
        if not r.get("grammar_pass", False):
            reason = r.get("grammar_reason", "unknown")
            cat = reason.split(":")[0].strip() if ":" in reason else reason
            grammar_fails[cat] += 1

    gen_error_count = sum(1 for r in results if r.get("generation_error", False))

    report = {
        "n_total": n_total,
        "n_passed": n_passed,
        "overall_pass_rate": round(overall_pass_rate, 4),
        "per_level": level_stats,
        "generation_error_count": gen_error_count,
        "grammar_failures": dict(grammar_fails.most_common()),
    }

    report_path = output_dir / "filter_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Report: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Run quality pilot.")
    parser.add_argument("--paths_file", default="outputs/runs/latest/paths.raw.jsonl")
    parser.add_argument("--output_dir", default="outputs/runs/quality_pilot")
    parser.add_argument("--n_per_level", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_llm_filters", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gen_path = output_dir / "generated_raw.jsonl"

    if args.skip_generation:
        print("Loading existing generated questions...")
        results = read_jsonl(gen_path)
        print(f"  Loaded {len(results)} items")
    else:
        print("Step 1: Sampling paths")
        sampled = load_and_sample_paths(args.paths_file, args.n_per_level, args.seed)
        print(f"  Total: {len(sampled)}")

        print("\nStep 2: Generating questions")
        results = generate_questions(sampled)
        write_jsonl(gen_path, results)

        grammar_pass = sum(1 for r in results if r.get("grammar_pass", False))
        print(f"  Grammar pass: {grammar_pass}/{len(results)}")

    print("\nStep 3: Quality filters")
    results = run_quality_filters(results, skip_llm=args.skip_llm_filters)

    print("\nStep 4: Saving results")
    write_jsonl(output_dir / "filtered_questions.jsonl", results)
    passed = [r for r in results if r.get("final_filter_pass", False)]
    write_jsonl(output_dir / "passed_questions.jsonl", passed)
    print(f"  {len(passed)}/{len(results)} passed")

    print("\nStep 5: Report")
    report = generate_report(results, output_dir)

    print(f"\nSummary:")
    print(f"  Pass rate: {report['overall_pass_rate']*100:.1f}%")
    for level in ["Easy", "Medium", "Hard"]:
        s = report["per_level"][level]
        print(f"  {level}: {s['passed']}/{s['total']} ({s['pass_rate']*100:.1f}%)")


if __name__ == "__main__":
    main()
