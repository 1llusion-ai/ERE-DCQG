"""
Quality Pilot: 90-item quality-first pilot with comprehensive filtering.

Runs 30 Easy + 30 Medium + 30 Hard through PathQG-HardAware generation
and the full quality filter pipeline.

Usage:
    python event_qg/src/quality_pilot.py [--skip_generation] [--skip_llm_filters]
"""
import json
import time
import random
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent))

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in open(env_path):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from compare_hardaware import (
    generate_with_retry_hardaware,
    prompt_pathqg_easy,
    prompt_pathqg_medium,
    prompt_pathqg_hard,
    fmt_ctx,
)
from quality_filter import quality_filter_pipeline

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

N_PER_LEVEL = 30
SEED = 42
OUTPUT_DIR = Path("event_qg/outputs/quality_pilot_90_v2")
PATHS_FILE = Path("event_qg/outputs/sampled_paths_preview.jsonl")


def load_and_sample_paths():
    """Load sampled paths and select 30 per level with seed=42."""
    with open(PATHS_FILE, encoding="utf-8") as f:
        all_paths = [json.loads(line) for line in f]

    by_level = defaultdict(list)
    for p in all_paths:
        by_level[p["difficulty"]].append(p)

    random.seed(SEED)
    sampled = []
    for level in ["Easy", "Medium", "Hard"]:
        pool = by_level[level]
        n = min(N_PER_LEVEL, len(pool))
        selected = random.sample(pool, n)
        sampled.extend(selected)
        print(f"  {level}: sampled {n} from {len(pool)} paths")

    random.shuffle(sampled)
    return sampled


def generate_questions(sampled_items):
    """Generate questions using PathQG-HardAware with retry."""
    results = []
    for i, item in enumerate(sampled_items):
        item["_item_id"] = i
        r, attempts = generate_with_retry_hardaware(item, max_attempts=3)
        results.append(r)

        if (i + 1) % 10 == 0:
            n_pass = sum(1 for r in results if r.get("grammar_pass", False))
            print(f"  [{i+1}/{len(sampled_items)}] generated, grammar_pass={n_pass}", flush=True)

        time.sleep(0.1)

    return results


def run_quality_filters(results, skip_llm=False):
    """Run all quality filters on generated results.
    Skips LLM-based filters for generation_error items.
    """
    for i, r in enumerate(results):
        # Skip LLM filters for generation_error items
        is_gen_error = r.get("generation_error", False)
        r = quality_filter_pipeline(r, skip_llm=(skip_llm or is_gen_error))

        if (i + 1) % 10 == 0:
            n_pass = sum(1 for r in results if r.get("final_filter_pass", False))
            print(f"  [{i+1}/{len(results)}] filtered, final_pass={n_pass}", flush=True)

        if not skip_llm and not is_gen_error:
            time.sleep(0.1)

    return results


def save_results(results):
    """Save all results to output files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # All 90 items with full filter fields
    all_path = OUTPUT_DIR / "filtered_questions.jsonl"
    with open(all_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(results)} items to {all_path}")

    # Only passed items
    passed = [r for r in results if r.get("final_filter_pass", False)]
    passed_path = OUTPUT_DIR / "passed_questions.jsonl"
    with open(passed_path, "w", encoding="utf-8") as f:
        for r in passed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(passed)} passed items to {passed_path}")


def generate_report(results):
    """Generate filter_report.json and filter_report.md."""
    n_total = len(results)
    passed = [r for r in results if r.get("final_filter_pass", False)]
    n_passed = len(passed)

    by_level = defaultdict(list)
    for r in results:
        by_level[r["difficulty"]].append(r)

    # ── Overall ──
    overall_pass_rate = n_passed / n_total if n_total else 0

    # ── Per-level pass rate ──
    level_stats = {}
    for level in ["Easy", "Medium", "Hard"]:
        items = by_level[level]
        n = len(items)
        p = sum(1 for r in items if r.get("final_filter_pass", False))
        level_stats[level] = {
            "total": n,
            "passed": p,
            "pass_rate": p / n if n else 0,
        }

    # ── Grammar failure distribution ──
    grammar_fails = Counter()
    grammar_fail_examples = {}
    for r in results:
        if not r.get("grammar_pass", False):
            reason = r.get("grammar_reason", "unknown")
            # Normalize reason category
            cat = reason.split(":")[0].strip() if ":" in reason else reason
            grammar_fails[cat] += 1
            if cat not in grammar_fail_examples:
                grammar_fail_examples[cat] = []
            if len(grammar_fail_examples[cat]) < 3:
                grammar_fail_examples[cat].append({
                    "question": r.get("generated_question", ""),
                    "reason": reason,
                    "difficulty": r.get("difficulty", ""),
                })

    # ── Weak trigger distribution ──
    wt_dist = Counter()
    wt_fail_examples = []
    for r in results:
        wt_type = r.get("weak_trigger_type", "none")
        wt_dist[wt_type] += 1
        if not r.get("weak_trigger_pass", False) and len(wt_fail_examples) < 5:
            wt_fail_examples.append({
                "question": r.get("generated_question", ""),
                "trigger": r.get("gold_answer_trigger", ""),
                "type": wt_type,
                "reason": r.get("weak_trigger_reason", ""),
            })

    # ── Answer phrase pass rate ──
    ap_pass = sum(1 for r in results if r.get("answer_phrase_pass", False))
    ap_rate = ap_pass / n_total if n_total else 0

    # ── Answer consistency ──
    ac_dist = Counter()
    ac_fail_examples = []
    for r in results:
        label = r.get("answer_consistency_label", "skipped")
        ac_dist[label] += 1
        if label == "no" and len(ac_fail_examples) < 5:
            ac_fail_examples.append({
                "question": r.get("generated_question", ""),
                "difficulty": r.get("difficulty", ""),
                "gold_phrase": r.get("gold_answer_phrase", ""),
                "expected": r.get("expected_answer_summary", ""),
                "reason": r.get("answer_consistency_reason", ""),
            })

    # ── Path coverage ──
    pc_by_level = {}
    pc_fail_examples = []
    for level in ["Easy", "Medium", "Hard"]:
        items = by_level[level]
        if not items:
            continue
        counts = [r.get("path_coverage_count", 0) for r in items]
        avg_cov = sum(counts) / len(counts)
        pc_pass = sum(1 for r in items if r.get("path_coverage_pass", False))
        pc_by_level[level] = {
            "avg_coverage": round(avg_cov, 2),
            "pass_count": pc_pass,
            "pass_rate": pc_pass / len(items) if items else 0,
        }
        for r in items:
            if not r.get("path_coverage_pass", False) and len(pc_fail_examples) < 5:
                pc_fail_examples.append({
                    "question": r.get("generated_question", ""),
                    "difficulty": r.get("difficulty", ""),
                    "coverage": r.get("path_coverage_count", 0),
                    "reason": r.get("path_coverage_reason", ""),
                })

    # ── Hard degraded ──
    hard_items = by_level.get("Hard", [])
    hard_degraded_count = sum(1 for r in hard_items if r.get("hard_degraded", False))
    hard_degraded_ratio = hard_degraded_count / len(hard_items) if hard_items else 0
    hd_fail_examples = []
    for r in hard_items:
        if r.get("hard_degraded", False) and len(hd_fail_examples) < 5:
            hd_fail_examples.append({
                "question": r.get("generated_question", ""),
                "reason": r.get("hard_degraded_reason", ""),
                "single_sentence": r.get("can_answer_from_single_sentence", ""),
                "need_intermediate": r.get("need_intermediate_events", ""),
            })

    # ── Grammar failure examples (top 5) ──
    grammar_fail_top5 = []
    for r in results:
        if not r.get("grammar_pass", False) and len(grammar_fail_top5) < 5:
            grammar_fail_top5.append({
                "question": r.get("generated_question", ""),
                "difficulty": r.get("difficulty", ""),
                "reason": r.get("grammar_reason", ""),
            })

    # ── Generation error tracking ──
    gen_error_count = sum(1 for r in results if r.get("generation_error", False))
    gen_error_rate = gen_error_count / n_total if n_total else 0

    # ── Judge error tracking ──
    judge_error_count = sum(1 for r in results if r.get("answer_consistency_label") == "judge_error")
    # Judge error rate among items that reached the judge (grammar pass, no gen error)
    judge_eligible = sum(1 for r in results if r.get("grammar_pass", False) and not r.get("generation_error", False))
    judge_error_rate = judge_error_count / judge_eligible if judge_eligible else 0

    # ── Success criteria check ──
    # For consistency rates, exclude judge_error items from denominator
    ac_valid = [r for r in results if r.get("answer_consistency_label") not in ("judge_error", None, "skipped")]
    ac_yes = sum(1 for r in ac_valid if r.get("answer_consistency_label") == "yes")
    ac_partial = sum(1 for r in ac_valid if r.get("answer_consistency_label") == "partial")
    ac_no = sum(1 for r in ac_valid if r.get("answer_consistency_label") == "no")
    ac_total_valid = len(ac_valid)
    ac_yes_partial_rate = (ac_yes + ac_partial) / ac_total_valid if ac_total_valid else 0
    ac_yes_rate = ac_yes / ac_total_valid if ac_total_valid else 0
    med_pc_pass = pc_by_level.get("Medium", {}).get("pass_rate", 0)
    hard_pc_pass = pc_by_level.get("Hard", {}).get("pass_rate", 0)
    grammar_fail_rate = sum(1 for r in results if not r.get("grammar_pass", False)) / n_total if n_total else 0

    success_criteria = {
        "overall_pass_rate_50pct": overall_pass_rate >= 0.5,
        "answer_consistency_yes_partial_80pct": ac_yes_partial_rate >= 0.8,
        "answer_consistency_yes_60pct": ac_yes_rate >= 0.6,
        "medium_path_coverage_70pct": med_pc_pass >= 0.7,
        "hard_path_coverage_80pct": hard_pc_pass >= 0.8,
        "hard_degraded_20pct": hard_degraded_ratio <= 0.2,
        "grammar_fail_15pct": grammar_fail_rate <= 0.15,
    }
    all_met = all(success_criteria.values())

    # ── Recommendation ──
    if all_met:
        recommendation = "All success criteria met. Ready to scale to 300/900."
    else:
        failed = [k for k, v in success_criteria.items() if not v]
        recommendation = (
            f"NOT READY to scale. Failed criteria: {', '.join(failed)}. "
            "Analyze failures and fix prompt/filter before scaling."
        )

    # ── Build report ──
    report = {
        "n_total": n_total,
        "n_passed": n_passed,
        "overall_pass_rate": round(overall_pass_rate, 4),
        "per_level": level_stats,
        "generation_error_count": gen_error_count,
        "generation_error_rate": round(gen_error_rate, 4),
        "judge_error_count": judge_error_count,
        "judge_error_rate": round(judge_error_rate, 4),
        "grammar_failures": dict(grammar_fails.most_common()),
        "grammar_fail_examples": grammar_fail_top5,
        "weak_trigger_distribution": dict(wt_dist),
        "weak_trigger_fail_examples": wt_fail_examples,
        "answer_phrase_pass_rate": round(ap_rate, 4),
        "answer_consistency_distribution": {"yes": ac_yes, "partial": ac_partial, "no": ac_no, "judge_error": judge_error_count},
        "answer_consistency_yes_rate": round(ac_yes_rate, 4),
        "answer_consistency_yes_partial_rate": round(ac_yes_partial_rate, 4),
        "answer_consistency_fail_examples": ac_fail_examples,
        "path_coverage_by_level": pc_by_level,
        "path_coverage_fail_examples": pc_fail_examples,
        "hard_degraded_count": hard_degraded_count,
        "hard_degraded_ratio": round(hard_degraded_ratio, 4),
        "hard_degraded_examples": hd_fail_examples,
        "success_criteria": success_criteria,
        "recommendation": recommendation,
    }

    # Save JSON
    report_json_path = OUTPUT_DIR / "filter_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Saved report to {report_json_path}")

    # Save MD
    report_md = _format_report_md(report)
    report_md_path = OUTPUT_DIR / "filter_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"  Saved markdown report to {report_md_path}")

    return report


def _format_report_md(report):
    """Format report as markdown."""
    lines = []
    lines.append("# Quality Pilot 90 v2 — Filter Report\n")
    lines.append(f"**Total samples:** {report['n_total']}")
    lines.append(f"**Passed:** {report['n_passed']} ({report['overall_pass_rate']*100:.1f}%)\n")

    lines.append("## Error Rates\n")
    lines.append(f"- **Generation error:** {report['generation_error_count']}/{report['n_total']} ({report['generation_error_rate']*100:.1f}%)")
    lines.append(f"- **Judge error:** {report['judge_error_count']} ({report['judge_error_rate']*100:.1f}% of eligible)\n")

    lines.append("## Per-Level Pass Rate\n")
    lines.append("| Level | Total | Passed | Pass Rate |")
    lines.append("|-------|-------|--------|-----------|")
    for level in ["Easy", "Medium", "Hard"]:
        s = report["per_level"][level]
        lines.append(f"| {level} | {s['total']} | {s['passed']} | {s['pass_rate']*100:.1f}% |")

    lines.append("\n## Grammar Failure Distribution\n")
    lines.append("| Reason | Count |")
    lines.append("|--------|-------|")
    for reason, count in report["grammar_failures"].items():
        lines.append(f"| {reason} | {count} |")
    lines.append(f"\nExamples:\n")
    for ex in report["grammar_fail_examples"]:
        lines.append(f'- [{ex["difficulty"]}] "{ex["question"]}" → {ex["reason"]}')

    lines.append("\n## Weak Trigger Distribution\n")
    lines.append("| Type | Count |")
    lines.append("|------|-------|")
    for t, c in report["weak_trigger_distribution"].items():
        lines.append(f"| {t} | {c} |")
    if report["weak_trigger_fail_examples"]:
        lines.append(f"\nFailures:\n")
        for ex in report["weak_trigger_fail_examples"]:
            lines.append(f'- trigger="{ex["trigger"]}" type={ex["type"]} → {ex["reason"]}')

    lines.append(f"\n## Answer Phrase Pass Rate\n")
    lines.append(f"- Pass rate: {report['answer_phrase_pass_rate']*100:.1f}%\n")

    lines.append("## Answer Consistency\n")
    lines.append("| Label | Count |")
    lines.append("|-------|-------|")
    for label, count in report["answer_consistency_distribution"].items():
        lines.append(f"| {label} | {count} |")
    lines.append(f"\n- yes rate: {report['answer_consistency_yes_rate']*100:.1f}%")
    lines.append(f"- yes+partial rate: {report['answer_consistency_yes_partial_rate']*100:.1f}%")
    if report["answer_consistency_fail_examples"]:
        lines.append(f"\nInconsistency examples:\n")
        for ex in report["answer_consistency_fail_examples"]:
            lines.append(f'- [{ex["difficulty"]}] Q: "{ex["question"]}"')
            lines.append(f'  gold_phrase="{ex["gold_phrase"]}" expected="{ex["expected"]}"')
            lines.append(f'  reason: {ex["reason"]}')

    lines.append("\n## Path Coverage\n")
    lines.append("| Level | Avg Coverage | Pass Count | Pass Rate |")
    lines.append("|-------|-------------|------------|-----------|")
    for level in ["Easy", "Medium", "Hard"]:
        s = report["path_coverage_by_level"].get(level, {})
        lines.append(f"| {level} | {s.get('avg_coverage', 0)} | {s.get('pass_count', 0)} | {s.get('pass_rate', 0)*100:.1f}% |")
    if report["path_coverage_fail_examples"]:
        lines.append(f"\nPath coverage failures:\n")
        for ex in report["path_coverage_fail_examples"]:
            lines.append(f'- [{ex["difficulty"]}] "{ex["question"]}" coverage={ex["coverage"]} → {ex["reason"]}')

    lines.append("\n## Hard Degraded\n")
    lines.append(f"- Degraded count: {report['hard_degraded_count']}")
    lines.append(f"- Degraded ratio: {report['hard_degraded_ratio']*100:.1f}%")
    if report["hard_degraded_examples"]:
        lines.append(f"\nDegraded examples:\n")
        for ex in report["hard_degraded_examples"]:
            lines.append(f'- "{ex["question"]}" → {ex["reason"]}')
            lines.append(f'  single={ex["single_sentence"]}, need_intermediate={ex["need_intermediate"]}')

    lines.append("\n## Success Criteria\n")
    lines.append("| Criterion | Target | Actual | Met |")
    lines.append("|-----------|--------|--------|-----|")
    criteria_display = [
        ("overall_pass_rate", ">=50%", f"{report['overall_pass_rate']*100:.1f}%", "overall_pass_rate_50pct"),
        ("answer_consistency_yes+partial", ">=80%", f"{report['answer_consistency_yes_partial_rate']*100:.1f}%", "answer_consistency_yes_partial_80pct"),
        ("answer_consistency_yes", ">=60%", f"{report['answer_consistency_yes_rate']*100:.1f}%", "answer_consistency_yes_60pct"),
        ("medium_path_coverage", ">=70%", f"{report['path_coverage_by_level'].get('Medium', {}).get('pass_rate', 0)*100:.1f}%", "medium_path_coverage_70pct"),
        ("hard_path_coverage", ">=80%", f"{report['path_coverage_by_level'].get('Hard', {}).get('pass_rate', 0)*100:.1f}%", "hard_path_coverage_80pct"),
        ("hard_degraded", "<=20%", f"{report['hard_degraded_ratio']*100:.1f}%", "hard_degraded_20pct"),
        ("grammar_fail", "<=15%", f"{sum(report['grammar_failures'].values()) / report['n_total'] * 100:.1f}%", "grammar_fail_15pct"),
    ]
    for name, target, actual, criteria_key in criteria_display:
        met = report["success_criteria"].get(criteria_key, False)
        lines.append(f"| {name} | {target} | {actual} | {'YES' if met else 'NO'} |")

    lines.append(f"\n## Recommendation\n")
    lines.append(report["recommendation"])

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip question generation, load existing results")
    parser.add_argument("--skip_llm_filters", action="store_true",
                        help="Skip LLM-based filters (consistency, coverage, degraded)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gen_path = OUTPUT_DIR / "generated_raw.jsonl"

    if args.skip_generation:
        print("Loading existing generated questions...")
        with open(gen_path, encoding="utf-8") as f:
            results = [json.loads(line) for line in f]
        print(f"  Loaded {len(results)} items")
    else:
        # Step 1: Sample paths
        print("=" * 60)
        print("Step 1: Sampling paths (seed=42, 30/level)")
        print("=" * 60)
        sampled = load_and_sample_paths()
        print(f"  Total sampled: {len(sampled)}")

        # Step 2: Generate questions
        print(f"\n{'='*60}")
        print("Step 2: Generating questions (PathQG-HardAware)")
        print("=" * 60)
        results = generate_questions(sampled)

        # Save raw generation
        with open(gen_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved raw generation to {gen_path}")

        # Stats
        grammar_pass = sum(1 for r in results if r.get("grammar_pass", False))
        print(f"\n  Generation stats:")
        print(f"    Total: {len(results)}")
        print(f"    Grammar pass: {grammar_pass} ({grammar_pass/len(results)*100:.0f}%)")
        for level in ["Easy", "Medium", "Hard"]:
            items = [r for r in results if r["difficulty"] == level]
            p = sum(1 for r in items if r.get("grammar_pass", False))
            print(f"    {level}: {p}/{len(items)} grammar pass")

    # Step 3: Quality filters
    print(f"\n{'='*60}")
    print("Step 3: Running quality filters")
    print("=" * 60)
    results = run_quality_filters(results, skip_llm=args.skip_llm_filters)

    # Step 4: Save results
    print(f"\n{'='*60}")
    print("Step 4: Saving results")
    print("=" * 60)
    save_results(results)

    # Step 5: Generate report
    print(f"\n{'='*60}")
    print("Step 5: Generating report")
    print("=" * 60)
    report = generate_report(results)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"  Overall pass rate: {report['overall_pass_rate']*100:.1f}%")
    for level in ["Easy", "Medium", "Hard"]:
        s = report["per_level"][level]
        print(f"  {level}: {s['passed']}/{s['total']} ({s['pass_rate']*100:.1f}%)")
    print(f"\n  Answer consistency:")
    print(f"    yes: {report['answer_consistency_yes_rate']*100:.1f}%")
    print(f"    yes+partial: {report['answer_consistency_yes_partial_rate']*100:.1f}%")
    print(f"  Hard degraded: {report['hard_degraded_ratio']*100:.1f}%")
    print(f"\n  Recommendation: {report['recommendation']}")


if __name__ == "__main__":
    main()
