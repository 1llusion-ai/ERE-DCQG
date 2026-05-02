"""Baseline alignment pilot: run ZeroShot / ICL / SelfRefine / PathQG on same strict paths.

Usage:
    python -m scripts.run_baseline_alignment
    python -m scripts.run_baseline_alignment --skip_solver
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from dcqg.utils.jsonl import read_jsonl, write_jsonl
from dcqg.generation.baselines import (
    build_zero_shot_targetqg_prompt,
    build_icl_targetqg_prompt,
    generate_baseline,
    generate_self_refine_v2,
    _make_result,
    _get_gold_trigger,
    _format_context,
    _call_llm,
    _parse_json_response,
)
from dcqg.generation.generator import generate_with_retry_hardaware
from dcqg.question_filter.pipeline import quality_filter_pipeline
from dcqg.evaluation.judge import llm_judge_v2, quality_judge, solve, target_event_hit


INPUT_FILE = "outputs/runs/path_filter_strict_pilot/paths.filtered.strict.jsonl"
SELECTED_PATHS = "outputs/runs/qg_pilot_strict_100_per_level/selected_paths.jsonl"
OUTPUT_DIR = "outputs/runs/baseline_alignment_pilot"
SEED = 42


def merge_context(result, source):
    """Copy metadata from source path to result if missing."""
    preserve_keys = [
        "title", "answer_event_id", "answer_trigger", "gold_answer_phrase",
        "gold_answer_sentence", "gold_event_type", "answer_phrase_status",
        "answer_phrase_pass", "answer_phrase_reason", "weak_trigger_type",
        "weak_trigger_pass", "weak_trigger_reason", "non_temporal_count",
        "relation_group", "support_span", "rule_single_sentence_risk",
        "prefilter_pass", "prefilter_reason", "llm_path_judge",
        "llm_path_judge_status", "llm_path_keep", "llm_path_keep_reason",
    ]
    for key in preserve_keys:
        if key in source and key not in result:
            result[key] = source[key]
    return result


def generate_pathqg(items):
    """Generate PathQG-HardAware questions (reuse from 100/level pilot if available)."""
    results = []
    for i, item in enumerate(items):
        diff = item.get("difficulty", "?")
        doc_id = item.get("doc_id", "")[:12]
        print(f"[PathQG {i+1}/{len(items)}] {diff} {doc_id}", end=" ")
        try:
            result, attempts = generate_with_retry_hardaware(item, max_attempts=3)
            result = merge_context(result, item)
            result["_item_id"] = i
            result["method"] = "PathQG-HardAware"
            gen_ok = bool(result.get("generated_question")) and not result.get("generation_error")
            print(f"-> {'OK' if gen_ok else 'FAIL'} (attempts={attempts})")
        except Exception as exc:
            result = {
                "_item_id": i,
                "doc_id": item.get("doc_id", ""),
                "difficulty": diff,
                "method": "PathQG-HardAware",
                "generated_question": "",
                "gold_answer_trigger": item.get("answer_trigger", ""),
                "generation_error": True,
                "generation_status": "exception",
                "generation_reason": f"{type(exc).__name__}: {exc}",
                "grammar_pass": False,
                "grammar_reason": "generation_exception",
                "events": item.get("events", []),
                "supporting_sentences": item.get("supporting_sentences", []),
                "relation_subtypes": item.get("relation_subtypes", []),
            }
            result = merge_context(result, item)
            print(f"-> EXCEPTION: {exc}")
        results.append(result)
        time.sleep(0.1)
    return results


def run_filter(results, skip_llm=False):
    """Run v3 quality filter on results. Modifies in-place."""
    filtered = []
    for r in results:
        if r.get("generation_error"):
            r.setdefault("final_filter_pass", False)
            r.setdefault("final_filter_reason", "generation_error")
            filtered.append(r)
            continue
        try:
            r = quality_filter_pipeline(r, skip_llm=skip_llm)
        except Exception as exc:
            r["final_filter_pass"] = False
            r["final_filter_reason"] = f"filter_exception: {type(exc).__name__}: {exc}"
        filtered.append(r)
    return filtered


def run_solver_eval(item):
    """Run solver + judge on one item."""
    if not item.get("final_filter_pass"):
        item["solver_eval_status"] = "not_run"
        item["solver_eval_reason"] = "final_filter_pass=false"
        return item

    question = item.get("generated_question", "")
    context = "\n".join(
        s if isinstance(s, str) else s[1]
        for s in item.get("supporting_sentences", [])
    )
    gold = item.get("gold_answer_trigger", "") or item.get("answer_trigger", "")
    path_events = item.get("events", [])
    difficulty = item.get("difficulty", "")

    try:
        solver_answer = solve(question, context)
        item["solver_answer"] = solver_answer

        hit_score, hit_method = target_event_hit(solver_answer, gold)
        item["target_event_hit"] = round(hit_score, 2)
        item["hit_method"] = hit_method

        answerable, solver_correct, support_covered = llm_judge_v2(question, context, gold, solver_answer)
        item["judge_answerable"] = round(answerable, 2)
        item["judge_solver_correct"] = round(solver_correct, 2)
        item["judge_support_covered"] = round(support_covered, 2)

        fluency, relevance, diff_align = quality_judge(question, path_events, difficulty)
        item["quality_fluency"] = round(fluency, 2)
        item["quality_path_relevance"] = round(relevance, 2)
        item["quality_difficulty_alignment"] = round(diff_align, 2)

        item["composite"] = round(
            0.25 * solver_correct + 0.20 * answerable + 0.15 * support_covered +
            0.15 * fluency + 0.10 * relevance + 0.15 * diff_align, 3)
        item["solver_eval_status"] = "ok"
        item["solver_eval_reason"] = ""
    except Exception as exc:
        item["solver_eval_status"] = "error"
        item["solver_eval_reason"] = f"{type(exc).__name__}: {exc}"
        for k in ["solver_answer", "judge_answerable", "judge_solver_correct",
                   "judge_support_covered", "quality_fluency", "quality_path_relevance",
                   "quality_difficulty_alignment", "composite"]:
            item.setdefault(k, 0)
    return item


def compute_method_stats(results, name):
    """Compute stats for a method's results."""
    by_level = defaultdict(list)
    for r in results:
        by_level[r.get("difficulty", "?")].append(r)

    total = len(results)
    gen_ok = sum(1 for x in results if x.get("generated_question") and not x.get("generation_error"))
    filter_pass = sum(1 for x in results if x.get("final_filter_pass"))
    pc_fail = sum(1 for x in results if "path_coverage=" in x.get("final_filter_reason", ""))
    ac_fail = sum(1 for x in results if "answer_consistency=" in x.get("final_filter_reason", ""))
    fe = sum(1 for x in results if "filter_exception" in x.get("final_filter_reason", ""))
    je = sum(1 for x in results if x.get("answer_consistency_label") == "judge_error")

    solver_xs = [x for x in results if x.get("solver_eval_status") == "ok"]
    solver_correct = sum(1 for x in solver_xs if x.get("judge_solver_correct", 0) >= 0.5)

    stats = {
        "name": name,
        "total": total,
        "generated": gen_ok,
        "gen_pct": round(gen_ok / total * 100, 1) if total else 0,
        "filter_pass": filter_pass,
        "filter_pct": round(filter_pass / total * 100, 1) if total else 0,
        "pc_fail": pc_fail,
        "ac_fail": ac_fail,
        "filter_exceptions": fe,
        "judge_errors": je,
        "solver_correct": solver_correct,
        "solver_total": len(solver_xs),
    }

    for level in ["Easy", "Medium", "Hard"]:
        xs = by_level[level]
        level_gen = sum(1 for x in xs if x.get("generated_question") and not x.get("generation_error"))
        level_pass = sum(1 for x in xs if x.get("final_filter_pass"))
        level_solver = [x for x in xs if x.get("solver_eval_status") == "ok"]
        level_correct = sum(1 for x in level_solver if x.get("judge_solver_correct", 0) >= 0.5)
        stats[f"{level}_total"] = len(xs)
        stats[f"{level}_gen"] = level_gen
        stats[f"{level}_pass"] = level_pass
        stats[f"{level}_solver_correct"] = level_correct
        stats[f"{level}_solver_total"] = len(level_solver)

    return stats


def generate_comparison_report(all_stats, output_dir):
    """Generate comparison report across all methods."""
    lines = []
    lines.append("# Baseline Alignment Pilot Report\n")
    lines.append(f"**Input:** `{SELECTED_PATHS}`")
    lines.append(f"**Methods:** ZeroShot-TargetQG, ICL-TargetQG, SelfRefine, PathQG-HardAware")
    lines.append(f"**Filter:** v3 quality filter pipeline")
    lines.append(f"**Total paths:** {all_stats[0]['total']}\n")

    # Selection
    lines.append("## Selection\n")
    lines.append("| Difficulty | Selected |")
    lines.append("|------------|---------:|")
    for level in ["Easy", "Medium", "Hard"]:
        lines.append(f"| {level} | {all_stats[0][f'{level}_total']} |")
    lines.append(f"| **Total** | **{all_stats[0]['total']}** |")

    # Main comparison table
    lines.append("\n## Method Comparison\n")
    header = "| Metric | " + " | ".join(s["name"] for s in all_stats) + " |"
    sep = "|--------|" + "|".join("---:" for _ in all_stats) + "|"
    lines.append(header)
    lines.append(sep)

    rows = [
        ("Generated", "generated", "gen_pct"),
        ("Filter Pass", "filter_pass", "filter_pct"),
        ("Path Coverage Fails", "pc_fail", None),
        ("Answer Consistency Fails", "ac_fail", None),
        ("Filter Exceptions", "filter_exceptions", None),
        ("Judge JSON Errors", "judge_errors", None),
    ]
    for label, key, pct_key in rows:
        vals = []
        for s in all_stats:
            v = s[key]
            if pct_key:
                vals.append(f"{v} ({s[pct_key]:.0f}%)")
            else:
                vals.append(str(v))
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    # Solver
    vals = [f"{s['solver_correct']}/{s['solver_total']}" for s in all_stats]
    lines.append(f"| Solver Correct | " + " | ".join(vals) + " |")

    # Per-difficulty
    lines.append("\n### Per-Difficulty Filter Pass\n")
    header = "| Difficulty | " + " | ".join(s["name"] for s in all_stats) + " |"
    sep = "|------------|" + "|".join("---:" for _ in all_stats) + "|"
    lines.append(header)
    lines.append(sep)
    for level in ["Easy", "Medium", "Hard"]:
        vals = [f"{s[f'{level}_pass']}/{s[f'{level}_total']}" for s in all_stats]
        lines.append(f"| {level} | " + " | ".join(vals) + " |")

    lines.append("\n### Per-Difficulty Solver Correct\n")
    lines.append(header)
    lines.append(sep)
    for level in ["Easy", "Medium", "Hard"]:
        vals = [f"{s[f'{level}_solver_correct']}/{s[f'{level}_solver_total']}" for s in all_stats]
        lines.append(f"| {level} | " + " | ".join(vals) + " |")

    # Key observations
    lines.append("\n## Key Observations\n")
    # Find method with highest filter pass
    best = max(all_stats, key=lambda s: s["filter_pass"])
    lines.append(f"- Highest filter pass: **{best['name']}** ({best['filter_pass']}/{best['total']}, {best['filter_pct']:.0f}%)")

    # Hard comparison
    hard_stats = [(s["name"], s["Hard_pass"], s["Hard_total"]) for s in all_stats]
    lines.append(f"- Hard filter pass: " + ", ".join(f"{n}={p}/{t}" for n, p, t in hard_stats))

    lines.append("\n## Conclusion\n")
    lines.append("- This is a baseline alignment pilot, NOT the final main experiment.")
    lines.append("- solver_correct is auxiliary — judge calibration still needed.")
    lines.append("- Primary comparison: filter pass, path coverage, answer consistency, difficulty consistency.")

    report_path = Path(output_dir) / "BASELINE_ALIGNMENT_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Baseline alignment pilot.")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--skip_solver", action="store_true")
    parser.add_argument("--methods", default="all", help="Comma-separated: zeroshot,icl,selfrefine,pathqg,all")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load selected paths
    print("=" * 60)
    print("Loading selected paths")
    print("=" * 60)
    items = read_jsonl(SELECTED_PATHS)
    for i, item in enumerate(items):
        item["_item_id"] = i
    print(f"Loaded {len(items)} paths")

    methods_to_run = args.methods.lower().split(",")
    run_all = "all" in methods_to_run

    all_stats = []

    # ── PathQG-HardAware ──
    if run_all or "pathqg" in methods_to_run:
        print("\n" + "=" * 60)
        print("PathQG-HardAware")
        print("=" * 60)
        pathqg_raw = output_dir / "PathQG-HardAware_questions.raw.jsonl"
        if pathqg_raw.exists():
            print(f"  Loading existing results from {pathqg_raw}")
            pathqg_results = read_jsonl(pathqg_raw)
        else:
            pathqg_results = generate_pathqg(items)
            write_jsonl(pathqg_raw, pathqg_results)

        pathqg_filtered = run_filter(pathqg_results)
        write_jsonl(output_dir / "PathQG-HardAware_questions.filtered.jsonl", pathqg_filtered)

        if not args.skip_solver:
            for r in pathqg_filtered:
                run_solver_eval(r)
        write_jsonl(output_dir / "PathQG-HardAware_solver.jsonl", pathqg_filtered)

        stats = compute_method_stats(pathqg_filtered, "PathQG-HardAware")
        all_stats.append(stats)
        print(f"  Filter pass: {stats['filter_pass']}/{stats['total']} ({stats['filter_pct']:.0f}%)")

    # ── ZeroShot-TargetQG ──
    if run_all or "zeroshot" in methods_to_run:
        print("\n" + "=" * 60)
        print("ZeroShot-TargetQG")
        print("=" * 60)
        zs_raw = output_dir / "ZeroShot_questions.raw.jsonl"
        if zs_raw.exists():
            print(f"  Loading existing results from {zs_raw}")
            zs_results = read_jsonl(zs_raw)
        else:
            zs_results = generate_baseline(items, build_zero_shot_targetqg_prompt, "ZeroShot-TargetQG", zs_raw)
        # Add missing fields for filter
        for i, r in enumerate(zs_results):
            r.setdefault("_item_id", i)
            r.setdefault("events", items[i].get("events", []) if i < len(items) else [])
            r.setdefault("supporting_sentences", items[i].get("supporting_sentences", []) if i < len(items) else [])
            r.setdefault("gold_answer_trigger", _get_gold_trigger(items[i]) if i < len(items) else "")

        zs_filtered = run_filter(zs_results)
        write_jsonl(output_dir / "ZeroShot_questions.filtered.jsonl", zs_filtered)

        if not args.skip_solver:
            for r in zs_filtered:
                run_solver_eval(r)
        write_jsonl(output_dir / "ZeroShot_solver.jsonl", zs_filtered)

        stats = compute_method_stats(zs_filtered, "ZeroShot-TargetQG")
        all_stats.append(stats)
        print(f"  Filter pass: {stats['filter_pass']}/{stats['total']} ({stats['filter_pct']:.0f}%)")

    # ── ICL-TargetQG ──
    if run_all or "icl" in methods_to_run:
        print("\n" + "=" * 60)
        print("ICL-TargetQG")
        print("=" * 60)
        icl_raw = output_dir / "ICL_questions.raw.jsonl"
        if icl_raw.exists():
            print(f"  Loading existing results from {icl_raw}")
            icl_results = read_jsonl(icl_raw)
        else:
            icl_results = generate_baseline(items, build_icl_targetqg_prompt, "ICL-TargetQG", icl_raw)
        for i, r in enumerate(icl_results):
            r.setdefault("_item_id", i)
            r.setdefault("events", items[i].get("events", []) if i < len(items) else [])
            r.setdefault("supporting_sentences", items[i].get("supporting_sentences", []) if i < len(items) else [])
            r.setdefault("gold_answer_trigger", _get_gold_trigger(items[i]) if i < len(items) else "")

        icl_filtered = run_filter(icl_results)
        write_jsonl(output_dir / "ICL_questions.filtered.jsonl", icl_filtered)

        if not args.skip_solver:
            for r in icl_filtered:
                run_solver_eval(r)
        write_jsonl(output_dir / "ICL_solver.jsonl", icl_filtered)

        stats = compute_method_stats(icl_filtered, "ICL-TargetQG")
        all_stats.append(stats)
        print(f"  Filter pass: {stats['filter_pass']}/{stats['total']} ({stats['filter_pct']:.0f}%)")

    # ── SelfRefine ──
    if run_all or "selfrefine" in methods_to_run:
        print("\n" + "=" * 60)
        print("SelfRefine")
        print("=" * 60)
        sr_raw = output_dir / "SelfRefine_questions.raw.jsonl"
        if sr_raw.exists():
            print(f"  Loading existing results from {sr_raw}")
            sr_results = read_jsonl(sr_raw)
        else:
            sr_results = generate_self_refine_v2(items, sr_raw)
        for i, r in enumerate(sr_results):
            r.setdefault("_item_id", i)
            r.setdefault("events", items[i].get("events", []) if i < len(items) else [])
            r.setdefault("supporting_sentences", items[i].get("supporting_sentences", []) if i < len(items) else [])
            r.setdefault("gold_answer_trigger", _get_gold_trigger(items[i]) if i < len(items) else "")

        sr_filtered = run_filter(sr_results)
        write_jsonl(output_dir / "SelfRefine_questions.filtered.jsonl", sr_filtered)

        if not args.skip_solver:
            for r in sr_filtered:
                run_solver_eval(r)
        write_jsonl(output_dir / "SelfRefine_solver.jsonl", sr_filtered)

        stats = compute_method_stats(sr_filtered, "SelfRefine")
        all_stats.append(stats)
        print(f"  Filter pass: {stats['filter_pass']}/{stats['total']} ({stats['filter_pct']:.0f}%)")

    # ── Report ──
    if all_stats:
        print("\n" + "=" * 60)
        print("Generating comparison report")
        print("=" * 60)
        generate_comparison_report(all_stats, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for s in all_stats:
        print(f"  {s['name']:25s}  gen={s['generated']}/{s['total']}  filter={s['filter_pass']}/{s['total']}  solver={s['solver_correct']}/{s['solver_total']}")
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
