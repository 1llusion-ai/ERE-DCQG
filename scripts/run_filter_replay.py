"""Filter replay: re-run filter/solver/judge on existing questions without re-generating.

Usage:
    python -m scripts.run_filter_replay --input_dir outputs/runs/qg_pilot_strict_29_v2 --output_dir outputs/runs/qg_pilot_strict_29_v3_filter_replay
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from dcqg.utils.jsonl import read_jsonl, write_jsonl
from dcqg.question_filter.pipeline import quality_filter_pipeline
from dcqg.evaluation.judge import llm_judge_v2, quality_judge, solve, target_event_hit
from dcqg.tracing import build_trace_from_pipeline_result, write_full_trace, write_readable_trace


def run_solver_eval(item):
    """Run solver + judge on one item. Modifies in-place."""
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
        item.setdefault("solver_answer", "")
        item.setdefault("judge_answerable", 0)
        item.setdefault("judge_solver_correct", 0)
        item.setdefault("judge_support_covered", 0)
        item.setdefault("quality_fluency", 0)
        item.setdefault("quality_path_relevance", 0)
        item.setdefault("quality_difficulty_alignment", 0)
        item.setdefault("composite", 0)

    return item


def main():
    parser = argparse.ArgumentParser(description="Filter replay: re-run filter/solver/judge on existing questions.")
    parser.add_argument("--input_dir", default="outputs/runs/qg_pilot_strict_29_v2", help="Source run directory (uses questions.raw.jsonl)")
    parser.add_argument("--output_dir", default="outputs/runs/qg_pilot_strict_29_v3_filter_replay", help="Output directory")
    parser.add_argument("--skip_solver", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    trace_dir = output_dir / "debug_traces"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load existing questions (raw, not filtered)
    print("=" * 60)
    print("Step 1: Load questions from raw.jsonl")
    print("=" * 60)
    raw_path = input_dir / "questions.raw.jsonl"
    generated = read_jsonl(raw_path)
    print(f"  Loaded {len(generated)} items from {raw_path}")

    # Copy selected_paths if exists
    src_paths = input_dir / "selected_paths.jsonl"
    if src_paths.exists():
        import shutil
        shutil.copy2(src_paths, output_dir / "selected_paths.jsonl")
        print(f"  Copied selected_paths.jsonl")

    # 2. Quality filter (v3 pipeline)
    print("\n" + "=" * 60)
    print("Step 2: Quality filter (v3 pipeline)")
    print("=" * 60)
    filtered = []
    for i, r in enumerate(generated):
        if r.get("generation_error"):
            r.setdefault("final_filter_pass", False)
            r.setdefault("final_filter_reason", "generation_error")
            filtered.append(r)
            continue
        try:
            r = quality_filter_pipeline(r, skip_llm=False)
        except Exception as exc:
            r["final_filter_pass"] = False
            r["final_filter_reason"] = f"filter_exception: {type(exc).__name__}: {exc}"
        filtered.append(r)

    write_jsonl(output_dir / "questions.raw.jsonl", generated)
    write_jsonl(output_dir / "questions.filtered.jsonl", filtered)
    pass_count = sum(1 for x in filtered if x.get("final_filter_pass"))
    print(f"Filter pass: {pass_count}/{len(filtered)}")

    # 3. Solver + Judge
    if not args.skip_solver:
        print("\n" + "=" * 60)
        print("Step 3: Solver + Judge")
        print("=" * 60)
        for i, r in enumerate(filtered):
            diff = r.get("difficulty", "?")
            if not r.get("final_filter_pass"):
                r["solver_eval_status"] = "not_run"
                r["solver_eval_reason"] = "final_filter_pass=false"
                print(f"[{i+1}/{len(filtered)}] {diff} SKIP (filter fail)")
                continue
            print(f"[{i+1}/{len(filtered)}] {diff}", end=" ")
            r = run_solver_eval(r)
            status = r.get("solver_eval_status", "?")
            correct = r.get("judge_solver_correct", "?")
            print(f"-> solver={status} correct={correct}")
            time.sleep(0.15)

        write_jsonl(output_dir / "solver_judge.jsonl", filtered)
        solver_ok = sum(1 for x in filtered if x.get("solver_eval_status") == "ok")
        solver_correct = sum(1 for x in filtered if x.get("judge_solver_correct", 0) >= 0.5)
        print(f"\nSolver OK: {solver_ok}, Correct: {solver_correct}")
    else:
        for r in filtered:
            r["solver_eval_status"] = "not_run"
            r["solver_eval_reason"] = "skip_solver"
        write_jsonl(output_dir / "solver_judge.jsonl", filtered)
        print("\nSolver skipped.")

    # 4. Traces
    print("\n" + "=" * 60)
    print("Step 4: Generate traces")
    print("=" * 60)
    traces = [build_trace_from_pipeline_result(r, item_id=i) for i, r in enumerate(filtered)]
    write_full_trace(traces, trace_dir)
    write_readable_trace(traces, trace_dir)
    print(f"Traces written to {trace_dir}")

    # 5. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    by_level = defaultdict(list)
    for r in filtered:
        by_level[r.get("difficulty", "?")].append(r)

    for level in ["Easy", "Medium", "Hard"]:
        xs = by_level[level]
        gen = sum(1 for x in xs if x.get("generated_question") and not x.get("generation_error"))
        passed = sum(1 for x in xs if x.get("final_filter_pass"))
        solver_xs = [x for x in xs if x.get("solver_eval_status") == "ok"]
        correct = sum(1 for x in solver_xs if x.get("judge_solver_correct", 0) >= 0.5)
        print(f"  {level}: selected={len(xs)} gen={gen} filter_pass={passed} solver_correct={correct}/{len(solver_xs)}")

    # Comparison stats
    print("\n--- v2 vs v3-replay comparison ---")
    v2_filtered = read_jsonl(input_dir / "questions.filtered.jsonl")
    for level in ["Easy", "Medium", "Hard"]:
        v2_xs = [x for x in v2_filtered if x.get("difficulty") == level]
        v3_xs = by_level[level]
        v2_pass = sum(1 for x in v2_xs if x.get("final_filter_pass"))
        v3_pass = sum(1 for x in v3_xs if x.get("final_filter_pass"))
        print(f"  {level}: v2={v2_pass}/{len(v2_xs)}  v3_replay={v3_pass}/{len(v3_xs)}")
    v2_total = sum(1 for x in v2_filtered if x.get("final_filter_pass"))
    v3_total = sum(1 for x in filtered if x.get("final_filter_pass"))
    print(f"  Total: v2={v2_total}/{len(v2_filtered)}  v3_replay={v3_total}/{len(filtered)}")

    fe = sum(1 for x in filtered if "filter_exception" in x.get("final_filter_reason", ""))
    print(f"\nFilter exceptions: {fe}")
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
