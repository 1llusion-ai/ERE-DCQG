"""Full pipeline orchestrator for DCQG.

Runs all stages end-to-end: sample -> filter -> generate -> evaluate.

Usage:
    python -m scripts.run_pipeline --limit 50 --skip_path_judge --skip_solver
"""
import argparse
import json
import time
import random
from pathlib import Path
from collections import defaultdict

from dcqg.graph import EventGraph
from dcqg.utils.jsonl import read_jsonl, write_jsonl
from dcqg.path.answer_extraction import enrich_path_item, is_valid_final_event
from dcqg.path.diagnostics import prefilter_path
from dcqg.path.selector import validate_answer_phrase
from dcqg.generation.generator import generate_with_retry_hardaware
from dcqg.question_filter.pipeline import quality_filter_pipeline
from dcqg.evaluation.judge import evaluate_item
from dcqg.tracing import build_trace_from_pipeline_result, write_full_trace, write_readable_trace


def merge_context(result, source):
    """Preserve upstream path/filter/judge context after generation."""
    for key, value in source.items():
        if key not in result:
            result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description="Run full DCQG pipeline.")
    parser.add_argument("--raw_data", default="data/raw/maven_ere/valid.jsonl")
    parser.add_argument("--output_dir", default="outputs/runs/full_pipeline")
    parser.add_argument("--limit", type=int, default=50, help="Total items to process")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_per_doc", type=int, default=3)
    parser.add_argument("--skip_path_judge", action="store_true")
    parser.add_argument("--skip_llm_filters", action="store_true")
    parser.add_argument("--skip_solver", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    trace_dir = output_dir / "traces"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Sample paths
    print("=" * 60)
    print("Stage 1: Sampling paths")
    print("=" * 60)
    docs = read_jsonl(args.raw_data)
    print(f"  Loaded {len(docs)} documents")

    all_paths = []
    target_counts = {"Easy": args.max_per_doc, "Medium": args.max_per_doc, "Hard": args.max_per_doc}
    rng = random.Random(args.seed)
    for doc in docs:
        from dcqg.path.sampler import sample_from_doc
        paths = sample_from_doc(EventGraph(doc), target_counts=target_counts, rng=rng)
        all_paths.extend(paths)

    # Enrich and prefilter
    all_paths = [enrich_path_item(p) for p in all_paths]
    filtered_paths = []
    for item in all_paths:
        item = prefilter_path(item)
        phrase = item.get("gold_answer_phrase", "")
        trigger = item.get("answer_trigger", "")
        status = item.get("answer_phrase_status", "unknown")
        passed, reason = validate_answer_phrase(phrase, trigger, status)
        item["answer_phrase_pass"] = passed
        item["answer_phrase_reason"] = reason
        filtered_paths.append(item)
    all_paths = filtered_paths

    valid = [p for p in all_paths if p.get("prefilter_pass", True)]
    print(f"  Sampled {len(all_paths)} paths, {len(valid)} prefilter-passed")

    # Balanced sample
    by_level = defaultdict(list)
    for p in valid:
        by_level[p["difficulty"]].append(p)
    random.seed(args.seed)
    sampled = []
    levels = ["Easy", "Medium", "Hard"]
    for pool in by_level.values():
        random.shuffle(pool)
    cursor = 0
    while len(sampled) < args.limit and any(by_level.values()):
        level = levels[cursor % len(levels)]
        cursor += 1
        if by_level[level]:
            sampled.append(by_level[level].pop())
    random.shuffle(sampled)
    print(f"  Selected {len(sampled)} items for pipeline")
    for i, item in enumerate(sampled):
        item["_item_id"] = i

    write_jsonl(output_dir / "paths.raw.jsonl", sampled)

    # Stage 2: LLM path judge
    print(f"\n{'='*60}")
    print("Stage 2: Path judge")
    print("=" * 60)
    if args.skip_path_judge:
        for item in sampled:
            item["llm_path_judge_status"] = "not_run"
            item["llm_path_keep"] = True
            item["llm_path_keep_reason"] = "skip_path_judge"
        judged = sampled
        print(f"  Skipped ({len(judged)} items)")
    else:
        from dcqg.path.llm_filter import judge_paths
        from types import SimpleNamespace
        from dcqg.utils.config import get_api_config
        cfg = get_api_config()
        judge_args = SimpleNamespace(
            dry_run=False, retries=1, sleep=0.25,
            api_url=cfg["AIHUBMIX_API_URL"], api_key=cfg["AIHUBMIX_API_KEY"],
            model=cfg["AIHUBMIX_MODEL"], max_tokens=300, temperature=0.0,
            timeout=90, no_json_mode=False, progress_every=10,
        )
        judged, _ = judge_paths(sampled, judge_args)
        kept = [j for j in judged if j.get("llm_path_keep", True)]
        print(f"  {len(kept)}/{len(judged)} kept")
    write_jsonl(output_dir / "paths.filtered.jsonl", judged)

    # Stage 3: Generate questions
    print(f"\n{'='*60}")
    print("Stage 3: Generate questions")
    print("=" * 60)
    gen_items = [j for j in judged if j.get("llm_path_keep", True)]
    results = []
    for i, item in enumerate(gen_items):
        r, _ = generate_with_retry_hardaware(item, max_attempts=3)
        r = merge_context(r, item)
        results.append(r)
        if (i + 1) % 20 == 0:
            n_pass = sum(1 for r in results if r.get("grammar_pass", False))
            print(f"  [{i+1}/{len(gen_items)}] grammar_pass={n_pass}", flush=True)
        time.sleep(0.1)

    n_grammar = sum(1 for r in results if r.get("grammar_pass", False))
    print(f"  Generated {len(results)}, {n_grammar} grammar-passed")
    write_jsonl(output_dir / "questions.raw.jsonl", results)

    # Stage 4: Quality filter
    print(f"\n{'='*60}")
    print("Stage 4: Quality filter")
    print("=" * 60)
    for i, r in enumerate(results):
        is_gen_error = r.get("generation_error", False)
        r = quality_filter_pipeline(r, skip_llm=(args.skip_llm_filters or is_gen_error))
        if (i + 1) % 20 == 0:
            n_pass = sum(1 for r in results if r.get("final_filter_pass", False))
            print(f"  [{i+1}/{len(results)}] filter_pass={n_pass}", flush=True)
        if not args.skip_llm_filters and not is_gen_error:
            time.sleep(0.1)

    n_filter = sum(1 for r in results if r.get("final_filter_pass", False))
    print(f"  {n_filter}/{len(results)} passed quality filter")
    write_jsonl(output_dir / "questions.filtered.jsonl", results)

    # Stage 5: Solver + Judge
    print(f"\n{'='*60}")
    print("Stage 5: Solver + Judge")
    print("=" * 60)
    if args.skip_solver:
        for r in results:
            r["solver_eval_status"] = "not_run"
        print("  Skipped")
    else:
        for i, r in enumerate(results):
            if r.get("final_filter_pass", False):
                r = evaluate_item(r)
                r["solver_eval_status"] = "ok"
            else:
                r["solver_eval_status"] = "skipped"
            if (i + 1) % 20 == 0:
                n_ok = sum(1 for r in results if r.get("solver_eval_status") == "ok")
                print(f"  [{i+1}/{len(results)}] solver_ok={n_ok}", flush=True)
            time.sleep(0.1)

    write_jsonl(output_dir / "solver_judge.jsonl", results)

    # Stage 6: Trace
    print(f"\n{'='*60}")
    print("Stage 6: Writing traces")
    print("=" * 60)
    traces = [build_trace_from_pipeline_result(r, item_id=i) for i, r in enumerate(results)]
    full_trace = write_full_trace(traces, trace_dir)
    readable_trace = write_readable_trace(traces, trace_dir)

    # Summary
    summary = {
        "n_total": len(results),
        "grammar_pass": sum(1 for r in results if r.get("grammar_pass", False)),
        "filter_pass": sum(1 for r in results if r.get("final_filter_pass", False)),
        "solver_ok": sum(1 for r in results if r.get("solver_eval_status") == "ok"),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Output: {output_dir}")
    print(f"  Results: {output_dir / 'solver_judge.jsonl'}")
    print(f"  Trace: {full_trace}")
    print(f"  Readable: {readable_trace}")


if __name__ == "__main__":
    main()
