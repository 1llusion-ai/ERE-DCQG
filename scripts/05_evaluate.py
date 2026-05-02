"""Stage 5: Run quality filter + solver + LLM judge on generated questions.

Usage:
    python -m scripts.05_evaluate --input outputs/runs/latest/questions.raw.jsonl --output outputs/runs/latest/solver_judge.jsonl --skip_solver
"""
import argparse
import json
import time
from pathlib import Path

from dcqg.utils.jsonl import read_jsonl, write_jsonl
from dcqg.question_filter.pipeline import quality_filter_pipeline
from dcqg.evaluation.judge import evaluate_item


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated questions with quality filters and solver.")
    parser.add_argument("--input", default="outputs/runs/latest/questions.raw.jsonl", help="Generated questions JSONL")
    parser.add_argument("--output", default="outputs/runs/latest/solver_judge.jsonl", help="Scored output JSONL")
    parser.add_argument("--limit", type=int, default=0, help="Limit items (0=all)")
    parser.add_argument("--skip_llm_filters", action="store_true", help="Skip LLM-based quality filters")
    parser.add_argument("--skip_solver", action="store_true", help="Skip solver + LLM judge evaluation")
    args = parser.parse_args()

    items = read_jsonl(args.input, n=args.limit or None)
    print(f"Loaded {len(items)} questions from {args.input}")

    # Quality filter
    filtered = []
    for i, item in enumerate(items):
        item = quality_filter_pipeline(item, skip_llm=args.skip_llm_filters)
        filtered.append(item)
        if (i + 1) % 20 == 0:
            n_pass = sum(1 for r in filtered if r.get("final_filter_pass", False))
            print(f"  [{i+1}/{len(items)}] filter_pass={n_pass}", flush=True)

    n_filter_pass = sum(1 for r in filtered if r.get("final_filter_pass", False))
    print(f"Quality filter: {n_filter_pass}/{len(filtered)} passed")

    # Solver + Judge
    if args.skip_solver:
        for item in filtered:
            item["solver_eval_status"] = "not_run"
        scored = filtered
    else:
        scored = []
        for i, item in enumerate(filtered):
            if item.get("final_filter_pass", False):
                item = evaluate_item(item)
                item["solver_eval_status"] = "ok"
            else:
                item["solver_eval_status"] = "skipped"
            scored.append(item)
            if (i + 1) % 20 == 0:
                n_ok = sum(1 for r in scored if r.get("solver_eval_status") == "ok")
                print(f"  [{i+1}/{len(scored)}] solver_ok={n_ok}", flush=True)
            time.sleep(0.1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, scored)

    n_solver_ok = sum(1 for r in scored if r.get("solver_eval_status") == "ok")
    print(f"\nDone: {n_filter_pass} filter-passed, {n_solver_ok} solver-evaluated")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
