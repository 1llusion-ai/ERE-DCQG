"""Stage 3: Filter paths with prefilter + LLM path judge.

Usage:
    python -m scripts.03_filter_paths --input outputs/runs/latest/paths.raw.jsonl --output_dir outputs/runs/latest --skip_llm_judge
"""
import argparse
import json
from pathlib import Path

from dcqg.path.diagnostics import prefilter_path
from dcqg.path.selector import validate_answer_phrase
from dcqg.path.answer_extraction import enrich_path_item
from dcqg.path.llm_filter import judge_paths
from dcqg.utils.jsonl import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description="Filter sampled paths with prefilter and LLM judge.")
    parser.add_argument("--input", default="outputs/runs/latest/paths.raw.jsonl", help="Sampled paths JSONL")
    parser.add_argument("--output_dir", default="outputs/runs/latest", help="Output directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit items (0=all)")
    parser.add_argument("--skip_llm_judge", action="store_true", help="Skip LLM path judge")
    parser.add_argument("--model", default=None, help="Path judge model override")
    parser.add_argument("--sample_per_level", type=int, default=0, help="Sample N per difficulty level (0=all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = read_jsonl(args.input, n=args.limit or None)
    print(f"Loaded {len(items)} paths from {args.input}")

    # Enrich with answer phrases
    items = [enrich_path_item(p) for p in items]

    # Prefilter
    prefiltered = []
    for item in items:
        item = prefilter_path(item)
        phrase = item.get("gold_answer_phrase", "")
        trigger = item.get("answer_trigger", "")
        status = item.get("answer_phrase_status", "unknown")
        passed, reason = validate_answer_phrase(phrase, trigger, status)
        item["answer_phrase_pass"] = passed
        item["answer_phrase_reason"] = reason
        prefiltered.append(item)

    prefilter_passed = [p for p in prefiltered if p.get("prefilter_pass", True)]
    print(f"Prefilter: {len(prefilter_passed)}/{len(prefiltered)} passed")
    write_jsonl(output_dir / "paths.prefiltered.jsonl", prefiltered)

    # Optional: sample per level
    if args.sample_per_level > 0:
        import random
        random.seed(42)
        by_level = {}
        for item in prefilter_passed:
            d = item.get("difficulty", "Easy")
            by_level.setdefault(d, []).append(item)
        sampled = []
        for level in ["Easy", "Medium", "Hard"]:
            pool = by_level.get(level, [])
            n = min(args.sample_per_level, len(pool))
            sampled.extend(random.sample(pool, n))
        prefilter_passed = sampled
        print(f"Sampled {len(prefilter_passed)} items ({args.sample_per_level}/level)")

    # LLM path judge
    if args.skip_llm_judge:
        for item in prefilter_passed:
            item["llm_path_judge_status"] = "not_run"
            item["llm_path_keep"] = True
            item["llm_path_keep_reason"] = "skip_llm_judge"
        judged = prefilter_passed
    else:
        from types import SimpleNamespace
        from dcqg.utils.config import get_api_config
        cfg = get_api_config()
        judge_args = SimpleNamespace(
            dry_run=False,
            retries=1,
            sleep=0.25,
            api_url=cfg["AIHUBMIX_API_URL"],
            api_key=cfg["AIHUBMIX_API_KEY"],
            model=args.model or cfg["AIHUBMIX_MODEL"],
            max_tokens=300,
            temperature=0.0,
            timeout=90,
            no_json_mode=False,
            progress_every=10,
        )
        judged, traces = judge_paths(prefilter_passed, judge_args)
        write_jsonl(output_dir / "path_judge_trace.jsonl", traces)

    kept = [j for j in judged if j.get("llm_path_keep", True)]
    print(f"LLM judge: {len(kept)}/{len(judged)} kept")
    write_jsonl(output_dir / "paths.filtered.jsonl", judged)
    print(f"Saved to {output_dir / 'paths.filtered.jsonl'}")


if __name__ == "__main__":
    main()
