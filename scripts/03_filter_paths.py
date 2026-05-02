"""Stage 3: Filter paths with prefilter + LLM path judge + strict/relaxed policy.

Usage:
    python -m scripts.03_filter_paths --input outputs/runs/latest/paths.raw.jsonl --output_dir outputs/runs/latest
    python -m scripts.03_filter_paths --input outputs/runs/latest/paths.raw.jsonl --output_dir outputs/runs/latest --skip_llm_judge
"""
import argparse
import json
from pathlib import Path

from dcqg.path.diagnostics import prefilter_path
from dcqg.path.selector import validate_answer_phrase
from dcqg.path.answer_extraction import enrich_path_item
from dcqg.path.llm_filter import (
    judge_paths, apply_policy, deduplicate, generate_filter_report,
)
from dcqg.utils.jsonl import read_jsonl, write_jsonl


TRACE_FIELDS = [
    "llm_path_judge_prompt",
    "llm_path_judge_raw_response",
    "llm_path_judge",
    "llm_path_judge_status",
    "llm_path_judge_parse_ok",
    "llm_path_judge_model",
    "llm_path_keep_reason",
    "policy_strict_keep",
    "policy_relaxed_keep",
    "policy_strict_reason",
    "policy_relaxed_reason",
    "gold_answer_phrase",
    "gold_answer_sentence",
    "gold_event_type",
    "answer_phrase_status",
    "answer_phrase_pass",
    "answer_phrase_reason",
    "prefilter_pass",
    "prefilter_reason",
    "relation_group",
    "relation_subtypes",
    "support_span",
    "rule_single_sentence_risk",
    "weak_trigger_type",
    "weak_trigger_pass",
    "weak_trigger_reason",
    "non_temporal_count",
    "dedup_key",
    "dedup_removed",
    "dedup_reason",
]


def main():
    parser = argparse.ArgumentParser(description="Filter paths with prefilter + LLM judge + strict/relaxed policy.")
    parser.add_argument("--input", default="outputs/runs/latest/paths.raw.jsonl", help="Sampled paths JSONL")
    parser.add_argument("--output_dir", default="outputs/runs/latest", help="Output directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit items (0=all)")
    parser.add_argument("--skip_llm_judge", action="store_true", help="Skip LLM path judge")
    parser.add_argument("--model", default=None, help="Path judge model override")
    parser.add_argument("--sample_per_level", type=int, default=0, help="Sample N per difficulty level for LLM judge (0=all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load raw ---
    items = read_jsonl(args.input, n=args.limit or None)
    raw_count = len(items)
    print(f"Loaded {raw_count} paths from {args.input}")

    # --- Enrich with answer phrases ---
    items = [enrich_path_item(p) for p in items]

    # --- Prefilter ---
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
    prefiltered_count = len(prefilter_passed)
    print(f"Prefilter: {prefiltered_count}/{raw_count} passed")
    write_jsonl(output_dir / "paths.prefiltered.jsonl", prefiltered)

    # --- Sample per level for LLM judge ---
    judge_input = prefilter_passed
    if args.sample_per_level > 0:
        import random
        rng = random.Random(args.seed)
        by_level = {}
        for item in judge_input:
            d = item.get("difficulty", "Easy")
            by_level.setdefault(d, []).append(item)
        sampled = []
        for level in ["Easy", "Medium", "Hard"]:
            pool = list(by_level.get(level, []))
            rng.shuffle(pool)
            n = min(args.sample_per_level, len(pool))
            sampled.extend(pool[:n])
        rng.shuffle(sampled)
        judge_input = sampled
        print(f"Sampled {len(judge_input)} items ({args.sample_per_level}/level)")

    # --- LLM path judge ---
    if args.skip_llm_judge:
        for item in judge_input:
            item["llm_path_judge"] = {
                "path_questionable": "yes",
                "expected_required_steps": "2",
                "single_sentence_risk": "medium",
                "recommended_difficulty": item.get("difficulty", "easy").lower(),
                "can_write_path_dependent_question": "yes",
                "reason": "skip_llm_judge",
            }
            item["llm_path_judge_status"] = "not_run"
            item["llm_path_judge_parse_ok"] = True
            item["llm_path_judge_model"] = "skip"
            item["llm_path_judge_prompt"] = ""
            item["llm_path_judge_raw_response"] = ""
        judged = judge_input
        traces = []
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
        judged, traces = judge_paths(judge_input, judge_args)
        write_jsonl(output_dir / "path_judge_trace.jsonl", traces)

    # --- Apply strict/relaxed policy ---
    judged = apply_policy(judged)

    strict_items = [x for x in judged if x.get("policy_strict_keep", False)]
    relaxed_items = [x for x in judged if x.get("policy_relaxed_keep", False)]
    rejected_items = [x for x in judged if not x.get("policy_relaxed_keep", False)]

    print(f"Strict kept: {len(strict_items)}/{len(judged)}")
    print(f"Relaxed kept: {len(relaxed_items)}/{len(judged)}")
    print(f"Rejected: {len(rejected_items)}/{len(judged)}")

    # --- Dedup ---
    strict_dedup, strict_dup_removed = deduplicate(strict_items)
    relaxed_dedup, relaxed_dup_removed = deduplicate(relaxed_items)

    # Add dedup fields to all items for trace
    all_with_dedup = []
    dedup_map = {}
    for x in strict_dedup:
        dedup_map[x.get("dedup_key", "")] = False
    for x in strict_dup_removed:
        dedup_map[x.get("dedup_key", "")] = True
    for x in relaxed_dedup:
        dedup_map.setdefault(x.get("dedup_key", ""), False)
    for x in relaxed_dup_removed:
        dedup_map.setdefault(x.get("dedup_key", ""), True)

    # Re-add dedup fields to judged items
    for item in judged:
        doc_id = item.get("doc_id", "")
        event_id = item.get("answer_event_id", "")
        phrase = item.get("gold_answer_phrase", "").lower().strip()
        key = f"{doc_id}::{event_id}"
        norm_phrase = " ".join(phrase.split())
        fallback_key = f"{doc_id}::phrase::{norm_phrase}"
        if key in dedup_map:
            item["dedup_key"] = key
            item["dedup_removed"] = dedup_map[key]
            item["dedup_reason"] = "duplicate doc_id+answer_event_id" if dedup_map[key] else ""
        elif fallback_key in dedup_map:
            item["dedup_key"] = fallback_key
            item["dedup_removed"] = dedup_map[fallback_key]
            item["dedup_reason"] = "duplicate doc_id+answer_phrase" if dedup_map[fallback_key] else ""
        else:
            item["dedup_key"] = key
            item["dedup_removed"] = False
            item["dedup_reason"] = ""
        all_with_dedup.append(item)

    print(f"Strict dedup: {len(strict_dedup)} kept, {len(strict_dup_removed)} removed")
    print(f"Relaxed dedup: {len(relaxed_dedup)} kept, {len(relaxed_dup_removed)} removed")

    # --- Write outputs ---
    # strict: main experiment, deduped
    write_jsonl(output_dir / "paths.filtered.strict.jsonl", strict_dedup)
    # relaxed: analysis pool, deduped
    write_jsonl(output_dir / "paths.filtered.relaxed.jsonl", relaxed_dedup)
    # relaxed: also save non-deduped for reference
    write_jsonl(output_dir / "paths.filtered.relaxed.all.jsonl", relaxed_items)
    # rejected
    write_jsonl(output_dir / "paths.rejected.jsonl", rejected_items)
    # all judged (for trace completeness)
    write_jsonl(output_dir / "paths.judged.all.jsonl", all_with_dedup)

    # --- Generate report ---
    generate_filter_report(
        all_items=all_with_dedup,
        strict_items=strict_dedup,
        relaxed_items=relaxed_dedup,
        rejected_items=rejected_items,
        dedup_removed_strict=strict_dup_removed,
        dedup_removed_relaxed=relaxed_dup_removed,
        raw_count=raw_count,
        prefiltered_count=prefiltered_count,
        report_path=output_dir / "PATH_FILTER_REPORT.md",
    )

    # --- Summary ---
    by_level = {}
    for x in strict_dedup:
        d = x.get("difficulty", "?")
        by_level[d] = by_level.get(d, 0) + 1
    print(f"\nStrict (deduped) by level: {by_level}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
