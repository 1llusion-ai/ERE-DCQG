"""End-to-end smoke test for the DCQG pipeline.

Runs a small sample through all stages and writes a full trace.

Usage:
    python -m scripts.run_smoke_test --limit 3 --skip_path_judge --skip_solver
"""
import argparse
import json
import random
import time
from pathlib import Path
from collections import defaultdict

from dcqg.graph import EventGraph
from dcqg.utils.jsonl import read_jsonl, write_jsonl
from dcqg.path.answer_extraction import enrich_path_item
from dcqg.path.selector import validate_answer_phrase
from dcqg.generation.generator import generate_with_retry_hardaware
from dcqg.question_filter.pipeline import quality_filter_pipeline
from dcqg.evaluation.solver import Solver
from dcqg.evaluation.judge import llm_judge_v2, quality_judge
from dcqg.tracing import build_trace_from_pipeline_result, write_full_trace, write_readable_trace


def sample_balanced(items, n_total, seed=42):
    """Take a small balanced sample across Easy/Medium/Hard."""
    rng = random.Random(seed)
    by_level = defaultdict(list)
    for item in items:
        by_level[item.get("difficulty", "Easy")].append(item)
    for pool in by_level.values():
        rng.shuffle(pool)
    levels = ["Easy", "Medium", "Hard"]
    sampled = []
    cursor = 0
    while len(sampled) < n_total and any(by_level.values()):
        level = levels[cursor % len(levels)]
        cursor += 1
        if by_level[level]:
            sampled.append(by_level[level].pop())
    rng.shuffle(sampled)
    return sampled


def load_graphs(raw_data):
    graphs = {}
    for doc in read_jsonl(raw_data):
        g = EventGraph(doc)
        graphs[g.doc_id] = g
    return graphs


def attach_graph_metadata(item, graphs):
    g = graphs.get(item.get("doc_id", ""))
    if not g:
        item["graph_nodes"] = 0
        item["graph_edges"] = 0
        return item
    item["graph_nodes"] = g.num_events
    item["graph_edges"] = g.num_edges
    return item


def refresh_answer_extraction(item):
    item = enrich_path_item(item)
    events = item.get("events", [])
    trigger = events[-1].get("trigger", item.get("answer_trigger", "")) if events else ""
    phrase = item.get("gold_answer_phrase", "")
    status = item.get("answer_phrase_status", "unknown")
    passed, reason = validate_answer_phrase(phrase, trigger, status)
    item["answer_phrase_pass"] = passed
    item["answer_phrase_reason"] = reason
    return item


def merge_context(result, source):
    preserve_keys = [
        "title", "answer_event_id", "answer_trigger", "gold_answer_phrase",
        "gold_answer_sentence", "gold_event_type", "answer_phrase_status",
        "answer_phrase_pass", "answer_phrase_reason", "weak_trigger_type",
        "weak_trigger_pass", "weak_trigger_reason", "non_temporal_count",
        "relation_group", "support_span", "rule_single_sentence_risk",
        "prefilter_pass", "prefilter_reason", "llm_path_judge",
        "llm_path_judge_status", "llm_path_keep", "llm_path_keep_reason",
        "llm_path_judge_model", "llm_path_judge_prompt", "llm_path_judge_raw_response",
        "graph_nodes", "graph_edges",
    ]
    for key in preserve_keys:
        if key in source and key not in result:
            result[key] = source[key]
    return result


def skipped_result(item, reason):
    return merge_context({
        "item_id": item.get("_item_id", 0),
        "doc_id": item.get("doc_id", ""),
        "difficulty": item.get("difficulty", ""),
        "method": "PathQG-HardAware",
        "generated_question": "",
        "gold_answer_trigger": item.get("answer_trigger", ""),
        "gold_answer_phrase": item.get("gold_answer_phrase", ""),
        "reasoning_type": "",
        "grammar_pass": False,
        "grammar_reason": reason,
        "retry_attempts": 0,
        "generation_error": True,
        "events": item.get("events", []),
        "supporting_sentences": item.get("supporting_sentences", []),
        "relation_subtypes": item.get("relation_subtypes", []),
        "generation_prompts": [],
        "generation_raw_responses": [],
        "solver_eval_status": "not_run",
        "solver_eval_reason": reason,
        "final_filter_pass": False,
        "final_filter_reason": reason,
    }, item)


def run_solver_eval(item, solver):
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

    solver_answer = solver.answer(question, context)
    answerable, solver_correct, support_covered = llm_judge_v2(question, context, gold, solver_answer)
    fluency, relevance, diff_align = quality_judge(question, path_events, difficulty)
    composite = round(
        0.25 * solver_correct + 0.20 * answerable + 0.15 * support_covered +
        0.15 * fluency + 0.10 * relevance + 0.15 * diff_align, 3)

    item["solver_eval_status"] = "ok"
    item["solver_answer"] = solver_answer
    item["judge_answerable"] = round(answerable, 2)
    item["judge_solver_correct"] = round(solver_correct, 2)
    item["judge_support_covered"] = round(support_covered, 2)
    item["quality_fluency"] = round(fluency, 2)
    item["quality_path_relevance"] = round(relevance, 2)
    item["quality_difficulty_alignment"] = round(diff_align, 2)
    item["composite"] = composite
    return item


def main():
    parser = argparse.ArgumentParser(description="Run a small full-pipeline smoke test.")
    parser.add_argument("--input", default="outputs/runs/latest/paths.prefiltered.jsonl")
    parser.add_argument("--raw_data", default="data/raw/maven_ere/valid.jsonl")
    parser.add_argument("--output_dir", default="outputs/runs/full_pipeline_smoke")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_path_judge", action="store_true")
    parser.add_argument("--skip_llm_filters", action="store_true")
    parser.add_argument("--skip_solver", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    trace_dir = output_dir / "debug_traces"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_items = read_jsonl(args.input)
    all_items = [x for x in all_items if x.get("prefilter_pass", True)]
    sampled = sample_balanced(all_items, args.limit, args.seed)
    for i, item in enumerate(sampled):
        item["_item_id"] = i

    graphs = load_graphs(args.raw_data)
    sampled = [refresh_answer_extraction(attach_graph_metadata(item, graphs)) for item in sampled]
    write_jsonl(output_dir / "sampled_input.jsonl", sampled)

    # Path judge (skip or run)
    if args.skip_path_judge:
        judged = []
        for item in sampled:
            out = dict(item)
            out["llm_path_judge_status"] = "not_run"
            out["llm_path_keep"] = True
            out["llm_path_keep_reason"] = "skip_path_judge"
            judged.append(out)
    else:
        from dcqg.path.llm_filter import judge_paths
        from types import SimpleNamespace
        from dcqg.utils.config import get_api_config
        cfg = get_api_config()
        judge_args = SimpleNamespace(
            dry_run=False, retries=1, sleep=0.25,
            api_url=cfg["AIHUBMIX_API_URL"], api_key=cfg["AIHUBMIX_API_KEY"],
            model=cfg["AIHUBMIX_MODEL"], max_tokens=300, temperature=0.0,
            timeout=90, no_json_mode=False, progress_every=1,
        )
        judged, _ = judge_paths(sampled, judge_args)
        write_jsonl(output_dir / "path_judged.jsonl", judged)

    # Generate + filter + evaluate
    solver = None if args.skip_solver else Solver()
    results = []
    for i, item in enumerate(judged):
        print(f"[{i+1}/{len(judged)}] {item.get('difficulty')} {item.get('doc_id')}")
        if not item.get("llm_path_keep", True):
            result = skipped_result(item, f"skipped_by_path_judge: {item.get('llm_path_keep_reason', '')}")
        else:
            generated, _ = generate_with_retry_hardaware(item, max_attempts=3)
            generated = merge_context(generated, item)
            generated = quality_filter_pipeline(generated, skip_llm=args.skip_llm_filters)
            generated = merge_context(generated, item)
            result = generated

        if args.skip_solver:
            result["solver_eval_status"] = "not_run"
            result["solver_eval_reason"] = "skip_solver"
        else:
            result = run_solver_eval(result, solver)

        results.append(result)
        time.sleep(0.1)

    write_jsonl(output_dir / "full_pipeline_results.jsonl", results)
    traces = [build_trace_from_pipeline_result(r, item_id=i) for i, r in enumerate(results)]
    full_trace = write_full_trace(traces, trace_dir)
    readable_trace = write_readable_trace(traces, trace_dir)

    summary = {
        "n_total": len(results),
        "generated": sum(1 for r in results if r.get("generated_question")),
        "filter_pass": sum(1 for r in results if r.get("final_filter_pass")),
        "solver_ok": sum(1 for r in results if r.get("solver_eval_status") == "ok"),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results: {output_dir / 'full_pipeline_results.jsonl'}")
    print(f"  Trace: {full_trace}")
    print(f"  Readable: {readable_trace}")


if __name__ == "__main__":
    main()
