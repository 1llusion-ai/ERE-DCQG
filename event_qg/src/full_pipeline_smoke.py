"""
Run a small end-to-end DCQG pipeline and write a complete trace.

Stages:
  prefiltered path -> LLM path judge -> PathQG-HardAware generation
  -> question filter -> solver + LLM judge -> full-chain trace

This script is intentionally for smoke testing and debugging, not for full
experiments. It keeps rejected/skipped items in the trace with explicit
not_run/skipped statuses instead of leaving empty fields.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from compare_hardaware import generate_with_retry_hardaware
from answer_extraction import enrich_path_item
from evaluator import Solver
from evaluator_v2 import llm_judge_v2, quality_judge
from graph_builder import EventGraph, load_jsonl
from path_llm_judge import (
    DEFAULT_API_KEY,
    DEFAULT_API_URL,
    DEFAULT_MODEL,
    judge_paths,
    read_jsonl,
    write_jsonl,
)
from quality_filter import quality_filter_pipeline
from path_prefilter import validate_answer_phrase
from trace_utils import build_trace_from_pipeline_result, write_full_trace, write_readable_trace


def sample_balanced(items, n_total, seed):
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
    for doc in load_jsonl(raw_data):
        g = EventGraph(doc)
        graphs[g.doc_id] = g
    return graphs


def attach_graph_metadata(item, graphs):
    g = graphs.get(item.get("doc_id", ""))
    if not g:
        item["graph_nodes"] = 0
        item["graph_edges"] = 0
        item["graph_isolated_nodes"] = 0
        item["graph_relation_distribution"] = {}
        return item
    out_sources = set(g.out_neighbors.keys())
    in_targets = {tgt for _, tgt, _, _ in g.edges}
    connected = out_sources | in_targets
    item["graph_nodes"] = g.num_events
    item["graph_edges"] = g.num_edges
    item["graph_isolated_nodes"] = len(set(g.events.keys()) - connected)
    item["graph_relation_distribution"] = g.relation_type_distribution()
    return item


def refresh_answer_extraction(item):
    """Refresh answer phrase fields before path judging.

    Prefiltered files may have been produced by an older extractor. The smoke
    test should always trace the current extractor output end to end.
    """
    item = enrich_path_item(item)
    events = item.get("events", [])
    final_event = events[-1] if events else {}
    trigger = final_event.get("trigger", item.get("answer_trigger", ""))
    phrase = item.get("gold_answer_phrase", "")
    status = item.get("answer_phrase_status", "unknown")
    passed, reason = validate_answer_phrase(phrase, trigger, status)
    item["answer_phrase_pass"] = passed
    item["answer_phrase_reason"] = reason
    return item


def merge_context(result, source):
    """Preserve upstream path/prefilter/judge fields after generation/filtering."""
    preserve_keys = [
        "title",
        "answer_event_id",
        "answer_trigger",
        "gold_answer_phrase",
        "gold_answer_sentence",
        "gold_event_type",
        "answer_phrase_status",
        "answer_phrase_pass",
        "answer_phrase_reason",
        "weak_trigger_type",
        "weak_trigger_pass",
        "weak_trigger_reason",
        "non_temporal_count",
        "relation_group",
        "support_span",
        "rule_single_sentence_risk",
        "prefilter_pass",
        "prefilter_reason",
        "llm_path_judge",
        "llm_path_judge_parse_ok",
        "llm_path_judge_status",
        "llm_path_keep",
        "llm_path_keep_reason",
        "llm_path_judge_model",
        "llm_path_judge_prompt",
        "llm_path_judge_raw_response",
        "graph_nodes",
        "graph_edges",
        "graph_isolated_nodes",
        "graph_relation_distribution",
    ]
    for key in preserve_keys:
        if key in source and key not in result:
            result[key] = source[key]
    return result


def skipped_result(item, reason):
    final_event = item.get("events", [{}])[-1] if item.get("events") else {}
    return merge_context(
        {
            "item_id": item.get("_item_id", 0),
            "doc_id": item.get("doc_id", ""),
            "difficulty": item.get("difficulty", ""),
            "method": "PathQG-HardAware",
            "generated_question": "",
            "gold_answer_trigger": item.get("answer_trigger", final_event.get("trigger", "")),
            "gold_answer_phrase": item.get("gold_answer_phrase", ""),
            "gold_answer_sentence": item.get("gold_answer_sentence", ""),
            "gold_event_type": item.get("gold_event_type", final_event.get("type", "")),
            "answer_phrase_status": item.get("answer_phrase_status", "unknown"),
            "reasoning_type": "",
            "grammar_pass": False,
            "grammar_reason": reason,
            "retry_attempts": 0,
            "generation_error": True,
            "covered_event_indices": [],
            "path_binding_method": "not_run",
            "events": item.get("events", []),
            "supporting_sentences": item.get("supporting_sentences", []),
            "relation_subtypes": item.get("relation_subtypes", []),
            "relation_distribution": item.get("relation_distribution", ""),
            "generation_prompts": [],
            "generation_raw_responses": [],
            "generation_status": "not_run",
            "generation_reason": reason,
            "final_filter_pass": False,
            "final_filter_reason": reason,
            "solver_eval_status": "not_run",
            "solver_eval_reason": reason,
        },
        item,
    )


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
    answerable, solver_correct, support_covered = llm_judge_v2(
        question, context, gold, solver_answer
    )
    fluency, relevance, diff_align = quality_judge(question, path_events, difficulty)
    composite = round(
        0.25 * solver_correct
        + 0.20 * answerable
        + 0.15 * support_covered
        + 0.15 * fluency
        + 0.10 * relevance
        + 0.15 * diff_align,
        3,
    )

    item["solver_eval_status"] = "ok"
    item["solver_eval_reason"] = ""
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
    parser.add_argument("--input", default="event_qg/outputs/prefiltered_paths.jsonl")
    parser.add_argument("--raw_data", default="event_qg/data/raw/valid.jsonl")
    parser.add_argument("--output_dir", default="event_qg/outputs/full_pipeline_smoke_5")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_failed_prefilter", action="store_true")
    parser.add_argument("--skip_path_judge", action="store_true")
    parser.add_argument("--skip_llm_filters", action="store_true")
    parser.add_argument("--skip_solver", action="store_true")
    parser.add_argument("--path_judge_model", default=DEFAULT_MODEL)
    parser.add_argument("--path_judge_api_url", default=DEFAULT_API_URL)
    parser.add_argument("--path_judge_api_key", default=DEFAULT_API_KEY)
    parser.add_argument("--path_judge_retries", type=int, default=1)
    parser.add_argument("--path_judge_sleep", type=float, default=0.25)
    parser.add_argument("--path_judge_timeout", type=int, default=90)
    parser.add_argument("--no_json_mode", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    trace_dir = output_dir / "debug_traces"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_items = read_jsonl(args.input)
    if not args.include_failed_prefilter:
        all_items = [x for x in all_items if x.get("prefilter_pass", True)]
    sampled = sample_balanced(all_items, args.limit, args.seed)
    for i, item in enumerate(sampled):
        item["_item_id"] = i

    graphs = load_graphs(args.raw_data)
    sampled = [refresh_answer_extraction(attach_graph_metadata(item, graphs)) for item in sampled]
    write_jsonl(output_dir / "sampled_input.jsonl", sampled)

    if args.skip_path_judge:
        judged = []
        judge_traces = []
        for item in sampled:
            out = dict(item)
            out["llm_path_judge"] = {}
            out["llm_path_judge_status"] = "not_run"
            out["llm_path_keep"] = True
            out["llm_path_keep_reason"] = "skip_path_judge"
            judged.append(out)
    else:
        judge_args = SimpleNamespace(
            dry_run=False,
            retries=args.path_judge_retries,
            sleep=args.path_judge_sleep,
            api_url=args.path_judge_api_url,
            api_key=args.path_judge_api_key,
            model=args.path_judge_model,
            max_tokens=300,
            temperature=0.0,
            timeout=args.path_judge_timeout,
            no_json_mode=args.no_json_mode,
            progress_every=1,
        )
        judged, judge_traces = judge_paths(sampled, judge_args)
        for item, trace in zip(judged, judge_traces):
            item["llm_path_judge_prompt"] = trace.get("prompt", "")
            item["llm_path_judge_raw_response"] = trace.get("raw_response", "")
        write_jsonl(output_dir / "path_judged.jsonl", judged)
        write_jsonl(output_dir / "path_judge_trace.jsonl", judge_traces)

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
        "path_judge_status": defaultdict(int),
        "path_keep": sum(1 for r in results if r.get("llm_path_keep", True)),
        "generated": sum(1 for r in results if r.get("generated_question")),
        "filter_pass": sum(1 for r in results if r.get("final_filter_pass")),
        "solver_ok": sum(1 for r in results if r.get("solver_eval_status") == "ok"),
    }
    for r in results:
        summary["path_judge_status"][r.get("llm_path_judge_status", "missing")] += 1
    summary["path_judge_status"] = dict(summary["path_judge_status"])
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"  results: {output_dir / 'full_pipeline_results.jsonl'}")
    print(f"  trace:   {full_trace}")
    print(f"  readable:{readable_trace}")
    print(f"  summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
