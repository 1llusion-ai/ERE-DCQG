"""
Full-chain debug trace for DCQG pipeline.
Shared by all pipeline stages. Each stage adds its own section to the trace record.

Usage:
    from trace_utils import TraceRecord, write_full_trace, write_readable_trace

    trace = TraceRecord(doc_id="...", item_id=0)
    trace.set_raw_source(doc_id, event_count, relation_count)
    trace.set_graph_stage(nodes, edges, isolated_nodes)
    # ... each stage adds its section ...
    traces = [build_trace_from_pipeline_result(r, item_id=i) for i, r in enumerate(results)]
    write_full_trace(traces, output_dir)
    write_readable_trace(traces, output_dir)
"""
import json
from pathlib import Path
from datetime import datetime


class TraceRecord:
    """Accumulates trace data across pipeline stages for one item."""

    def __init__(self, doc_id="", item_id=0):
        self.data = {
            "item_id": item_id,
            "doc_id": doc_id,
            "timestamp": datetime.now().isoformat(),
        }

    def set_raw_source(self, doc_id, event_count, relation_count, title=""):
        self.data["raw_source"] = {
            "doc_id": doc_id,
            "event_count": event_count,
            "relation_count": relation_count,
            "title": title,
        }

    def set_graph_stage(self, nodes, edges, isolated_nodes, relation_distribution=None):
        self.data["graph_stage"] = {
            "nodes": nodes,
            "edges": edges,
            "isolated_nodes": isolated_nodes,
            "relation_distribution": relation_distribution or {},
        }

    def set_path_sampling(self, difficulty, hop_count, path_events, relation_subtypes,
                          relation_distribution="", difficulty_score=None):
        self.data["path_sampling"] = {
            "difficulty": difficulty,
            "hop_count": hop_count,
            "path_events": path_events,
            "relation_subtypes": relation_subtypes,
            "relation_distribution": relation_distribution,
            "difficulty_score": difficulty_score or {},
        }

    def set_answer_extraction(self, trigger, answer_phrase, answer_sentence,
                              answer_phrase_status, extraction_method=""):
        self.data["answer_extraction"] = {
            "trigger": trigger,
            "answer_phrase": answer_phrase,
            "answer_sentence": answer_sentence,
            "answer_phrase_status": answer_phrase_status,
            "extraction_method": extraction_method,
        }

    def set_prefilter(self, prefilter_pass, prefilter_reason, weak_trigger_type="",
                      relation_group="", support_span=0, rule_single_sentence_risk=""):
        self.data["prefilter"] = {
            "prefilter_pass": prefilter_pass,
            "prefilter_reason": prefilter_reason,
            "weak_trigger_type": weak_trigger_type,
            "relation_group": relation_group,
            "support_span": support_span,
            "rule_single_sentence_risk": rule_single_sentence_risk,
        }

    def set_llm_path_judge(self, status, path_questionable="", recommended_difficulty="",
                           judge_raw_response="", expected_required_steps="",
                           single_sentence_risk="", keep=None, keep_reason="",
                           model="", prompt=""):
        self.data["llm_path_judge"] = {
            "llm_path_judge_status": status,
            "path_questionable": path_questionable,
            "expected_required_steps": expected_required_steps,
            "single_sentence_risk": single_sentence_risk,
            "recommended_difficulty": recommended_difficulty,
            "keep": keep,
            "keep_reason": keep_reason,
            "model": model,
            "prompt": prompt,
            "judge_raw_response": judge_raw_response,
        }

    def set_qg_generation(self, generator_model="", prompts=None, raw_responses=None,
                          parsed_question="", retry_attempts=0, status="", reason=""):
        self.data["qg_generation"] = {
            "status": status,
            "reason": reason,
            "generator_model": generator_model,
            "prompts": prompts or [],
            "raw_responses": raw_responses or [],
            "parsed_question": parsed_question,
            "retry_attempts": retry_attempts,
        }

    def set_quality_filter(self, grammar_check=None, weak_trigger_check=None,
                           answer_phrase_check=None, consistency_judge=None,
                           path_coverage=None, asks_target_event=None,
                           hard_degraded=None, filter_pass=False, filter_reason=""):
        self.data["quality_filter"] = {
            "grammar_check": grammar_check or {},
            "weak_trigger_check": weak_trigger_check or {},
            "answer_phrase_check": answer_phrase_check or {},
            "consistency_judge": consistency_judge or {},
            "path_coverage": path_coverage or {},
            "asks_target_event": asks_target_event,
            "hard_degraded": hard_degraded or {},
            "filter_pass": filter_pass,
            "filter_reason": filter_reason,
        }

    def set_solver_eval(self, solver_result="", solver_confidence=0.0,
                        status="", reason="", judge_answerable=None,
                        judge_solver_correct=None, judge_support_covered=None,
                        quality_fluency=None, quality_path_relevance=None,
                        quality_difficulty_alignment=None, composite=None):
        self.data["solver_eval"] = {
            "status": status,
            "reason": reason,
            "solver_result": solver_result,
            "solver_confidence": solver_confidence,
            "judge_answerable": judge_answerable,
            "judge_solver_correct": judge_solver_correct,
            "judge_support_covered": judge_support_covered,
            "quality_fluency": quality_fluency,
            "quality_path_relevance": quality_path_relevance,
            "quality_difficulty_alignment": quality_difficulty_alignment,
            "composite": composite,
        }

    def to_dict(self):
        return self.data

    def to_json(self):
        return json.dumps(self.data, ensure_ascii=False)


def write_full_trace(traces, output_dir):
    """Write full_trace.jsonl -- one JSON line per item, no truncation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "full_trace.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for trace in traces:
            f.write(trace.to_json() + "\n")
    return path


def write_readable_trace(traces, output_dir, failures_only=False):
    """Write readable_trace.md -- human-readable summary of all items."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "readable_trace.md"

    lines = []
    lines.append("# Full-Chain Debug Trace\n")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    lines.append(f"**Total items:** {len(traces)}\n")
    lines.append("---\n")

    for trace in traces:
        d = trace.to_dict()
        item_id = d.get("item_id", "?")
        doc_id = d.get("doc_id", "?")
        difficulty = d.get("path_sampling", {}).get("difficulty", "?")

        qf = d.get("quality_filter", {})
        passed = qf.get("filter_pass", False)
        status = "PASS" if passed else "FAIL"

        if failures_only and passed:
            continue

        lines.append(f"## Item {item_id} [{difficulty}] -- {status}\n")
        lines.append(f"**doc_id:** {doc_id}")

        # Raw source
        rs = d.get("raw_source", {})
        if rs:
            lines.append(f"**Raw source:** events={rs.get('event_count', 0)}, relations={rs.get('relation_count', 0)}")

        # Graph
        gs = d.get("graph_stage", {})
        if gs:
            lines.append(f"**Graph:** nodes={gs.get('nodes', 0)}, edges={gs.get('edges', 0)}")

        # Path
        ps = d.get("path_sampling", {})
        if ps:
            path_events = ps.get("path_events", [])
            path_str = " -> ".join(f'"{e.get("trigger", "")}"' for e in path_events)
            lines.append(f"**Path ({ps.get('hop_count', 0)} hops):** {path_str}")
            lines.append(f"**Relations:** {', '.join(ps.get('relation_subtypes', []))}")

        # Answer extraction
        ae = d.get("answer_extraction", {})
        if ae:
            lines.append(f"**Answer phrase:** \"{ae.get('answer_phrase', '')}\" (status={ae.get('answer_phrase_status', '')})")

        # Prefilter
        pf = d.get("prefilter", {})
        if pf:
            pf_status = "PASS" if pf.get("prefilter_pass") else "FAIL"
            lines.append(f"**Prefilter:** {pf_status} -- {pf.get('prefilter_reason', '')}")

        # LLM path judge
        lpj = d.get("llm_path_judge", {})
        if lpj:
            lines.append(f"**Path judge:** status={lpj.get('llm_path_judge_status', '')} questionable={lpj.get('path_questionable', '')} recommended={lpj.get('recommended_difficulty', '')}")

        # QG
        qg = d.get("qg_generation", {})
        if qg:
            lines.append(
                f"**Question:** \"{qg.get('parsed_question', '')}\" "
                f"(status={qg.get('status', '')}, attempts={qg.get('retry_attempts', 0)})"
            )

        # Quality filter
        if qf:
            grammar = qf.get("grammar_check", {})
            lines.append(f"**Grammar:** {'PASS' if grammar.get('pass') else 'FAIL'}")
            lines.append(f"**Filter reason:** {qf.get('filter_reason', '')}")

        # Solver
        se = d.get("solver_eval", {})
        if se:
            lines.append(
                f"**Solver:** status={se.get('status', '')} "
                f"result={se.get('solver_result', '')} "
                f"correct={se.get('judge_solver_correct', '')} "
                f"confidence={se.get('solver_confidence', 0)}"
            )

        lines.append("\n---\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def build_trace_from_pipeline_result(r, item_id=0):
    """Build a TraceRecord from an existing pipeline result dict.
    Used by quality_pilot.py to produce traces without changing its internal flow.
    """
    trace = TraceRecord(doc_id=r.get("doc_id", ""), item_id=item_id)

    events = r.get("events", [])

    # Raw source
    trace.set_raw_source(
        doc_id=r.get("doc_id", ""),
        event_count=len(events),
        relation_count=len(r.get("relation_subtypes", [])),
        title=r.get("title", ""),
    )

    # Path sampling
    path_events = []
    for e in events:
        path_events.append({
            "id": e.get("id", ""),
            "type": e.get("type", ""),
            "trigger": e.get("trigger", ""),
            "sent_id": e.get("sent_id", -1),
            "offset": e.get("offset", []),
        })
    trace.set_path_sampling(
        difficulty=r.get("difficulty", ""),
        hop_count=len(events) - 1 if events else 0,
        path_events=path_events,
        relation_subtypes=r.get("relation_subtypes", []),
        relation_distribution=r.get("relation_distribution", ""),
        difficulty_score={
            "D": r.get("difficulty_score", 0),
            "PL": r.get("PL", 0),
            "RD": r.get("RD", 0),
            "ES": r.get("ES", 0),
            "EA": r.get("EA", 0),
        },
    )

    if any(k in r for k in ["graph_nodes", "graph_edges", "graph_isolated_nodes", "graph_relation_distribution"]):
        trace.set_graph_stage(
            nodes=r.get("graph_nodes", 0),
            edges=r.get("graph_edges", 0),
            isolated_nodes=r.get("graph_isolated_nodes", 0),
            relation_distribution=r.get("graph_relation_distribution", {}),
        )

    # Answer extraction
    trace.set_answer_extraction(
        trigger=r.get("gold_answer_trigger", ""),
        answer_phrase=r.get("gold_answer_phrase", ""),
        answer_sentence=r.get("gold_answer_sentence", ""),
        answer_phrase_status=r.get("answer_phrase_status", "unknown"),
    )

    # Prefilter
    trace.set_prefilter(
        prefilter_pass=r.get("prefilter_pass", True),
        prefilter_reason=r.get("prefilter_reason", ""),
        weak_trigger_type=r.get("weak_trigger_type", ""),
        relation_group=r.get("relation_group", ""),
        support_span=r.get("support_span", 0),
        rule_single_sentence_risk=r.get("rule_single_sentence_risk", ""),
    )

    # LLM path judge
    llm_path_judge = r.get("llm_path_judge", {}) or {}
    trace.set_llm_path_judge(
        status=r.get("llm_path_judge_status", "not_run"),
        path_questionable=llm_path_judge.get("path_questionable", ""),
        expected_required_steps=llm_path_judge.get("expected_required_steps", ""),
        single_sentence_risk=llm_path_judge.get("single_sentence_risk", ""),
        recommended_difficulty=llm_path_judge.get("recommended_difficulty", ""),
        judge_raw_response=r.get("llm_path_judge_raw_response", ""),
        keep=r.get("llm_path_keep"),
        keep_reason=r.get("llm_path_keep_reason", ""),
        model=r.get("llm_path_judge_model", ""),
        prompt=r.get("llm_path_judge_prompt", ""),
    )

    # QG generation
    qg_prompts = r.get("generation_prompts", [])
    qg_raw = r.get("generation_raw_responses", [])
    if not qg_prompts and not qg_raw:
        qg_status = r.get("generation_status", "not_run")
        qg_reason = r.get("generation_reason", r.get("final_filter_reason", ""))
    elif r.get("generation_error", False):
        qg_status = r.get("generation_status", "error")
        qg_reason = r.get("generation_reason", r.get("grammar_reason", ""))
    else:
        qg_status = r.get("generation_status", "ok")
        qg_reason = r.get("generation_reason", "")

    trace.set_qg_generation(
        generator_model=r.get("method", ""),
        prompts=qg_prompts,
        raw_responses=qg_raw,
        parsed_question=r.get("generated_question", ""),
        retry_attempts=r.get("retry_attempts", 0),
        status=qg_status,
        reason=qg_reason,
    )

    # Quality filter
    trace.set_quality_filter(
        grammar_check={"pass": r.get("grammar_pass", False), "reason": r.get("grammar_reason", "")},
        weak_trigger_check={"pass": r.get("weak_trigger_pass", True), "type": r.get("weak_trigger_type", ""), "reason": r.get("weak_trigger_reason", "")},
        answer_phrase_check={"pass": r.get("answer_phrase_pass", False), "reason": r.get("answer_phrase_reason", ""), "raw": r.get("answer_phrase_raw", "")},
        consistency_judge={"label": r.get("answer_consistency_label", ""), "reason": r.get("answer_consistency_reason", ""), "raw": r.get("consistency_judge_raw", [])},
        path_coverage={"pass": r.get("path_coverage_pass", False), "count": r.get("path_coverage_count", 0), "reason": r.get("path_coverage_reason", ""), "raw": r.get("path_coverage_raw", "")},
        asks_target_event=r.get("asks_target_event"),
        hard_degraded={"is_degraded": r.get("hard_degraded", False), "reason": r.get("hard_degraded_reason", ""), "raw": r.get("hard_degraded_raw", "")},
        filter_pass=r.get("final_filter_pass", False),
        filter_reason=r.get("final_filter_reason", ""),
    )

    # Solver eval
    trace.set_solver_eval(
        status=r.get("solver_eval_status", "not_run"),
        reason=r.get("solver_eval_reason", ""),
        solver_result=r.get("solver_answer", ""),
        solver_confidence=r.get("solver_confidence", 0.0),
        judge_answerable=r.get("judge_answerable"),
        judge_solver_correct=r.get("judge_solver_correct"),
        judge_support_covered=r.get("judge_support_covered"),
        quality_fluency=r.get("quality_fluency"),
        quality_path_relevance=r.get("quality_path_relevance"),
        quality_difficulty_alignment=r.get("quality_difficulty_alignment"),
        composite=r.get("composite"),
    )

    return trace
