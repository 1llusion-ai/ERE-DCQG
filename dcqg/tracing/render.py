from pathlib import Path
from datetime import datetime

from dcqg.tracing.record import TraceRecord


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
