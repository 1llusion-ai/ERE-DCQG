"""
Full-chain debug trace for DCQG pipeline.
Shared by all pipeline stages. Each stage adds its own section to the trace record.

Usage:
    from dcqg.tracing import TraceRecord, write_full_trace, write_readable_trace

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
                          relation_distribution=""):
        self.data["path_sampling"] = {
            "difficulty": difficulty,
            "hop_count": hop_count,
            "path_events": path_events,
            "relation_subtypes": relation_subtypes,
            "relation_distribution": relation_distribution,
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
