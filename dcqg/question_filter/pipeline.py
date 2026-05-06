"""
Quality filter pipeline: full chain for evaluating a single generated question.

- apply_final_filter: final pass/fail logic on a fully-evaluated record
- quality_filter_pipeline: run a single record through all filters
- _fill_remaining_fields: fill defaults when early-exiting the pipeline
- apply_quality_filter_with_judges: post-judge quality-only filter (primary evaluation gate)
"""
from dcqg.question_filter.grammar import enhanced_grammar_filter, check_weak_trigger
from dcqg.question_filter.consistency import extract_gold_answer_phrase, answer_event_consistency_judge
from dcqg.question_filter.path_coverage import path_coverage_judge
from dcqg.question_filter.shortcut import hard_degraded_check
from dcqg.question_filter.hard_implicitness import hard_implicitness_check
import re as _re

# Double-question detection patterns (duplicated from dcqg/generation/generator.py to avoid circular import)
_DOUBLE_Q_PATTERNS = [
    _re.compile(r'(?i)^How\s+did\s+.+,\s*and\s+what\b'),
    _re.compile(r'(?i)^Why\s+did\s+.+,\s*(ultimately|and)\b'),
    _re.compile(r'(?i)^[Ww]hat\s+led\s+to\s+.+,\s*and\s+how\b'),
    _re.compile(r'(?i)^How\s+did\s+.+,\s*and\s+how\b'),
    _re.compile(r'(?i)^What\s+.+,\s*and\s+how\s+did\b'),
    _re.compile(r'(?i)^What\s+.+,\s*and\s+what\b'),
    _re.compile(r'(?i)^How\s+.+,\s*and\s+why\b'),
    _re.compile(r'(?i)\band\s+(what|how|why|when|where|who)\b.*\?'),
]


# ═══════════════════════════════════════════════════════════════
# FINAL FILTER LOGIC
# ═══════════════════════════════════════════════════════════════

def apply_final_filter(record):
    """
    Apply final filter logic to a fully-evaluated record.
    Returns (pass: bool, reason: str).
    """
    reasons = []

    # Grammar
    if not record.get("grammar_pass", False):
        reasons.append(f"grammar={record.get('grammar_reason', '?')}")

    # Weak trigger
    if not record.get("weak_trigger_pass", False):
        reasons.append(f"weak_trigger={record.get('weak_trigger_reason', '?')}")

    # Answer phrase
    if not record.get("answer_phrase_pass", False):
        reasons.append(f"answer_phrase={record.get('answer_phrase_reason', '?')}")

    # Answer consistency (judge_error is excluded from filter, not a fail)
    label = record.get("answer_consistency_label", "no")
    if label == "judge_error":
        pass  # don't fail on judge errors — mark separately
    elif label not in ("yes", "partial"):
        reasons.append(f"answer_consistency={label}: {record.get('answer_consistency_reason', '?')}")

    # Path coverage
    if not record.get("path_coverage_pass", False):
        reasons.append(f"path_coverage={record.get('path_coverage_reason', '?')}")

    # Hard shortcut (only for Hard difficulty)
    # fail only if shortcut_without_path=yes AND needs_prior=no
    if record.get("difficulty") == "Hard" and record.get("hard_degraded", False):
        reasons.append(f"hard_shortcut={record.get('hard_degraded_reason', '?')}")

    # Hard implicitness: fail if too many prior triggers explicitly in question
    if record.get("difficulty") == "Hard" and not record.get("hard_implicit_chain_pass", True):
        reasons.append(f"hard_implicit={record.get('hard_implicit_chain_reason', '?')}")

    if not reasons:
        return True, "all checks passed"
    return False, "; ".join(reasons)


# ═══════════════════════════════════════════════════════════════
# FULL PIPELINE: evaluate one sample through all filters
# ═══════════════════════════════════════════════════════════════

def quality_filter_pipeline(record, skip_llm=False):
    """
    Run a single record through the full quality filter pipeline.
    Modifies record in-place and returns it.
    """
    q = record.get("generated_question", "")
    events = record.get("events", [])
    difficulty = record.get("difficulty", "Easy")
    gold_trigger = record.get("gold_answer_trigger", "")
    supporting = record.get("supporting_sentences", [])

    # ── 1. Enhanced grammar filter ──
    g_pass, g_reason = enhanced_grammar_filter(q, events)
    record["grammar_pass"] = g_pass
    record["grammar_reason"] = g_reason

    # Word count for analysis
    record["question_word_count"] = len(q.split()) if q else 0

    if not g_pass:
        # Still fill in remaining fields for analysis
        _fill_remaining_fields(record, skip_llm=True)
        return record

    # ── 2. Find answer sentence and event info ──
    answer_event_id = record.get("answer_event_id", "")
    answer_event = None
    for e in events:
        if e["id"] == answer_event_id:
            answer_event = e
            break
    if not answer_event and events:
        answer_event = events[-1]
        answer_event_id = answer_event["id"]

    answer_sentence = ""
    answer_event_type = ""
    if answer_event:
        answer_event_type = answer_event.get("type", "")
        sent_id = answer_event.get("sent_id", -1)
        for s in supporting:
            sid = s[0] if isinstance(s, (list, tuple)) else supporting.index(s)
            if sid == sent_id:
                answer_sentence = s[1] if isinstance(s, (list, tuple)) else s
                break

    record["answer_event_id"] = answer_event_id

    # ── 3. Gold answer phrase ──
    upstream_phrase = record.get("gold_answer_phrase", "")
    upstream_pass = record.get("answer_phrase_pass", None)
    upstream_reason = record.get("answer_phrase_reason", "")
    if skip_llm:
        phrase = upstream_phrase or gold_trigger
        a_type = "invalid"
        a_pass = upstream_pass if upstream_pass is not None else True
        a_reason = upstream_reason or "skipped LLM"
        a_raw = ""
    else:
        llm_phrase, a_type, llm_pass, llm_reason, a_raw = extract_gold_answer_phrase(
            answer_sentence, gold_trigger, answer_event_type
        )
        record["llm_answer_phrase"] = llm_phrase
        record["llm_answer_phrase_pass"] = llm_pass
        record["llm_answer_phrase_reason"] = llm_reason
        if upstream_phrase:
            phrase = upstream_phrase
            a_pass = upstream_pass if upstream_pass is not None else True
            a_reason = upstream_reason or "upstream_answer_phrase"
        else:
            phrase = llm_phrase
            a_pass = llm_pass
            a_reason = llm_reason

    record["gold_answer_phrase"] = phrase
    record["answer_type"] = a_type
    record["gold_answer_sentence"] = answer_sentence
    record["answer_phrase_pass"] = a_pass
    record["answer_phrase_reason"] = a_reason
    record["answer_phrase_raw"] = a_raw

    # ── 4. Weak trigger ──
    wt_result = check_weak_trigger(gold_trigger, phrase)
    record.update(wt_result)

    # If weak trigger check fails, still continue for analysis
    # but skip expensive LLM calls
    if not wt_result["weak_trigger_pass"] and skip_llm:
        _fill_remaining_fields(record, skip_llm=True)
        return record

    if skip_llm:
        _fill_remaining_fields(record, skip_llm=True)
        return record

    # ── 5. Answer-event consistency judge ──
    cons_result = answer_event_consistency_judge(
        q, supporting, events,
        answer_event_id, gold_trigger, phrase, answer_sentence
    )
    record["expected_answer_type"] = cons_result["expected_answer_type"]
    record["expected_answer_summary"] = cons_result["expected_answer_summary"]
    record["answer_consistency_label"] = cons_result["answer_consistency"]
    record["answer_consistency_reason"] = cons_result["answer_consistency_reason"]
    record["answer_consistency_pass"] = cons_result["answer_consistency"] in ("yes", "partial", "judge_error")
    record["asks_target_event"] = cons_result.get("asks_target_event")
    record["judge_answerable"] = cons_result.get("judge_answerable")
    record["consistency_judge_raw"] = cons_result.get("judge_raw_responses", [])

    # ── 6. Path coverage ──
    cov_result = path_coverage_judge(q, supporting, events, difficulty)
    record["path_coverage_count"] = cov_result["path_coverage_count"]
    record["path_coverage_prior_count"] = cov_result["path_coverage_prior_count"]
    record["path_coverage_all_count"] = cov_result["path_coverage_all_count"]
    record["path_covered_events"] = cov_result["path_covered_events"]
    record["path_covered_prior_events"] = cov_result["path_covered_prior_events"]
    record["path_coverage_pass"] = cov_result["path_coverage_pass"]
    record["path_coverage_reason"] = cov_result["path_coverage_reason"]
    record["path_coverage_method"] = cov_result["path_coverage_method"]
    record["path_coverage_details"] = cov_result["path_coverage_details"]
    record["path_coverage_raw"] = cov_result["path_coverage_raw"]

    # ── 7. Hard shortcut / degraded ──
    if difficulty == "Hard":
        hd_result = hard_degraded_check(q, supporting, phrase, events)
        record.update(hd_result)
    else:
        record["shortcut_without_path"] = "N/A"
        record["needs_prior_events_to_identify_answer"] = "N/A"
        record["shortcut_sentence_id"] = "N/A"
        record["shortcut_reason"] = ""
        record["can_answer_from_single_sentence"] = "N/A"
        record["single_sentence_id"] = "N/A"
        record["need_intermediate_events"] = "N/A"
        record["evidence_hops_used"] = 0
        record["hard_degraded"] = False
        record["hard_degraded_reason"] = "not Hard"
        record["hard_degraded_raw"] = ""

    # ── 7b. Hard implicitness check ──
    implicit_result = hard_implicitness_check(q, events, difficulty)
    record.update(implicit_result)

    # ── 8. Final filter ──
    final_pass, final_reason = apply_final_filter(record)
    record["final_filter_pass"] = final_pass
    record["final_filter_reason"] = final_reason

    return record


def _fill_remaining_fields(record, skip_llm=True):
    """Fill remaining fields when early-exit from pipeline."""
    defaults = {
        "gold_answer_phrase": record.get("gold_answer_trigger", ""),
        "answer_type": "invalid",
        "gold_answer_sentence": "",
        "answer_phrase_pass": False,
        "answer_phrase_reason": "skipped (early exit)",
        "answer_phrase_raw": "",
        "weak_trigger_flag": False,
        "weak_trigger_type": "none",
        "weak_trigger_pass": True,
        "weak_trigger_reason": "not checked",
        "expected_answer_type": "unknown",
        "expected_answer_summary": "",
        "answer_consistency_label": "no",
        "answer_consistency_reason": "skipped (early exit)",
        "answer_consistency_pass": False,
        "asks_target_event": None,
        "judge_answerable": None,
        "consistency_judge_raw": [],
        "path_coverage_count": 0,
        "path_coverage_prior_count": 0,
        "path_coverage_all_count": 0,
        "path_covered_events": [],
        "path_covered_prior_events": [],
        "path_coverage_pass": False,
        "path_coverage_reason": "skipped (early exit)",
        "path_coverage_method": "not_checked",
        "path_coverage_details": [],
        "path_coverage_raw": "",
        # New shortcut fields
        "shortcut_without_path": "N/A",
        "needs_prior_events_to_identify_answer": "N/A",
        "shortcut_sentence_id": "N/A",
        "shortcut_reason": "",
        # Old fields (backward compat)
        "can_answer_from_single_sentence": "N/A",
        "single_sentence_id": "N/A",
        "need_intermediate_events": "N/A",
        "evidence_hops_used": 0,
        "hard_degraded": False,
        "hard_degraded_reason": "not checked",
        "hard_degraded_raw": "",
        "hard_explicit_prior_count": 0,
        "hard_implicit_chain_pass": True,
        "hard_implicit_chain_reason": "not checked",
    }
    for k, v in defaults.items():
        if k not in record:
            record[k] = v

    # Final filter
    final_pass, final_reason = apply_final_filter(record)
    record["final_filter_pass"] = final_pass
    record["final_filter_reason"] = final_reason


# ═══════════════════════════════════════════════════════════════
# DEPRECATED HARD FILTERS: strict and relaxed variants
# Kept for backward compatibility. Use apply_quality_filter_with_judges instead.
# These functions use difficulty-level metrics as pass/fail gates, which is
# no longer the desired behavior for main evaluation set filtering.
# ═══════════════════════════════════════════════════════════════

# DEPRECATED: kept for backward compatibility. Use apply_quality_filter_with_judges instead.
def _common_hard_checks(record):
    """Shared checks for both strict and relaxed Hard filters.
    Returns list of failure reasons (empty = pass).
    """
    reasons = []

    # Grammar
    if not record.get("grammar_pass", False):
        reasons.append(f"grammar={record.get('grammar_reason', '?')}")

    # Weak trigger
    if not record.get("weak_trigger_pass", True):
        reasons.append(f"weak_trigger={record.get('weak_trigger_reason', '?')}")

    # Answer phrase
    if not record.get("answer_phrase_pass", True):
        reasons.append(f"answer_phrase={record.get('answer_phrase_reason', '?')}")

    # Hard implicitness
    if not record.get("hard_implicit_chain_pass", True):
        reasons.append(f"hard_implicit={record.get('hard_implicit_chain_reason', '?')}")

    # Trigger leakage
    question = record.get("generated_question", "")
    gold_trigger = record.get("gold_answer_trigger", "")
    if gold_trigger and question and gold_trigger.lower() in question.lower():
        reasons.append("trigger_leakage")

    # Blind difficulty judge: must be Hard
    blind = record.get("blind_difficulty_judge", {})
    blind_pred = blind.get("predicted_difficulty", "Medium")
    if blind_pred != "Hard":
        reasons.append(f"blind_pred={blind_pred}")

    # Blind judge: answerable
    blind_ans = blind.get("answerable", "partial")
    if blind_ans not in ("yes", "partial"):
        reasons.append(f"blind_answerable={blind_ans}")

    # Blind judge: final_event_consistent (yes/partial required)
    blind_fec = blind.get("final_event_consistent", "partial")
    if blind_fec not in ("yes", "partial"):
        reasons.append(f"blind_fec={blind_fec}")

    # Blind judge: single_sentence_answerable must be no
    blind_ssa = blind.get("single_sentence_answerable", "partial")
    if blind_ssa != "no":
        reasons.append(f"single_sentence_answerable={blind_ssa}")

    # Path dependency: must be strong
    pd = record.get("path_dependency_judge", {})
    pd_level = pd.get("path_dependency", "partial")
    if pd_level != "strong":
        reasons.append(f"path_dependency={pd_level}")

    # Hard answer alignment
    align = record.get("hard_alignment", {})
    asks = align.get("asks_expected_answer", "partial")
    if asks not in ("yes", "partial"):
        reasons.append(f"alignment_asks={asks}")
    natural = align.get("expected_answer_natural", "partial")
    if natural not in ("yes", "partial"):
        reasons.append(f"alignment_natural={natural}")
    drift = align.get("target_drift", "no")
    if drift == "yes":
        reasons.append(f"target_drift=yes")

    return reasons


# DEPRECATED: kept for backward compatibility. Use apply_quality_filter_with_judges instead.
def apply_strict_hard_filter(record):
    """DEPRECATED: Strict Hard filter. Uses difficulty metrics as gates."""
    reasons = _common_hard_checks(record)

    # Answer consistency — strict: must be yes or partial
    label = record.get("answer_consistency_label", "no")
    if label not in ("yes", "partial"):
        reasons.append(f"answer_consistency={label}")

    if not reasons:
        record["strict_new_hard_filter_pass"] = True
        record["strict_new_hard_filter_reason"] = "all checks passed"
    else:
        record["strict_new_hard_filter_pass"] = False
        record["strict_new_hard_filter_reason"] = "; ".join(reasons)

    # Also set the old field for backward compat
    record["new_hard_filter_pass"] = record["strict_new_hard_filter_pass"]
    record["new_hard_filter_reason"] = record["strict_new_hard_filter_reason"]


# DEPRECATED: kept for backward compatibility. Use apply_quality_filter_with_judges instead.
def apply_relaxed_hard_filter(record):
    """DEPRECATED: Relaxed Hard filter. Uses difficulty metrics as gates."""
    reasons = _common_hard_checks(record)

    # Answer consistency — relaxed: allow "no" if FEC=yes and pathdep=strong
    label = record.get("answer_consistency_label", "no")
    if label not in ("yes", "partial"):
        blind = record.get("blind_difficulty_judge", {})
        blind_fec = blind.get("final_event_consistent", "partial")
        pd = record.get("path_dependency_judge", {})
        pd_level = pd.get("path_dependency", "partial")
        if not (label == "no" and blind_fec in ("yes",) and pd_level == "strong"):
            reasons.append(f"answer_consistency={label}")

    if not reasons:
        record["relaxed_new_hard_filter_pass"] = True
        record["relaxed_new_hard_filter_reason"] = "all checks passed"
    else:
        record["relaxed_new_hard_filter_pass"] = False
        record["relaxed_new_hard_filter_reason"] = "; ".join(reasons)


# ═══════════════════════════════════════════════════════════════
# QUALITY FILTER: post-judge quality-only filtering
# This is the primary filter for the main evaluation set.
# Difficulty-level metrics are reported but NEVER used as pass/fail gates.
# ═══════════════════════════════════════════════════════════════

def apply_quality_filter_with_judges(record):
    """
    Quality-only filter using validity/quality criteria.
    Sets quality_filter_pass (bool) and quality_filter_reason (str) on record.

    EXCLUDED from pass/fail (difficulty metrics, reported only):
    - blind_difficulty_judge.predicted_difficulty
    - blind_difficulty_judge.required_steps
    - blind_difficulty_judge.single_sentence_answerable
    - path_dependency_judge.path_dependency
    - shortcut_without_path
    - single_sentence_risk
    - llm_path_judge.recommended_difficulty
    """
    reasons = []

    # 1. generation_error must be false
    if record.get("generation_error"):
        reasons.append("generation_error")

    # 2. grammar_pass must be true
    if not record.get("grammar_pass", False):
        reasons.append(f"grammar={record.get('grammar_reason', '?')}")

    # 3. generated_question must exist and be non-empty
    question = record.get("generated_question", "")
    if not question or not question.strip():
        reasons.append("empty_question")

    # 4. No double question
    if question:
        for pat in _DOUBLE_Q_PATTERNS:
            if pat.search(question):
                reasons.append("double_question")
                break

    # 5. No trigger leakage
    gold_trigger = record.get("gold_answer_trigger", "")
    if gold_trigger and question and gold_trigger.lower() in question.lower():
        reasons.append("trigger_leakage")

    # 6. No answer leakage (gold_answer_phrase not in question, if phrase != trigger)
    gold_phrase = record.get("gold_answer_phrase", "")
    if (gold_phrase and question
            and gold_phrase.lower() != gold_trigger.lower()
            and gold_phrase.lower() in question.lower()):
        reasons.append("answer_leakage")

    # 7. answer_phrase_pass must be true
    if not record.get("answer_phrase_pass", True):
        reasons.append(f"answer_phrase={record.get('answer_phrase_reason', '?')}")

    # 8. answer_type must not be "invalid"
    answer_type = record.get("answer_type", "")
    if answer_type == "invalid":
        reasons.append("answer_type=invalid")

    # ── Judge-dependent checks ──

    blind = record.get("blind_difficulty_judge", {})
    blind_status = record.get("blind_difficulty_judge_status", "skipped")

    # 9. blind judge must have run
    if blind_status != "ok":
        reasons.append(f"blind_judge_status={blind_status}")
    else:
        # 10. answerable must be yes/partial
        blind_ans = blind.get("answerable", "no")
        if blind_ans not in ("yes", "partial"):
            reasons.append(f"answerable={blind_ans}")

        # 11. final_event_consistent must be yes/partial
        blind_fec = blind.get("final_event_consistent", "no")
        if blind_fec not in ("yes", "partial"):
            reasons.append(f"fec={blind_fec}")

    # 12. alignment: asks_expected_answer must be yes/partial
    align = record.get("hard_alignment", {})
    align_status = record.get("hard_alignment_status", "skipped")
    if align_status != "ok":
        reasons.append(f"alignment_status={align_status}")
    else:
        asks = align.get("asks_expected_answer", "no")
        if asks not in ("yes", "partial"):
            reasons.append(f"alignment_asks={asks}")

        # 13. target_drift must NOT be "yes"
        drift = align.get("target_drift", "no")
        if drift == "yes":
            reasons.append("target_drift=yes")

    # 14. answer_consistency_label must not be "no" (judge_error passes)
    label = record.get("answer_consistency_label", "no")
    if label == "judge_error":
        pass
    elif label == "no":
        reasons.append("answer_consistency=no")

    # Set result
    if not reasons:
        record["quality_filter_pass"] = True
        record["quality_filter_reason"] = "all checks passed"
    else:
        record["quality_filter_pass"] = False
        record["quality_filter_reason"] = "; ".join(reasons)
