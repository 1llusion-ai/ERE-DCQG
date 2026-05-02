"""
Stage 1: Rule-based Path Prefilter (diagnostic / prefilter logic).

Cleans event paths AFTER sampling but BEFORE question generation.
Reduces downstream generation and judge failures by filtering paths
with weak triggers, missing answer phrases, or unfavorable compositions.

No LLM calls -- pure rules.
"""
from dcqg.path.answer_extraction import (
    extract_answer_phrase_local,
    enrich_path_item,
    GENERIC_TRIGGERS,
    WEAK_TRIGGERS,
    _simple_stem,
)
from dcqg.question_filter.grammar import (
    HARD_BLACKLIST_TRIGGERS,
    WEAK_TRIGGER_NEEDS_PHRASE,
    check_weak_trigger,
)


# ================================================================
# COMBINED WEAK TRIGGER SETS
# ================================================================

# Hard blacklist: always fail unless valid answer phrase
HARD_WEAK_TRIGGERS = HARD_BLACKLIST_TRIGGERS | GENERIC_TRIGGERS | {
    "become", "began", "ended", "continued", "included",
    "showed", "found", "called", "seen",
}

# Soft weak triggers: flag but allow if answer phrase is good
SOFT_WEAK_TRIGGERS = WEAK_TRIGGERS | WEAK_TRIGGER_NEEDS_PHRASE


# ================================================================
# RELATION ANALYSIS
# ================================================================

def classify_relations(relation_subtypes):
    """Classify relation composition.
    Returns (non_temporal_count, relation_group).
    """
    if not relation_subtypes:
        return 0, "NONE"

    non_temporal = 0
    has_cause = False
    has_subevent = False

    for r in relation_subtypes:
        r_upper = r.upper()
        if not r_upper.startswith("TEMPORAL"):
            non_temporal += 1
        if "CAUSE" in r_upper:
            has_cause = True
        if "SUBEVENT" in r_upper:
            has_subevent = True

    if has_cause and has_subevent:
        group = "MIXED"
    elif has_cause:
        group = "CAUSE"
    elif has_subevent:
        group = "SUBEVENT"
    elif non_temporal > 0:
        group = "MIXED"
    else:
        group = "TEMPORAL"

    return non_temporal, group


# ================================================================
# SUPPORT SPAN ANALYSIS
# ================================================================

def analyze_support_span(events, supporting_sentences):
    """Compute support span and single-sentence risk.
    Returns (support_span, single_sentence_risk).
    """
    support_span = len(supporting_sentences) if supporting_sentences else 0

    # Check if all path events are in the same sentence
    sent_ids = set()
    for e in events:
        sid = e.get("sent_id", -1)
        if sid >= 0:
            sent_ids.add(sid)

    all_same_sentence = len(sent_ids) <= 1 and len(events) > 1

    # Determine risk
    if support_span <= 2:
        risk = "high"
    elif all_same_sentence:
        risk = "high"
    elif support_span <= 3:
        risk = "medium"
    else:
        risk = "low"

    return support_span, risk


# ================================================================
# MAIN PREFILTER
# ================================================================

def prefilter_path(item):
    """Apply all prefilter rules to a single path item.
    Returns the item with prefilter fields added.
    """
    item = dict(item)  # shallow copy
    events = item.get("events", [])
    difficulty = item.get("difficulty", "Easy")
    supporting = item.get("supporting_sentences", [])
    relation_subtypes = item.get("relation_subtypes", [])

    # Enrich with answer phrase data
    item = enrich_path_item(item)

    final_event = events[-1] if events else {}
    trigger = final_event.get("trigger", "")
    answer_phrase = item.get("gold_answer_phrase", "")
    answer_sentence = item.get("gold_answer_sentence", "")

    # -- 1. Answer phrase extraction & validation --
    answer_phrase_status = item.get("answer_phrase_status", "unknown")

    # -- 2. Weak trigger check --
    trigger_lower = trigger.lower().strip()
    wt_result = check_weak_trigger(trigger, answer_phrase)
    item["weak_trigger_type"] = wt_result["weak_trigger_type"]
    item["weak_trigger_pass"] = wt_result["weak_trigger_pass"]
    item["weak_trigger_reason"] = wt_result["weak_trigger_reason"]

    # -- 3. Relation composition --
    non_temporal_count, relation_group = classify_relations(relation_subtypes)
    item["non_temporal_count"] = non_temporal_count
    item["relation_group"] = relation_group

    # -- 4. Support span --
    support_span, single_sentence_risk = analyze_support_span(events, supporting)
    item["support_span"] = support_span
    item["rule_single_sentence_risk"] = single_sentence_risk

    # -- 5. Pass / fail logic --
    reasons = []

    # Must-fail: hard blacklisted trigger (always, regardless of phrase)
    if trigger_lower in HARD_WEAK_TRIGGERS:
        reasons.append(f"hard_weak_trigger='{trigger}'")

    # Must-fail: answer phrase not extractable
    # (validate_answer_phrase is in selector; here we use answer_phrase_status)
    from dcqg.path.selector import validate_answer_phrase
    ap_pass, ap_reason = validate_answer_phrase(answer_phrase, trigger, answer_phrase_status)
    item["answer_phrase_pass"] = ap_pass
    item["answer_phrase_reason"] = ap_reason

    if not ap_pass:
        reasons.append(f"answer_phrase_fail: {ap_reason}")

    # Must-fail: soft weak trigger without valid phrase
    if trigger_lower in SOFT_WEAK_TRIGGERS and not ap_pass:
        reasons.append(f"soft_weak_trigger='{trigger}' with no valid phrase")

    # High risk markers (not fail, just flag)
    risk_flags = []
    if difficulty == "Hard" and relation_group == "TEMPORAL":
        risk_flags.append("temporal_only_hard")
    if difficulty in ("Medium", "Hard") and single_sentence_risk == "high":
        risk_flags.append("single_sentence_risk_high")

    prefilter_pass = len(reasons) == 0
    prefilter_reason = "; ".join(reasons) if reasons else "pass"
    if risk_flags:
        prefilter_reason += f" [risk: {', '.join(risk_flags)}]"

    item["prefilter_pass"] = prefilter_pass
    item["prefilter_reason"] = prefilter_reason

    return item
