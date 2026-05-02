"""
Path coverage checking: lexical and LLM-based.

- check_path_coverage_lexical: stem-based overlap between question and path events
- path_coverage_judge: LLM judge for semantic path event coverage (structured JSON)
"""
import json
import re

from dcqg.utils.text import simple_stem
from dcqg.utils.api_client import call_api


# ═══════════════════════════════════════════════════════════════
# LEXICAL PATH COVERAGE
# ═══════════════════════════════════════════════════════════════

def check_path_coverage_lexical(generated_question, path_events):
    """
    Lexical overlap check: how many path events are referenced in the question.
    Returns (covered_count, covered_event_ids).
    """
    q_lower = generated_question.lower()
    q_words = set(q_lower.split())
    q_stems = {simple_stem(w) for w in q_words}
    covered = []
    for e in path_events:
        trigger = e["trigger"].lower()
        trigger_stem = simple_stem(trigger)
        # Direct match
        if trigger in q_lower:
            covered.append(e["id"])
            continue
        # Stem match
        if trigger_stem in q_stems:
            covered.append(e["id"])
            continue
        # Substring containment
        matched = False
        for qw in q_words:
            if len(qw) >= 3 and len(trigger) >= 3:
                if trigger in qw or qw in trigger or trigger_stem in qw or qw in trigger_stem:
                    covered.append(e["id"])
                    matched = True
                    break
        if matched:
            continue
        # Entity/type match
        etype = e.get("type", "").lower()
        if etype and len(etype) >= 4 and etype in q_lower:
            covered.append(e["id"])
    return len(covered), covered


def _lexical_coverage_details(generated_question, path_events):
    """
    Lexical coverage with per-event details for fallback.
    Returns list of dicts with event_id, trigger, is_prior, covered, match_type.
    """
    q_lower = generated_question.lower()
    q_words = set(q_lower.split())
    q_stems = {simple_stem(w) for w in q_words}
    details = []
    for i, e in enumerate(path_events):
        trigger = e["trigger"].lower()
        trigger_stem = simple_stem(trigger)
        covered = False
        match_type = "not_covered"

        if trigger in q_lower:
            covered = True
            match_type = "exact"
        elif trigger_stem in q_stems:
            covered = True
            match_type = "lemma"
        else:
            for qw in q_words:
                if len(qw) >= 3 and len(trigger) >= 3:
                    if trigger in qw or qw in trigger or trigger_stem in qw or qw in trigger_stem:
                        covered = True
                        match_type = "lemma"
                        break
        if not covered:
            etype = e.get("type", "").lower()
            if etype and len(etype) >= 4 and etype in q_lower:
                covered = True
                match_type = "paraphrase"

        details.append({
            "event_id": e["id"],
            "trigger": e["trigger"],
            "is_prior": i < len(path_events) - 1,
            "covered": covered,
            "evidence": "",
            "match_type": match_type,
        })
    return details


# ═══════════════════════════════════════════════════════════════
# LLM PATH COVERAGE JUDGE (structured JSON)
# ═══════════════════════════════════════════════════════════════

def path_coverage_judge(generated_question, supporting_sentences, path_events, difficulty):
    """
    LLM judge for path coverage: how many PRIOR events does the question reference?

    Returns structured per-event coverage with:
    - path_coverage_count (prior count for Medium/Hard, all count for Easy)
    - path_coverage_prior_count
    - path_coverage_all_count
    - path_covered_events (all covered event IDs)
    - path_covered_prior_events (covered prior event IDs)
    - path_coverage_pass
    - path_coverage_reason
    - path_coverage_method
    - path_coverage_raw (raw LLM response)
    """
    # Build event list with IDs for the prompt
    event_lines = []
    for i, e in enumerate(path_events):
        is_prior = i < len(path_events) - 1
        role = "PRIOR" if is_prior else "FINAL"
        event_lines.append(
            f'  {i+1}. id={e["id"]}  trigger="{e["trigger"]}"  type={e.get("type","?")}  role={role}'
        )
    event_list = "\n".join(event_lines)

    ctx = "\n".join(
        f"[S{s[0]}] {s[1]}" if isinstance(s, (list, tuple)) else f"[S{i}] {s}"
        for i, s in enumerate(supporting_sentences[:6])
    )

    # Compact event list for the prompt
    event_specs = []
    for i, e in enumerate(path_events):
        is_prior = i < len(path_events) - 1
        event_specs.append(f'{e["id"]}|{e["trigger"]}|prior={is_prior}')
    event_spec_str = " ; ".join(event_specs)

    prompt = f"""Question: "{generated_question}"
Events: {event_spec_str}

For each event, does the question reference it? Counts as covered: exact word, stem/lemma, paraphrase, or semantic reference.
Reply ONLY as JSON array, one per event:
[{{"id":"...","trigger":"...","prior":true,"covered":true,"evidence":"quote","match":"exact|lemma|paraphrase|not_covered"}},...]"""

    resp = call_api(prompt, temperature=0.0, max_tokens=300, timeout=90)

    # Parse structured response
    parsed = _parse_coverage_json(resp) if resp else None
    method = "llm_structured"

    # Normalize parsed result: accept both array and {covered_events: [...]} formats
    llm_events = None
    if isinstance(parsed, list):
        llm_events = parsed
    elif isinstance(parsed, dict):
        llm_events = parsed.get("covered_events")

    if llm_events and isinstance(llm_events, list):
        # Map by event_id (or id) for reliable lookup
        llm_map = {}
        for item in llm_events:
            if isinstance(item, dict):
                eid = item.get("event_id") or item.get("id")
                if eid:
                    llm_map[eid] = item

        # Build final details, filling from LLM output
        details = []
        for i, e in enumerate(path_events):
            is_prior = i < len(path_events) - 1
            llm_item = llm_map.get(e["id"])
            if llm_item and isinstance(llm_item, dict):
                details.append({
                    "event_id": e["id"],
                    "trigger": e["trigger"],
                    "is_prior": is_prior,
                    "covered": bool(llm_item.get("covered", False)),
                    "evidence": str(llm_item.get("evidence", "")),
                    "match_type": str(
                        llm_item.get("match_type") or llm_item.get("match") or "not_covered"
                    ),
                })
            else:
                # LLM didn't include this event — mark as not covered
                details.append({
                    "event_id": e["id"],
                    "trigger": e["trigger"],
                    "is_prior": is_prior,
                    "covered": False,
                    "evidence": "",
                    "match_type": "not_covered",
                })
    else:
        # LLM parse failed — fall back to lexical
        details = _lexical_coverage_details(generated_question, path_events)
        method = "lexical_fallback"

    # Also do lexical check; if lexical finds events that LLM missed, add them
    lex_count, lex_ids = check_path_coverage_lexical(generated_question, path_events)
    lex_set = set(lex_ids)
    for d in details:
        if not d["covered"] and d["event_id"] in lex_set:
            d["covered"] = True
            if d["match_type"] == "not_covered":
                d["match_type"] = "lemma"
                d["evidence"] = "(lexical fallback)"

    # Compute counts
    covered_all = [d for d in details if d["covered"]]
    covered_prior = [d for d in details if d["covered"] and d["is_prior"]]
    all_count = len(covered_all)
    prior_count = len(covered_prior)

    # Determine pass/fail by difficulty
    if difficulty == "Easy":
        min_required = 1
        coverage_count = all_count
        count_label = "all events"
    elif difficulty == "Medium":
        min_required = 1
        coverage_count = prior_count
        count_label = "prior events"
    else:  # Hard
        min_required = 2
        coverage_count = prior_count
        count_label = "prior events"

    passed = coverage_count >= min_required
    reason = (
        f"covers {coverage_count} {count_label}, need >= {min_required}"
        + (" [PASS]" if passed else " [FAIL]")
    )

    covered_event_ids = [d["event_id"] for d in details if d["covered"]]
    covered_prior_ids = [d["event_id"] for d in details if d["covered"] and d["is_prior"]]

    return {
        "path_coverage_count": coverage_count,
        "path_coverage_prior_count": prior_count,
        "path_coverage_all_count": all_count,
        "path_covered_events": covered_event_ids,
        "path_covered_prior_events": covered_prior_ids,
        "path_coverage_pass": passed,
        "path_coverage_reason": reason,
        "path_coverage_method": method,
        "path_coverage_details": details,
        "path_coverage_raw": resp or "",
    }


def _parse_coverage_json(resp):
    """Parse JSON from coverage judge response. Returns dict, list, or None."""
    # 1. Direct parse
    try:
        parsed = json.loads(resp)
        if isinstance(parsed, (dict, list)):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. Extract first {...} or [...]
    for open_char, close_char in [("{", "}"), ("[", "]")]:
        try:
            s_idx = resp.index(open_char)
            e_idx = resp.rindex(close_char) + 1
            parsed = json.loads(resp[s_idx:e_idx])
            if isinstance(parsed, (dict, list)):
                return parsed
        except (ValueError, json.JSONDecodeError):
            pass

    # 3. Fix common JSON errors
    cleaned = resp.strip()
    cleaned = re.sub(r'^```(?:json)?', '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'```$', '', cleaned).strip()
    # Fix missing quotes around keys
    cleaned = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', cleaned)
    # Fix single quotes
    cleaned = cleaned.replace("'", '"')
    # Fix trailing commas
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    # Fix duplicated quotes
    cleaned = re.sub(r'""+', '"', cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, (dict, list)):
            return parsed
    except json.JSONDecodeError:
        pass

    # 4. Try to extract just the covered_events array
    try:
        m = re.search(r'"covered_events"\s*:\s*\[', resp)
        if m:
            # Find matching ]
            bracket_start = m.end() - 1
            depth = 0
            for ci in range(bracket_start, len(resp)):
                if resp[ci] == '[':
                    depth += 1
                elif resp[ci] == ']':
                    depth -= 1
                    if depth == 0:
                        arr_str = resp[bracket_start:ci+1]
                        arr = json.loads(arr_str)
                        return {"covered_events": arr, "reason": ""}
    except (json.JSONDecodeError, ValueError):
        pass

    return None
