"""
Shortcut and degradation detection for generated questions.

- hard_degraded_check: LLM-based check for Hard questions answerable from a single sentence
- check_banned_phrases: regex check for banned shortcut phrases in Hard questions
"""
import re

from dcqg.utils.api_client import call_api


# ═══════════════════════════════════════════════════════════════
# BANNED SHORTCUT PHRASES (Hard only)
# ═══════════════════════════════════════════════════════════════

BANNED_PATTERNS_HARD = [
    r"(?i)what\s+was\s+the\s+final\s+outcome",
    r"(?i)what\s+action\s+was\s+taken",
    r"(?i)what\s+happened\s+after\s+the\s+incident",
    r"(?i)what\s+happened\s+after\s+the\s+event",
    r"(?i)what\s+happened\s+as\s+a\s+result",
    r"(?i)what\s+was\s+the\s+result",
    r"(?i)what\s+did\s+\w+\s+do\s+after\s+the\s+(incident|event|crash|accident|disaster)",
    r"(?i)following\s+the\s+(incident|event|occurrence)",
]


def check_banned_phrases(question):
    """Return (has_banned, matched_pattern) for Hard shortcut detection."""
    for pat in BANNED_PATTERNS_HARD:
        m = re.search(pat, question)
        if m:
            return True, m.group(0)
    # Also check: question uses "after X" where X is a single, vague event reference
    after_match = re.search(r'(?i)after\s+the\s+(\w+)\s*[?,]?\s*$', question)
    if after_match:
        vague_word = after_match.group(1).lower()
        if vague_word in {'incident', 'event', 'crash', 'accident', 'disaster', 'attack', 'battle', 'war', 'conflict'}:
            return True, f"vague after-reference: '{vague_word}'"
    return False, ""


# ═══════════════════════════════════════════════════════════════
# HARD DEGRADED CHECK
# ═══════════════════════════════════════════════════════════════

def hard_degraded_check(
    generated_question, supporting_sentences, gold_answer_phrase, path_events
):
    """
    For Hard questions: check if the question is a shortcut that doesn't
    actually require path knowledge.

    NEW LOGIC: Instead of rejecting because "answer is in one sentence",
    we check whether the QUESTION depends on prior path events to locate
    the answer. A question can have its answer in one sentence AND still
    require multi-hop reasoning to identify that sentence.

    Returns dict with all output fields.
    """
    path_str = " -> ".join(f'"{e["trigger"]}"' for e in path_events)
    prior_triggers = [e["trigger"] for e in path_events[:-1]]
    prior_list = ", ".join(f'"{t}"' for t in prior_triggers)
    ctx = "\n".join(
        f"[S{s[0]}] {s[1]}" if isinstance(s, (list, tuple)) else f"[S{i}] {s}"
        for i, s in enumerate(supporting_sentences[:8])
    )

    prompt = f"""Context:
{ctx}

Question: "{generated_question}"
Path events: {path_str}
Prior events (before the final answer): {prior_list}
Gold answer phrase: "{gold_answer_phrase}"

For this Hard-level question, answer:

1. shortcut_without_path: Could someone find the correct answer WITHOUT knowing about the prior events? Would the question make sense and be answerable if the prior events were removed from context? (yes/partial/no)

2. needs_prior_events_to_identify_answer: Does the question require understanding the prior events to LOCATE or IDENTIFY which sentence contains the answer? (yes/partial/no)

3. If shortcut_without_path=yes, which single sentence contains the answer? (S# or N/A)

Reply ONLY as JSON:
{{"shortcut_without_path":"no","needs_prior_events_to_identify_answer":"yes","shortcut_sentence_id":"N/A","reason":"brief explanation"}}"""

    # Initialize defaults
    shortcut_without_path = "no"
    needs_prior = "yes"
    sent_id = "N/A"
    reason_text = ""

    resp = call_api(prompt, temperature=0.0, max_tokens=80, timeout=90)

    if resp:
        parsed = _parse_shortcut_response(resp)
        if parsed:
            shortcut_without_path = parsed.get("shortcut_without_path", "no")
            needs_prior = parsed.get("needs_prior_events_to_identify_answer", "yes")
            sent_id = parsed.get("shortcut_sentence_id", "N/A")
            reason_text = parsed.get("reason", "")

    # Also keep old fields for backward compat (informational only)
    can_single = shortcut_without_path  # approximate mapping

    # NEW degradation logic: fail only if shortcut=yes AND needs_prior=no
    degraded = (shortcut_without_path == "yes" and needs_prior == "no")

    if degraded:
        deg_reason = f"shortcut_without_path=yes, needs_prior=no (sent={sent_id}): {reason_text}"
    else:
        deg_reason = f"not degraded: shortcut={shortcut_without_path}, needs_prior={needs_prior}"

    # Map hops (keep for backward compat)
    hops_num = 1 if shortcut_without_path == "yes" else (2 if shortcut_without_path == "partial" else 3)

    return {
        # New fields (used for filtering)
        "shortcut_without_path": shortcut_without_path,
        "needs_prior_events_to_identify_answer": needs_prior,
        "shortcut_sentence_id": sent_id,
        "shortcut_reason": reason_text,
        # Old fields (backward compat, informational only)
        "can_answer_from_single_sentence": can_single,
        "single_sentence_id": sent_id,
        "need_intermediate_events": "no" if shortcut_without_path == "yes" else "yes",
        "evidence_hops_used": hops_num,
        # Degradation decision
        "hard_degraded": degraded,
        "hard_degraded_reason": deg_reason,
        "hard_degraded_raw": resp or "",
    }


def _parse_shortcut_response(resp):
    """Parse shortcut judge response with robust JSON extraction.
    Always returns a dict or None — never a bare string/number."""
    import json
    # Try direct JSON parse (must be a dict, not a bare string/number)
    try:
        parsed = json.loads(resp)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try extracting first {...}
    try:
        s_idx = resp.index("{")
        e_idx = resp.rindex("}") + 1
        parsed = json.loads(resp[s_idx:e_idx])
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, json.JSONDecodeError):
        pass

    # Try fixing common JSON errors
    cleaned = resp.strip()
    # Remove markdown code fences
    import re
    cleaned = re.sub(r'^```(?:json)?', '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'```$', '', cleaned).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try extracting key-value pairs from text
    result = {}
    for key in ["shortcut_without_path", "needs_prior_events_to_identify_answer",
                 "shortcut_sentence_id", "reason"]:
        m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', resp)
        if m:
            result[key] = m.group(1)
    if result:
        return result

    # Try KEY=value format
    resp_upper = resp.upper().replace(",", " ")
    for part in resp_upper.split():
        if part.startswith("SHORTCUT="):
            val = part.split("=", 1)[1].strip().lower()
            if val in ("yes", "partial", "no"):
                result["shortcut_without_path"] = val
        elif part.startswith("NEED="):
            val = part.split("=", 1)[1].strip().lower()
            if val in ("yes", "partial", "no"):
                result["needs_prior_events_to_identify_answer"] = val
        elif part.startswith("SENT_ID="):
            val = part.split("=", 1)[1].strip()
            if val.startswith("S") and val[1:].isdigit():
                result["shortcut_sentence_id"] = val

    return result if result else None
