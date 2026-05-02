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
    For Hard questions: check if single sentence can answer it.
    Returns dict with all output fields.
    """
    path_str = " -> ".join(f'"{e["trigger"]}"' for e in path_events)
    ctx = "\n".join(
        f"[S{s[0]}] {s[1]}" if isinstance(s, (list, tuple)) else f"[S{i}] {s}"
        for i, s in enumerate(supporting_sentences[:8])
    )

    prompt = f"""Context:
{ctx}

Question: "{generated_question}"
Path events: {path_str}
Gold answer phrase: "{gold_answer_phrase}"

For this Hard-level question, answer:

1. Can this question be correctly answered by reading only ONE sentence from the context? (yes/partial/no)
2. If yes/partial, which sentence? (give the sentence number, e.g., S0, S1, etc.)
3. Does answering require understanding intermediate events in the path? (yes/partial/no)
4. How many evidence hops are needed? (1/2/3+)

Reply:
SINGLE=<yes|partial|no>
SENT_ID=<S# or N/A>
NEED=<yes|partial|no>
HOPS=<1|2|3+>"""

    resp = call_api(prompt, temperature=0.0, max_tokens=40, timeout=90)

    can_single = "no"
    sent_id = None
    need_intermediate = "yes"
    hops = "2"

    if resp:
        resp_upper = resp.upper().replace(",", " ")
        for part in resp_upper.split():
            if part.startswith("SINGLE="):
                val = part.split("=", 1)[1].strip().lower()
                if val in ("yes", "partial", "no"):
                    can_single = val
            elif part.startswith("SENT_ID="):
                val = part.split("=", 1)[1].strip()
                if val.startswith("S") and val[1:].isdigit():
                    sent_id = val
            elif part.startswith("NEED="):
                val = part.split("=", 1)[1].strip().lower()
                if val in ("yes", "partial", "no"):
                    need_intermediate = val
            elif part.startswith("HOPS="):
                val = part.split("=", 1)[1].strip()
                if val in ("1", "2", "3+"):
                    hops = val

    # Determine degraded
    degraded = False
    reason = "not degraded"

    if can_single == "yes":
        degraded = True
        reason = f"can_answer_from_single_sentence=yes (sent={sent_id})"
    elif need_intermediate == "no":
        degraded = True
        reason = "need_intermediate_events=no"

    # Map hops to numeric
    hops_num = {"1": 1, "2": 2, "3+": 3}.get(hops, 2)

    return {
        "can_answer_from_single_sentence": can_single,
        "single_sentence_id": sent_id or "N/A",
        "need_intermediate_events": need_intermediate,
        "evidence_hops_used": hops_num,
        "hard_degraded": degraded,
        "hard_degraded_reason": reason,
        "hard_degraded_raw": resp or "",
    }
