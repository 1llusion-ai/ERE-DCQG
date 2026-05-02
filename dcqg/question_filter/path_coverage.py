"""
Path coverage checking: lexical and LLM-based.

- check_path_coverage_lexical: stem-based overlap between question and path events
- path_coverage_judge: LLM judge for semantic path event coverage
"""
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


# ═══════════════════════════════════════════════════════════════
# LLM PATH COVERAGE JUDGE
# ═══════════════════════════════════════════════════════════════

def path_coverage_judge(generated_question, supporting_sentences, path_events, difficulty):
    """
    LLM judge for path coverage: how many path events does the question actually use?
    Returns (coverage_count, covered_events, pass, reason).
    """
    path_str = " -> ".join(f'"{e["trigger"]}"' for e in path_events)
    event_list = "\n".join(
        f'  {i+1}. "{e["trigger"]}" (type={e.get("type","?")}, sent={e.get("sent_id","?")})'
        for i, e in enumerate(path_events)
    )

    ctx = "\n".join(
        f"[S{s[0]}] {s[1]}" if isinstance(s, (list, tuple)) else f"[S{i}] {s}"
        for i, s in enumerate(supporting_sentences[:6])
    )

    min_required = {"Easy": 1, "Medium": 2, "Hard": 2}.get(difficulty, 1)

    prompt = f"""Question: "{generated_question}"
Path events: {path_str}
Event details:
{event_list}

Context:
{ctx}

For each path event above, determine if the question semantically references or uses it.
The question does NOT need to mention the trigger word verbatim, but must clearly relate to the event.

Reply with ONLY the numbers of referenced events (comma-separated).
Example: "1, 2, 3" or "1, 3" or "2"
If none: "0"
"""

    resp = call_api(prompt, temperature=0.0, max_tokens=30)

    covered_indices = []
    if resp:
        nums = re.findall(r'\d+', resp.split("\n")[0])
        covered_indices = [int(n) for n in nums if 1 <= int(n) <= len(path_events)]

    covered_events = []
    for idx in covered_indices:
        if 1 <= idx <= len(path_events):
            covered_events.append(path_events[idx - 1]["id"])

    # If LLM returned nothing, fall back to lexical
    if not covered_events:
        lex_count, lex_ids = check_path_coverage_lexical(generated_question, path_events)
        covered_events = lex_ids

    # Deduplicate
    covered_events = list(set(covered_events))
    coverage_count = len(covered_events)

    passed = coverage_count >= min_required
    reason = (
        f"covers {coverage_count}/{len(path_events)} events, need >= {min_required}"
        + (" [PASS]" if passed else " [FAIL]")
    )

    return coverage_count, covered_events, passed, reason, resp or ""
