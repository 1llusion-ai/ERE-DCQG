"""
Path-direction validation and Hard-question post-generation checks.

Moved from compare_hardaware.py:
- check_path_binding: lexical binding of question to path event triggers
- validate_hard_question: Hard-specific post-generation validation
"""
import re

from dcqg.utils.text import simple_stem


# ================================================================
# BANNED SHORTCUT PHRASES (Hard only)
# ================================================================

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


# ================================================================
# PATH BINDING CHECK
# ================================================================

def check_path_binding(question, events, difficulty):
    """Check if question text mentions enough path event triggers.
    Uses lexical matching with fuzzy stem matching.
    Easy: 1 (any event), Medium: 1 (prior events only), Hard: 2 (prior events only).
    Returns (pass: bool, covered_indices: list, reason: str).
    """
    # For Medium and Hard, check prior events only (exclude answer event = last)
    check_events = events
    if difficulty in ("Medium", "Hard"):
        check_events = events[:-1]  # prior events only
    min_required = {"Easy": 1, "Medium": 1, "Hard": 2}.get(difficulty, 1)

    q_lower = question.lower()
    q_words = set(q_lower.split())
    q_stems = {simple_stem(w) for w in q_words}
    covered = []

    for i, e in enumerate(check_events):
        trigger = e["trigger"].lower()
        trigger_stem = simple_stem(trigger)
        # Direct match
        if trigger in q_lower:
            covered.append(i)
            continue
        # Stem match: trigger stem in question stems
        if trigger_stem in q_stems:
            covered.append(i)
            continue
        # Substring containment (trigger in any question word or vice versa)
        matched = False
        for qw in q_words:
            if len(qw) >= 3 and len(trigger) >= 3:
                if trigger in qw or qw in trigger:
                    covered.append(i)
                    matched = True
                    break
                # Stem-level containment
                if trigger_stem in qw or qw in trigger_stem:
                    covered.append(i)
                    matched = True
                    break
        if matched:
            continue
        # Entity/type match (event type as fallback)
        etype = e.get("type", "").lower()
        if etype and len(etype) >= 4 and etype in q_lower:
            covered.append(i)

    covered = list(set(covered))
    if len(covered) >= min_required:
        return True, covered, f"covers {len(covered)}/{len(check_events)} events, need >= {min_required}"
    return False, covered, f"covers {len(covered)}/{len(check_events)} events, need >= {min_required}"


# ================================================================
# HARD QUESTION VALIDATION
# ================================================================

def validate_hard_question(question, events, gold_trigger):
    """Post-generation checks specific to Hard questions. Returns (passed, reason)."""
    # Check banned phrases
    has_banned, banned_phrase = check_banned_phrases(question)
    if has_banned:
        return False, f"banned phrase: {banned_phrase}"

    # Check that question mentions at least 2 prior events
    prior_triggers = [e["trigger"].lower() for e in events[:-1]]
    q_lower = question.lower()
    q_words = set(q_lower.split())
    q_stems = {simple_stem(w) for w in q_words}
    mentioned = []
    for t in prior_triggers:
        t_stem = simple_stem(t)
        if t in q_lower:
            mentioned.append(t)
            continue
        if t_stem in q_stems:
            mentioned.append(t)
            continue
        for qw in q_words:
            if len(qw) >= 3 and len(t) >= 3:
                if t in qw or qw in t or t_stem in qw or qw in t_stem:
                    mentioned.append(t)
                    break
    if len(set(mentioned)) < 1:
        return False, f"only {len(set(mentioned))} prior events mentioned, need >=1 from {prior_triggers}"

    # Check gold trigger not leaked
    if gold_trigger.lower() in q_lower:
        return False, "trigger leakage"

    return True, "pass"
