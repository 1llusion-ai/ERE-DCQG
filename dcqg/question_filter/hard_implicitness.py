"""Hard implicitness check: verify Hard questions don't over-explicitly list path triggers.

Counts how many prior event triggers appear explicitly in the question text.
Hard questions should have at most 1 explicit prior trigger — the rest must be
discovered by the solver from context.
"""
from dcqg.utils.text import simple_stem


def count_explicit_prior_triggers(question, events):
    """Count how many prior event triggers appear explicitly in question text.

    Uses exact/stem/near-exact matching ONLY — does NOT count paraphrases.
    Returns count of distinct prior event triggers found in question.
    """
    if not question or not events:
        return 0

    prior_events = events[:-1]  # exclude final event
    if not prior_events:
        return 0

    q_lower = question.lower()
    q_words = set(q_lower.split())
    q_stems = {simple_stem(w) for w in q_words}

    count = 0
    for e in prior_events:
        trigger = e.get("trigger", "").lower()
        if not trigger or len(trigger) < 2:
            continue
        trigger_stem = simple_stem(trigger)

        # Exact substring match
        if trigger in q_lower:
            count += 1
            continue
        # Stem match
        if trigger_stem in q_stems:
            count += 1
            continue
        # Substring containment (trigger in question word or vice versa)
        found = False
        for qw in q_words:
            if len(qw) >= 3 and len(trigger) >= 3:
                if trigger in qw or qw in trigger:
                    count += 1
                    found = True
                    break
                if trigger_stem in qw or qw in trigger_stem:
                    count += 1
                    found = True
                    break
        if found:
            continue
        # Event type fallback (same as check_path_binding)
        etype = e.get("type", "").lower()
        if etype and len(etype) >= 4 and etype in q_lower:
            count += 1

    return count


def hard_implicitness_check(question, events, difficulty):
    """Check if a Hard question avoids over-explicitly listing path triggers.

    Returns dict with:
        hard_explicit_prior_count: int — how many prior triggers explicitly in question
        hard_implicit_chain_pass: bool — True if count <= 1
        hard_implicit_chain_reason: str — explanation
    """
    if difficulty != "Hard":
        return {
            "hard_explicit_prior_count": 0,
            "hard_implicit_chain_pass": True,
            "hard_implicit_chain_reason": "not Hard",
        }

    explicit_count = count_explicit_prior_triggers(question, events)

    if explicit_count >= 3:
        return {
            "hard_explicit_prior_count": explicit_count,
            "hard_implicit_chain_pass": False,
            "hard_implicit_chain_reason": f"{explicit_count} prior triggers explicitly in question (max 2 allowed)",
        }

    return {
        "hard_explicit_prior_count": explicit_count,
        "hard_implicit_chain_pass": True,
        "hard_implicit_chain_reason": f"{explicit_count} prior triggers in question (max 2 allowed, pass)",
    }
