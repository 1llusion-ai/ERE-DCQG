"""Answer phrase extraction and final-event validity helpers.

These helpers are shared by path prefiltering, pilots, and generation. They are
kept separate from `compare_hardaware.py` so the current pipeline does not
depend on a monolithic experiment script for basic data preparation.
"""


def simple_stem(word):
    """Simple English stemmer for matching event triggers to question words."""
    w = word.lower().strip(".,;:!?\"'")
    if len(w) <= 3:
        return w
    for suffix in [
        "ing", "tion", "sion", "ment", "ness", "ity", "ence", "ance",
        "ized", "ised", "ated", "ened", "ified", "ally", "edly",
        "ies", "ed", "er", "es", "ly", "s",
    ]:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)]
    return w


# Backward-compatible name for older call sites.
_simple_stem = simple_stem


CLAUSE_BOUNDARY_WORDS = {
    ",", ";", "but", "and", "that", "which", "who", "where", "when",
    "because", "although", "however", "while", "though", "yet", "so",
    "nor", "or", "if", "unless", "since", "after", "before",
}


GENERIC_TRIGGERS = {
    "occurred", "happened", "took place", "made", "did", "was", "were",
    "is", "are", "had", "has", "have", "said", "told", "asked",
    "went", "came", "got", "gave", "took", "put", "set",
}


WEAK_TRIGGERS = {
    "influence", "battle", "war", "opening", "played", "held",
    "marked", "commented", "earned", "toured", "operate", "control",
    "receiving", "sent", "start", "started", "formalize",
}


def extract_answer_phrase_local(sentence, trigger):
    """Extract a natural answer phrase using clause-aware expansion.

    Expands from trigger to nearby clause boundaries instead of a fixed word
    window. Returns `(phrase, status)` where status is `complete`, `partial`,
    or `invalid`.
    """
    if not sentence or not trigger:
        return trigger, "invalid"

    words = sentence.split()
    trigger_words = trigger.lower().split()

    trigger_start = -1
    for i in range(len(words)):
        candidate = " ".join(
            w.lower().strip(".,;:!?\"'") for w in words[i:i + len(trigger_words)]
        )
        if candidate == trigger.lower():
            trigger_start = i
            break
    if trigger_start < 0:
        for i, word in enumerate(words):
            wl = word.lower()
            tl = trigger.lower()
            if tl in wl or wl in tl:
                trigger_start = i
                break
    if trigger_start < 0:
        return trigger, "invalid"

    trigger_end = trigger_start + len(trigger_words)

    left = trigger_start
    while left > 0:
        w_clean = words[left - 1].lower().strip(".,;:!?\"'")
        if w_clean in CLAUSE_BOUNDARY_WORDS:
            break
        if words[left - 1] in (".", "!", "?"):
            break
        left -= 1

    right = trigger_end
    while right < len(words):
        w_clean = words[right].lower().strip(".,;:!?\"'")
        if w_clean in CLAUSE_BOUNDARY_WORDS:
            break
        if words[right][-1:] in (".", "!", "?"):
            right += 1
            break
        right += 1

    # Handle "subject (parenthetical) was trigger ..." constructions. A normal
    # clause-boundary expansion can start inside the parenthetical, yielding
    # fragments such as "proof-tested ... ) was employed ...".
    paren_open = -1
    paren_close = -1
    for i in range(0, trigger_start):
        if "(" in words[i] and paren_open < 0:
            paren_open = i
        if ")" in words[i]:
            paren_close = i
    if 0 <= paren_open < paren_close < trigger_start and paren_open > 0:
        prefix = words[:paren_open]
        tail = words[paren_close + 1:right]
        if prefix and tail:
            phrase = " ".join(prefix + tail).strip(".,;:!?\"'")
            if len(phrase.split()) >= 2:
                return phrase, "complete"

    phrase = " ".join(words[left:right]).strip(".,;:!?\"'")

    if len(phrase.split()) < 2:
        return trigger, "invalid"
    if left == 0 and right == len(words):
        status = "partial" if len(words) > 15 else "complete"
    else:
        status = "complete"

    return phrase, status


def enrich_path_item(item):
    """Add answer phrase fields to a path item. Returns an enriched copy."""
    item = dict(item)
    events = item.get("events", [])
    if not events:
        return item

    final_event = events[-1]
    trigger = final_event.get("trigger", "")
    event_type = final_event.get("type", "")
    sent_id = final_event.get("sent_id", -1)

    answer_sentence = ""
    for sentence in item.get("supporting_sentences", []):
        sid = sentence[0] if isinstance(sentence, (list, tuple)) else -1
        if sid == sent_id:
            answer_sentence = sentence[1] if isinstance(sentence, (list, tuple)) else sentence
            break

    answer_phrase, answer_phrase_status = extract_answer_phrase_local(answer_sentence, trigger)

    item["gold_answer_phrase"] = answer_phrase
    item["gold_answer_sentence"] = answer_sentence
    item["gold_event_type"] = event_type
    item["answer_phrase_status"] = answer_phrase_status
    return item


def is_valid_final_event(item):
    """Check whether the final event is a usable answer target."""
    events = item.get("events", [])
    if not events:
        return False, "no events"

    final = events[-1]
    trigger = final.get("trigger", "").lower().strip()
    answer_phrase = item.get("gold_answer_phrase", trigger)

    if trigger in GENERIC_TRIGGERS:
        return False, f"generic trigger: '{trigger}'"

    if trigger in WEAK_TRIGGERS and len(answer_phrase.split()) < 3:
        return False, f"weak trigger without sufficient phrase: '{trigger}' -> '{answer_phrase}'"

    if len(answer_phrase.split()) < 2 and len(trigger) <= 3:
        return False, f"answer phrase too short: '{answer_phrase}'"

    if not item.get("gold_answer_sentence", ""):
        return False, "no answer sentence found"

    return True, "valid"
