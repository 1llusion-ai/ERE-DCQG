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
    ",", ";", "and", "or", "but", "that", "which", "who", "where", "when",
    "because", "although", "however", "while", "though", "yet", "so",
    "nor", "if", "unless", "since", "after", "before",
}

# Words that look like they belong to a coordinated noun-phrase list.
# Used to decide whether "and"/"or" is a list coordinator (skip) or a
# clause joiner (stop).
_LIST_LIKE_POS = {
    # determiners / quantifiers
    "a", "an", "the", "this", "that", "these", "those",
    "some", "any", "no", "every", "each", "all", "both",
    "many", "several", "few", "other", "another",
    # common adjectives
    "new", "old", "large", "small", "good", "bad", "great",
    "major", "minor", "more", "most", "less", "least",
    # numbers / ordinals
    "one", "two", "three", "four", "five", "first", "second",
}


GENERIC_TRIGGERS = {
    "occurred", "happened", "took place", "made", "did", "was", "were",
    "is", "are", "had", "has", "have", "said", "told", "asked",
    "went", "came", "got", "gave", "took", "put", "set",
}


# Words that indicate a phrase is incomplete when they appear at the end.
# These are prepositions, title-introducing words, or conjunctions that
# require a complement after them.
DANGLING_END_WORDS = {
    # prepositions
    "by", "for", "with", "in", "on", "at", "to", "from", "of",
    "into", "onto", "upon", "over", "under", "through", "during",
    "before", "after", "between", "among", "against", "toward",
    "towards", "within", "without", "along", "across", "around",
    "beyond", "behind", "beside", "beneath", "despite", "except",
    "inside", "outside", "throughout", "until", "via",
    # title-introducing / naming words
    "titled", "called", "named", "known", "entitled",
    # other open-ended words
    "including", "such", "according",
}

# Multi-word dangling phrases (last N words of the phrase)
DANGLING_END_PHRASES = {
    "known as", "referred to as", "described as", "released as",
    "released in", "according to", "such as",
}


def _check_phrase_completeness(phrase):
    """Check whether an extracted phrase ends with a dangling word/phrase.

    Returns (is_complete: bool, reason: str).
    """
    if not phrase:
        return True, ""

    words = phrase.split()
    if not words:
        return True, ""

    # Check unclosed brackets
    for open_c, close_c in [("(", ")"), ("[", "]"), ("{", "}")]:
        if phrase.count(open_c) > phrase.count(close_c):
            return False, "partial extraction: unclosed bracket or quote"

    # Check unclosed quotes (odd number of double or single quotes)
    if phrase.count('"') % 2 != 0:
        return False, "partial extraction: unclosed bracket or quote"
    # Only check single quotes if there are 3+ (possessives have 1)
    if phrase.count("'") >= 3 and phrase.count("'") % 2 != 0:
        return False, "partial extraction: unclosed bracket or quote"

    last_word = words[-1].lower().strip(".,;:!?\"'")

    # Check single dangling word
    if last_word in DANGLING_END_WORDS:
        return False, f"partial extraction: phrase ends with dangling word '{last_word}'"

    # Check multi-word dangling phrases
    phrase_lower = phrase.lower()
    for dangling_phrase in DANGLING_END_PHRASES:
        if phrase_lower.rstrip(".,;:!?\"' ").endswith(dangling_phrase):
            return False, f"partial extraction: phrase ends with '{dangling_phrase}'"

    # Reject bare-fragment starters: participles, modals, or auxiliaries
    # at the beginning without a subject or real object.
    # e.g. "making landfalls on Long Island", "could operate in environments"
    first_word = words[0].lower()
    _fragment_starters = {
        # bare participles
        "making", "starting", "operating", "following", "including",
        "according", "leading", "resulting", "beginning", "moving",
        "being", "having", "going", "coming", "taking", "getting",
        "doing", "giving", "putting", "setting",
        # modals
        "could", "would", "should", "might", "may", "can", "must", "shall",
    }
    if first_word in _fragment_starters:
        return False, f"partial extraction: bare fragment starting with '{first_word}'"

    # Check passive-only structure: starts with auxiliary + past participle
    # but lacks a real object/complement after the verb.
    # e.g. "was released in VHS titled" — has aux + verb + prepositional
    # fragment but no actual noun-phrase object.
    if first_word in ("was", "were", "been", "is", "are", "be"):
        # If the phrase is only aux + verb + preposition(s), it's incomplete.
        # Count non-auxiliary content words after the participle.
        non_aux = [w for w in words if w.lower() not in (
            "was", "were", "been", "is", "are", "be",
            "has", "have", "had", "did", "do", "does",
        )]
        # If all remaining words are prepositions or the trigger itself,
        # the phrase has no real object.
        non_preposition = [w for w in non_aux if w.lower().strip(".,;:!?\"'") not in DANGLING_END_WORDS]
        if len(non_preposition) <= 1:
            return False, "partial extraction: passive structure without object"

    return True, ""


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
        raw = words[left - 1]
        if raw[-1:] in (".", "!", "?"):
            break
        last_char = raw[-1:] if len(raw) > 1 else raw
        if last_char in (",", ";") or raw in ("—", "--"):
            break
        w_clean = raw.lower().strip(".,;:!?\"'")
        if w_clean in CLAUSE_BOUNDARY_WORDS:
            break
        left -= 1

    right = trigger_end
    while right < len(words):
        raw = words[right]
        # Check sentence-ending punctuation on the raw word BEFORE stripping.
        if raw[-1:] in (".", "!", "?"):
            right += 1
            break
        # Check clause-boundary punctuation: the word IS a comma/semicolon,
        # or ENDS with one (e.g. "services,").
        last_char = raw[-1:] if len(raw) > 1 else raw
        if last_char in (",", ";") or raw in ("—", "--"):
            break
        w_clean = raw.lower().strip(".,;:!?\"'")
        if w_clean in CLAUSE_BOUNDARY_WORDS:
            # "and"/"or": peek at neighbors to decide if this is inside a
            # noun-phrase list (skip) or joining clauses (stop).
            if w_clean in ("and", "or") and right > 0 and right + 1 < len(words):
                prev_w = words[right - 1].lower().strip(".,;:!?\"'")
                next_w = words[right + 1].lower().strip(".,;:!?\"'")
                _clause_starters_after = {
                    "he", "she", "it", "they", "we", "i", "you",
                    "that", "which", "who", "where", "when", "while",
                    "although", "because", "if", "but", "so", "yet",
                }
                # Skip "and/or" if the previous word is noun-like AND the
                # next word is not a clause-starter pronoun/conjunction.
                prev_is_noun_like = (
                    prev_w in _LIST_LIKE_POS
                    or not prev_w.endswith(("ed", "ing", "ize", "ise"))
                )
                if prev_is_noun_like and next_w not in _clause_starters_after:
                    right += 1
                    continue
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

    # Completeness check: catch dangling prepositions, unclosed brackets, etc.
    if status == "complete":
        is_complete, reason = _check_phrase_completeness(phrase)
        if not is_complete:
            status = "partial"

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
