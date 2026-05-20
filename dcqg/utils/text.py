"""Consolidated text utilities.

simple_stem: canonical from answer_extraction.py
normalize, fuzzy_match, text_similarity, detect_loop: from evaluator.py / evaluator_v2.py
"""
import re


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


# Backward-compatible alias
_simple_stem = simple_stem


def normalize(text):
    """Lowercase, remove punctuation, strip."""
    return re.sub(r'[^\w\s]', '', text.lower()).strip()


def fuzzy_match(answer, trigger):
    """Check if answer contains trigger or semantic equivalent.
    Returns: 'exact' | 'fuzzy' | 'stem' | 'none'
    Uses v2 logic with LCS similarity fallback.
    """
    a = normalize(answer)
    t = normalize(trigger)
    if not a or not t:
        return 'none'
    if t in a or a in t:
        return 'exact'
    a_words = {w for w in a.split() if len(w) > 2}
    t_words = {w for w in t.split() if len(w) > 2}
    if a_words & t_words:
        return 'fuzzy'
    for aw in a_words:
        for tw in t_words:
            if len(aw) >= 4 and len(tw) >= 4 and (aw.startswith(tw[:4]) or aw in tw or tw in aw):
                return 'stem'
    # Text similarity fallback
    a_tok = a.split()
    t_tok = t.split()
    m = len(a_tok)
    n = len(t_tok)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    mx = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a_tok[i - 1] == t_tok[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                mx = max(mx, dp[i][j])
    lcs = mx / max(m, n) if max(m, n) > 0 else 0
    jac = len(a_words & t_words) / len(a_words | t_words) if (a_words | t_words) else 0
    sim = 0.4 * lcs + 0.6 * jac
    return 'fuzzy' if sim > 0.5 else ('stem' if sim > 0.3 else 'none')


def text_similarity(a, b):
    """Compute text similarity using token overlap + longest common substring.
    Returns score 0-1. No API call needed.
    """
    a_tok = normalize(a).split()
    b_tok = normalize(b).split()
    if not a_tok or not b_tok:
        return 0.0

    # Token overlap (Jaccard)
    a_set = set(a_tok)
    b_set = set(b_tok)
    intersection = a_set & b_set
    union = a_set | b_set
    jaccard = len(intersection) / len(union) if union else 0

    # Longest common token subsequence
    m, n = len(a_tok), len(b_tok)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a_tok[i - 1] == b_tok[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_len = max(max_len, dp[i][j])

    lcs_score = max_len / max(m, n) if max(m, n) > 0 else 0

    # Containment bonus
    a_str = normalize(a)
    b_str = normalize(b)
    containment = 1.0 if (a_str and b_str and (a_str in b_str or b_str in a_str)) else 0.0

    return 0.3 * jaccard + 0.4 * lcs_score + 0.3 * containment


# ── Dialogue-aware sentence splitting ────────────────────────────

_SPEECH_VERBS = frozenset([
    'said', 'replied', 'asked', 'whispered', 'cried', 'answered',
    'exclaimed', 'continued', 'began', 'spoke', 'called', 'shouted',
    'muttered', 'added', 'told', 'yelled', 'screamed', 'begged',
    'insisted', 'demanded', 'commanded', 'ordered', 'thought',
    'remarked', 'observed', 'declared', 'announced', 'suggested',
    'offered', 'inquired', 'questioned', 'responded', 'retorted',
])


def _is_speech_tag(text):
    """Check if text is a short speech attribution tag (not a dialogue sentence)."""
    text = text.strip()
    if not text:
        return False
    if text.startswith('"') or text.startswith('"'):
        return False
    if re.search(r'[.!?]\s+\S', text):
        return False
    if '--' in text:
        return False
    if text.endswith(':') or text.endswith(','):
        return True
    if len(text) <= 40:
        lower = text.lower()
        return any(v in lower for v in _SPEECH_VERBS)
    return False


def _find_dialogue_spans(text):
    """Find all complete quote-pair spans in text."""
    spans = []
    i = 0
    while i < len(text):
        open_pos = text.find('"', i)
        if open_pos == -1:
            break
        close_pos = text.find('"', open_pos + 1)
        if close_pos == -1:
            break
        spans.append((open_pos, close_pos))
        i = close_pos + 1
    return spans


def _dialogue_inner_ends_comma(text, s, e):
    """Check if dialogue content (inside quotes) ends with a comma."""
    inner = text[s + 1:e].rstrip()
    return inner.endswith(',') if inner else False


def _build_dialogue_units(text, raw_spans):
    """Build dialogue units from raw quote spans.

    1. Merges consecutive spans separated by a comma-ending speech tag
       (interrupted dialogue: "dialogue1," said X, "dialogue2").
    2. Absorbs speech tags that follow each merged span.
    """
    if not raw_spans:
        return []

    merged_spans = [raw_spans[0]]
    for s, e in raw_spans[1:]:
        prev_s, prev_e = merged_spans[-1]
        if _dialogue_inner_ends_comma(text, prev_s, prev_e):
            between = text[prev_e + 1:s].strip()
            if between and _is_speech_tag(between) and between.endswith(','):
                merged_spans[-1] = (prev_s, e)
                continue
        merged_spans.append((s, e))

    units = []
    for d_start, d_end in merged_spans:
        dialogue_text = text[d_start:d_end + 1]
        tag_end = d_end
        after_text = text[d_end + 1:]
        if after_text:
            m = re.match(r'\s*([^.!?\n"""]*?(?:[.!?]|$))', after_text)
            if m:
                seg = m.group(1).strip()
                if seg and _is_speech_tag(seg):
                    tag_end = d_end + 1 + m.end()
                    dialogue_text = dialogue_text + " " + seg
        units.append((d_start, tag_end, dialogue_text))

    return units


def split_sentences(text):
    """Dialogue-aware sentence splitter.

    Keeps multi-sentence dialogue intact within a single quote pair.
    Splits narrative text between dialogue segments at sentence boundaries.
    Merges speech tags with their associated dialogue.
    Merges interrupted dialogue ("dialogue," said X, "continuation") into
    one sentence when the tag ends with a comma.
    """
    if not text:
        return []
    text = text.strip()

    raw_spans = _find_dialogue_spans(text)
    if not raw_spans:
        return [s for s in re.split(r'(?<=[.!?])\s+', text) if s]

    units = _build_dialogue_units(text, raw_spans)

    sentences = []
    prev_end = 0
    for u_start, u_end, u_text in units:
        before = text[prev_end:u_start].strip()
        if before:
            sentences.extend([s for s in re.split(r'(?<=[.!?])\s+', before) if s])
        sentences.append(u_text)
        prev_end = u_end + 1

    after = text[prev_end:].strip()
    if after:
        sentences.extend([s for s in re.split(r'(?<=[.!?])\s+', after) if s])

    sentences = [s for s in sentences if s.strip()]

    merged = []
    i = 0
    while i < len(sentences):
        s = sentences[i]
        if (_is_speech_tag(s) and i + 1 < len(sentences)
                and sentences[i + 1].startswith('"')):
            merged.append(s + " " + sentences[i + 1])
            i += 2
        else:
            merged.append(s)
            i += 1

    # Post-process: merge quoted names/titles that were incorrectly split out.
    # Pattern: prev is a short fragment, current is a short quoted span,
    # next continues the sentence (lowercase start).
    fixed = []
    i = 0
    while i < len(merged):
        cur = merged[i].strip()
        prev = fixed[-1].strip() if fixed else ""
        nxt = merged[i + 1].strip() if i + 1 < len(merged) else ""

        is_short_quoted = (
            len(cur) <= 25
            and (cur.startswith('"') or cur.startswith('"') or cur.startswith("'"))
        )
        prev_is_fragment = (
            prev
            and len(prev) < 15
            and not prev.endswith((".", "!", "?"))
        )
        next_continues = (
            nxt
            and len(nxt) > 0
            and nxt[0].islower()
        )

        if is_short_quoted and prev_is_fragment and next_continues:
            # Merge prev + cur + next into one sentence
            fixed[-1] = prev + " " + cur + " " + nxt
            i += 2  # skip next
        else:
            fixed.append(merged[i])
            i += 1

    return fixed


def detect_loop(text):
    """Return cleaned text if looping detected, else original."""
    # Pattern 1: repeated word 4+ times consecutively
    if re.search(r'\b(\w+)\b(\s+\1\b){3,}', text):
        cleaned = re.sub(r'(\b\w+\b)(\s+\1\b){3,}.*', r'\1', text, flags=re.IGNORECASE)
        return cleaned.strip()

    # Pattern 2: same 3-gram repeated
    words = text.split()
    for i in range(len(words) - 5):
        if words[i:i + 3] == words[i + 3:i + 6]:
            return ' '.join(words[:i + 3])

    # Pattern 3: trailing garbage syllables
    if re.search(r'(isis|onon|availableavailable|meansmeans|usedused)', text.lower()):
        cleaned = re.sub(r'\s*(isis|onon|availableavailable|meansmeans|usedused)\S*.*$', '', text, flags=re.IGNORECASE)
        return cleaned.strip()

    return text
