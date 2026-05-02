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
