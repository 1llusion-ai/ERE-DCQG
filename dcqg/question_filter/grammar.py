"""
Enhanced grammar filter for generated questions.

Modules:
- grammar_filter: base v2 grammar checks (question mark, word count, repetition, bad start)
- enhanced_grammar_filter: adds repeat tokens, broken patterns, too long, vague checks
- check_weak_trigger: weak trigger handling (hard blacklist + needs_phrase)
"""
import re

from dcqg.utils.text import normalize


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Specific repeat token pairs to reject
REPEAT_TOKEN_PATTERNS = [
    r'\bon\s+on\b',
    r'\bnot\s+not\b',
    r'\bdevelop\s+develop\b',
    r'\bdid\s+did\b',
    r'\bwas\s+was\b',
    r'\bwere\s+were\b',
    r'\bhad\s+had\b',
    r'\bhas\s+has\b',
    r'\bhave\s+have\b',
    r'\bis\s+is\b',
    r'\bare\s+are\b',
    r'\bdo\s+do\b',
    r'\bdoes\s+does\b',
    r'\bcan\s+can\b',
    r'\bwill\s+will\b',
    r'\bwould\s+would\b',
    r'\bshould\s+should\b',
    r'\bcould\s+could\b',
]

# Repeat question mark patterns
REPEAT_QMARK_PATTERNS = [
    r'\?\?',
    r'\?\s+\?',
]

# Broken grammar start patterns
BROKEN_START_PATTERNS = [
    r'^What\s+after\s+',
    r'^What\s+the\s+\w+\s+(?!(?:was|were|is|are|did|do|does|had|has|have|could|would|should|can|will)\b)',
    r'^What\s+did\s+\w+\s+after\s+',
    r'^What\s+was\s+the\s+\w+\s+after\s+connect\s+',
]

# Vague / empty questions (no specific event info)
VAGUE_QUESTIONS = [
    r'^What\s+happened\s*\?$',
    r'^What\s+was\s+the\s+event\s*\?$',
    r'^What\s+was\s+the\s+final\s+event\s*\?$',
    r'^What\s+was\s+the\s+result\s*\?$',
    r'^What\s+occurred\s*\?$',
    r'^What\s+took\s+place\s*\?$',
]

WORD_LIMIT_STRICT = 35   # mark as too_long
WORD_LIMIT_HARD = 45     # reject

# Weak trigger sets
HARD_BLACKLIST_TRIGGERS = {
    "said", "occurred", "happened", "took place",
    "made", "did", "was", "were",
}

WEAK_TRIGGER_NEEDS_PHRASE = {
    "held", "marked", "commented", "earned", "toured",
    "operate", "control", "influence", "receiving",
    "sent", "start", "started", "formalize",
    "played", "battle", "war", "opening",
}

ANSWER_TYPES = [
    "event_phrase", "action", "outcome", "state_change",
    "location_time", "entity", "invalid",
]


# ═══════════════════════════════════════════════════════════════
# BASE GRAMMAR FILTER (v2 from evaluator_v2.py)
# ═══════════════════════════════════════════════════════════════

def grammar_filter(question):
    q = question.strip()
    if not q or not q.endswith("?"):
        return False, "no question mark"
    words = q.lower().split()
    if len(words) < 4:
        return False, "too short"
    for i in range(len(words) - 1):
        if words[i] == words[i+1] and len(words[i]) > 1:
            return False, f"word repetition: {words[i]}"
    for i in range(len(words) - 5):
        if words[i:i+3] == words[i+1:i+4] == words[i+2:i+5]:
            return False, "looping trigram"
    for word in set(words):
        if len(word) <= 2:
            continue
        if words.count(word) >= 5 and words.count(word) / len(words) > 0.4:
            return False, f"excessive repetition: {word}"
    q_starters = {"what", "who", "when", "where", "why", "how", "which", "whose",
                  "did", "was", "were", "is", "are", "do", "does", "had", "has", "have",
                  "can", "could", "would", "should", "will",
                  "after", "before", "during", "following"}
    if words[0] not in q_starters:
        return False, f"bad start: {words[0]}"
    return True, "pass"


# ═══════════════════════════════════════════════════════════════
# ENHANCED GRAMMAR FILTER
# ═══════════════════════════════════════════════════════════════

def enhanced_grammar_filter(question, path_events=None):
    """
    Enhanced grammar filter. Returns (pass: bool, reason: str).
    Runs base grammar_filter first, then new rules.
    """
    q = question.strip()
    if not q:
        return False, "empty"
    words = q.split()
    q_lower = q.lower()

    # --- Base grammar filter ---
    base_pass, base_reason = grammar_filter(q)
    if not base_pass:
        return False, f"base: {base_reason}"

    # --- Repeat tokens (any consecutive duplicate word) ---
    q_words = q_lower.split()
    for i in range(len(q_words) - 1):
        if q_words[i] == q_words[i + 1] and len(q_words[i]) > 1:
            return False, f"repeat_token: {q_words[i]}"
    # Specific patterns (catches "on on" with punctuation between)
    for pat in REPEAT_TOKEN_PATTERNS:
        if re.search(pat, q_lower):
            m = re.search(pat, q_lower)
            return False, f"repeat_token_pattern: {m.group(0).strip()}"

    # --- Repeat question marks ---
    for pat in REPEAT_QMARK_PATTERNS:
        if re.search(pat, q):
            return False, "repeat_question_mark"

    # --- Broken grammar starts ---
    for pat in BROKEN_START_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            m = re.search(pat, q, re.IGNORECASE)
            return False, f"broken_grammar: {m.group(0).strip()}"

    # --- Too long ---
    if len(words) > WORD_LIMIT_HARD:
        return False, f"too_long_hard: {len(words)} words"
    if len(words) > WORD_LIMIT_STRICT:
        # Mark but don't reject (will be flagged)
        pass  # handled by caller via word count field

    # --- Vague questions (unless they contain 2+ path event info) ---
    for pat in VAGUE_QUESTIONS:
        if re.search(pat, q, re.IGNORECASE):
            # Check if question mentions 2+ path events
            if path_events:
                triggers = [e["trigger"].lower() for e in path_events]
                mentioned = sum(1 for t in triggers if t in q_lower)
                if mentioned < 2:
                    return False, f"vague_question: {re.match(pat, q, re.IGNORECASE).group(0)}"
            else:
                return False, f"vague_question: {re.match(pat, q, re.IGNORECASE).group(0)}"

    # Flag too_long (soft)
    too_long_flag = len(words) > WORD_LIMIT_STRICT
    reason = "pass" if not too_long_flag else f"pass_but_too_long: {len(words)} words"
    return True, reason


# ═══════════════════════════════════════════════════════════════
# WEAK TRIGGER HANDLING
# ═══════════════════════════════════════════════════════════════

def check_weak_trigger(gold_answer_trigger, gold_answer_phrase=None):
    """
    Check if trigger is weak. Returns dict:
    - weak_trigger_flag: bool
    - weak_trigger_type: "none" | "hard_blacklist" | "needs_phrase"
    - weak_trigger_pass: bool
    - weak_trigger_reason: str
    """
    trigger_lower = gold_answer_trigger.lower().strip()

    if trigger_lower in HARD_BLACKLIST_TRIGGERS:
        return {
            "weak_trigger_flag": True,
            "weak_trigger_type": "hard_blacklist",
            "weak_trigger_pass": False,
            "weak_trigger_reason": f"hard_blacklisted trigger: '{gold_answer_trigger}'",
        }

    if trigger_lower in WEAK_TRIGGER_NEEDS_PHRASE:
        # Needs a valid phrase
        if gold_answer_phrase and len(gold_answer_phrase.split()) >= 3:
            phrase_lower = gold_answer_phrase.lower()
            # Check phrase is not just the trigger repeated
            if phrase_lower.strip() != trigger_lower:
                return {
                    "weak_trigger_flag": True,
                    "weak_trigger_type": "needs_phrase",
                    "weak_trigger_pass": True,
                    "weak_trigger_reason": f"weak trigger but valid phrase: '{gold_answer_phrase}'",
                }
        return {
            "weak_trigger_flag": True,
            "weak_trigger_type": "needs_phrase",
            "weak_trigger_pass": False,
            "weak_trigger_reason": f"needs_phrase trigger without valid phrase: '{gold_answer_trigger}'",
        }

    return {
        "weak_trigger_flag": False,
        "weak_trigger_type": "none",
        "weak_trigger_pass": True,
        "weak_trigger_reason": "not a weak trigger",
    }
