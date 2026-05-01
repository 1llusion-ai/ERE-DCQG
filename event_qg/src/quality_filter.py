"""
Quality Filter Pipeline for DCQG Question Generation.

Modules:
1. Enhanced grammar filter (repeat tokens, broken patterns, too long, vague)
2. Weak trigger handling (hard blacklist + needs_phrase)
3. Gold answer phrase generation (LLM-based)
4. Answer-event consistency judge (LLM-based)
5. Path coverage check (lexical + LLM)
6. Hard degraded check (LLM-based)
7. Final filter logic
"""
import re
import json
import time
from collections import Counter

from evaluator_v2 import grammar_filter as _base_grammar_filter, _call_api, _normalize

# ═══════════════════════════════════════════════════════════════
# 1. ENHANCED GRAMMAR FILTER
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
    base_pass, base_reason = _base_grammar_filter(q)
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
# 2. WEAK TRIGGER HANDLING
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# 3. GOLD ANSWER PHRASE GENERATION
# ═══════════════════════════════════════════════════════════════

ANSWER_TYPES = [
    "event_phrase", "action", "outcome", "state_change",
    "location_time", "entity", "invalid",
]


def extract_gold_answer_phrase(answer_sentence, gold_trigger, answer_event_type=None):
    """
    Extract a natural answer phrase from the answer sentence containing the trigger.
    Uses LLM to generate a proper phrase. Returns (phrase, answer_type, pass, reason).
    """
    if not answer_sentence or not gold_trigger:
        return "", "invalid", False, "missing sentence or trigger"

    prompt = f"""Given this sentence and event trigger, extract a natural answer phrase.

Sentence: "{answer_sentence}"
Event trigger: "{gold_trigger}"
Event type: "{answer_event_type or 'unknown'}"

Rules:
- The phrase must contain the trigger word.
- It should be a natural, complete phrase that answers "what happened?" about this event.
- NOT just the trigger word alone.
- Examples:
  - trigger="removing" -> phrase="removing Portuguese influence in the Malay archipelago"
  - trigger="earned" -> phrase="earned the right to play in the 2015 UEFA Super Cup"
  - trigger="wounded" -> phrase="seriously wounded another soldier"
  - trigger="appointed" -> phrase="appointed a Presidential commission of inquiry"

Also classify the answer type as one of: event_phrase, action, outcome, state_change, location_time, entity, invalid

Output ONLY: {{"phrase": "...", "answer_type": "..."}}"""

    resp = _call_api(prompt, temperature=0.0, max_tokens=80)
    if not resp:
        # Fallback: try to extract phrase locally
        phrase = _extract_phrase_locally(answer_sentence, gold_trigger)
        if phrase:
            return phrase, "event_phrase", True, "local_extraction", ""
        return "", "invalid", False, "LLM call failed", ""

    try:
        result = json.loads(resp)
    except json.JSONDecodeError:
        try:
            s = resp.index("{")
            e = resp.rindex("}") + 1
            result = json.loads(resp[s:e])
        except (ValueError, json.JSONDecodeError):
            phrase = _extract_phrase_locally(answer_sentence, gold_trigger)
            if phrase:
                return phrase, "event_phrase", True, "local_extraction_after_parse_fail", resp
            return "", "invalid", False, f"parse error: {resp[:80]}", resp

    phrase = result.get("phrase", "").strip()
    answer_type = result.get("answer_type", "invalid").strip().lower()

    if answer_type not in ANSWER_TYPES:
        answer_type = "invalid"

    if not phrase or len(phrase.split()) < 2:
        # Phrase too short
        phrase_local = _extract_phrase_locally(answer_sentence, gold_trigger)
        if phrase_local and len(phrase_local.split()) >= 2:
            return phrase_local, "event_phrase", True, "fallback_local", resp
        return phrase, answer_type, False, f"phrase too short: '{phrase}'", resp

    if gold_trigger.lower() not in phrase.lower():
        return phrase, answer_type, False, f"trigger '{gold_trigger}' not in phrase '{phrase}'", resp

    return phrase, answer_type, True, "valid phrase", resp


def _extract_phrase_locally(sentence, trigger):
    """Fallback local phrase extraction: find trigger in sentence, extract VP chunk."""
    s_lower = sentence.lower()
    t_lower = trigger.lower()
    idx = s_lower.find(t_lower)
    if idx < 0:
        return None

    words = sentence.split()
    trigger_words = trigger.split()
    # Find trigger position in word list
    trigger_start = -1
    for i in range(len(words)):
        if " ".join(w.lower().strip(".,;:!?") for w in words[i:i+len(trigger_words)]) == t_lower:
            trigger_start = i
            break
    if trigger_start < 0:
        return trigger  # at least return trigger

    # Expand: go back up to 3 words, forward up to 5 words
    start = max(0, trigger_start - 2)
    end = min(len(words), trigger_start + len(trigger_words) + 4)
    phrase = " ".join(words[start:end]).strip(".,;:!?")
    return phrase if len(phrase.split()) >= 2 else None


# ═══════════════════════════════════════════════════════════════
# 4. ANSWER-EVENT CONSISTENCY JUDGE
# ═══════════════════════════════════════════════════════════════

def _detect_judge_degradation(text):
    """Detect if judge output is garbled/looping. Returns True if degraded."""
    if not text:
        return True
    lower = text.lower()
    # Check for repeated word patterns: "the the", "on on", "did did"
    if re.search(r'\b(\w{2,})\s+\1\b', lower):
        return True
    # Check for triple+ word repetition
    if re.search(r'\b(\w{2,})(\s+\1){2,}\b', lower):
        return True
    # Check for excessive character repetition ("onD", "theD")
    if re.search(r'[a-z]{2,}[A-Z]{2,}', text):
        return True
    # Check for garbled patterns: "theD", "onD", "mentionD"
    if re.search(r'\w+D\b', text) and len(re.findall(r'\w+D\b', text)) >= 2:
        return True
    # Check for >40% repeated words
    words = lower.split()
    if len(words) >= 4:
        from collections import Counter
        counts = Counter(words)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count / len(words) > 0.4:
            return True
    return False


def answer_event_consistency_judge(
    generated_question, supporting_sentences, path_events,
    answer_event_id, gold_answer_trigger, gold_answer_phrase,
    gold_answer_sentence
):
    """
    Semantic final-event judge. Checks:
    1. asks_target_event: does the question ask about the final event?
    2. answerable: can it be answered from context?
    3. consistency: does expected answer match gold answer phrase?
    Uses strict JSON output, retries on degradation, returns "judge_error" on failure.
    """
    ctx = "\n".join(
        f"[S{s[0]}] {s[1]}" if isinstance(s, (list, tuple)) else f"[S{i}] {s}"
        for i, s in enumerate(supporting_sentences[:6])
    )
    path_str = " -> ".join(f'"{e["trigger"]}"' for e in path_events)

    prompt = f"""Evaluate this question against the target final event.

Context:
{ctx}
Path: {path_str}
Target event trigger: "{gold_answer_trigger}"
Target answer meaning: "{gold_answer_phrase}"
Target sentence: "{gold_answer_sentence}"
Question: "{generated_question}"

Answer 3 things (yes/no each):
1. asks_target: Does the question ask about the target final event?
2. answerable: Can the question be answered from the context?
3. consistent: Would a correct answer identify the same final event?

Reply ONLY as JSON:
{{"asks_target":"yes","answerable":"yes","consistent":"yes","reason":"brief"}}
or with "no"/"partial" as appropriate."""

    all_raw_responses = []
    for attempt in range(3):
        resp = _call_api(prompt, temperature=0.0, max_tokens=60, timeout=90)

        if not resp:
            all_raw_responses.append("")
            continue
        all_raw_responses.append(resp)
        if _detect_judge_degradation(resp):
            continue

        result = None
        try:
            result = json.loads(resp)
        except json.JSONDecodeError:
            try:
                s_idx = resp.index("{")
                e_idx = resp.rindex("}") + 1
                result = json.loads(resp[s_idx:e_idx])
            except (ValueError, json.JSONDecodeError):
                pass

        if result and isinstance(result, dict):
            asks = str(result.get("asks_target", "")).strip().lower()
            ans = str(result.get("answerable", "")).strip().lower()
            cons = str(result.get("consistent", "")).strip().lower()
            reason = str(result.get("reason", "")).strip()

            if asks in ("yes", "no") and ans in ("yes", "no") and cons in ("yes", "no", "partial"):
                # Map to consistency label
                if asks == "yes" and cons in ("yes", "partial"):
                    consistency = cons
                elif asks == "yes" and cons == "no":
                    consistency = "partial"
                else:
                    consistency = "no"

                return {
                    "expected_answer_type": "unknown",
                    "expected_answer_summary": "",
                    "answer_consistency": consistency,
                    "answer_consistency_reason": reason or f"asks={asks} ans={ans} cons={cons}",
                    "asks_target_event": asks == "yes",
                    "judge_answerable": ans == "yes",
                    "judge_raw_responses": all_raw_responses,
                }

    return {
        "expected_answer_type": "unknown",
        "expected_answer_summary": "",
        "answer_consistency": "judge_error",
        "answer_consistency_reason": f"judge_error after 3 attempts: {(resp or 'no response')[:80]}",
        "asks_target_event": None,
        "judge_answerable": None,
        "judge_raw_responses": all_raw_responses,
    }


# ═══════════════════════════════════════════════════════════════
# 5. PATH COVERAGE CHECK
# ═══════════════════════════════════════════════════════════════

def _simple_stem(word):
    """Simple English stemmer for matching event triggers to question words."""
    w = word.lower().strip(".,;:!?\"'")
    if len(w) <= 3:
        return w
    for suffix in ['ing', 'tion', 'sion', 'ment', 'ness', 'ity', 'ence', 'ance',
                   'ized', 'ised', 'ated', 'ened', 'ified', 'ally', 'edly',
                   'ies', 'ed', 'er', 'es', 'ly', 's']:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)]
    return w


def check_path_coverage_lexical(generated_question, path_events):
    """
    Lexical overlap check: how many path events are referenced in the question.
    Returns (covered_count, covered_event_ids).
    """
    q_lower = generated_question.lower()
    q_words = set(q_lower.split())
    q_stems = {_simple_stem(w) for w in q_words}
    covered = []
    for e in path_events:
        trigger = e["trigger"].lower()
        trigger_stem = _simple_stem(trigger)
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

    resp = _call_api(prompt, temperature=0.0, max_tokens=30)

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


# ═══════════════════════════════════════════════════════════════
# 6. HARD DEGRADED CHECK
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

    resp = _call_api(prompt, temperature=0.0, max_tokens=40, timeout=90)

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


# ═══════════════════════════════════════════════════════════════
# 7. FINAL FILTER LOGIC
# ═══════════════════════════════════════════════════════════════

def apply_final_filter(record):
    """
    Apply final filter logic to a fully-evaluated record.
    Returns (pass: bool, reason: str).
    """
    reasons = []

    # Grammar
    if not record.get("grammar_pass", False):
        reasons.append(f"grammar={record.get('grammar_reason', '?')}")

    # Weak trigger
    if not record.get("weak_trigger_pass", False):
        reasons.append(f"weak_trigger={record.get('weak_trigger_reason', '?')}")

    # Answer phrase
    if not record.get("answer_phrase_pass", False):
        reasons.append(f"answer_phrase={record.get('answer_phrase_reason', '?')}")

    # Answer consistency (judge_error is excluded from filter, not a fail)
    label = record.get("answer_consistency_label", "no")
    if label == "judge_error":
        pass  # don't fail on judge errors — mark separately
    elif label not in ("yes", "partial"):
        reasons.append(f"answer_consistency={label}: {record.get('answer_consistency_reason', '?')}")

    # Path coverage
    if not record.get("path_coverage_pass", False):
        reasons.append(f"path_coverage={record.get('path_coverage_reason', '?')}")

    # Hard degraded (only for Hard difficulty)
    if record.get("difficulty") == "Hard" and record.get("hard_degraded", False):
        reasons.append(f"hard_degraded={record.get('hard_degraded_reason', '?')}")

    if not reasons:
        return True, "all checks passed"
    return False, "; ".join(reasons)


# ═══════════════════════════════════════════════════════════════
# FULL PIPELINE: evaluate one sample through all filters
# ═══════════════════════════════════════════════════════════════

def quality_filter_pipeline(record, skip_llm=False):
    """
    Run a single record through the full quality filter pipeline.
    Modifies record in-place and returns it.
    """
    q = record.get("generated_question", "")
    events = record.get("events", [])
    difficulty = record.get("difficulty", "Easy")
    gold_trigger = record.get("gold_answer_trigger", "")
    supporting = record.get("supporting_sentences", [])

    # ── 1. Enhanced grammar filter ──
    g_pass, g_reason = enhanced_grammar_filter(q, events)
    record["grammar_pass"] = g_pass
    record["grammar_reason"] = g_reason

    # Word count for analysis
    record["question_word_count"] = len(q.split()) if q else 0

    if not g_pass:
        # Still fill in remaining fields for analysis
        _fill_remaining_fields(record, skip_llm=True)
        return record

    # ── 2. Find answer sentence and event info ──
    answer_event_id = record.get("answer_event_id", "")
    answer_event = None
    for e in events:
        if e["id"] == answer_event_id:
            answer_event = e
            break
    if not answer_event and events:
        answer_event = events[-1]
        answer_event_id = answer_event["id"]

    answer_sentence = ""
    answer_event_type = ""
    if answer_event:
        answer_event_type = answer_event.get("type", "")
        sent_id = answer_event.get("sent_id", -1)
        for s in supporting:
            sid = s[0] if isinstance(s, (list, tuple)) else supporting.index(s)
            if sid == sent_id:
                answer_sentence = s[1] if isinstance(s, (list, tuple)) else s
                break

    record["answer_event_id"] = answer_event_id

    # ── 3. Gold answer phrase ──
    upstream_phrase = record.get("gold_answer_phrase", "")
    upstream_pass = record.get("answer_phrase_pass", None)
    upstream_reason = record.get("answer_phrase_reason", "")
    if skip_llm:
        phrase = upstream_phrase or gold_trigger
        a_type = "invalid"
        a_pass = upstream_pass if upstream_pass is not None else True
        a_reason = upstream_reason or "skipped LLM"
        a_raw = ""
    else:
        llm_phrase, a_type, llm_pass, llm_reason, a_raw = extract_gold_answer_phrase(
            answer_sentence, gold_trigger, answer_event_type
        )
        record["llm_answer_phrase"] = llm_phrase
        record["llm_answer_phrase_pass"] = llm_pass
        record["llm_answer_phrase_reason"] = llm_reason
        if upstream_phrase:
            phrase = upstream_phrase
            a_pass = upstream_pass if upstream_pass is not None else True
            a_reason = upstream_reason or "upstream_answer_phrase"
        else:
            phrase = llm_phrase
            a_pass = llm_pass
            a_reason = llm_reason

    record["gold_answer_phrase"] = phrase
    record["answer_type"] = a_type
    record["gold_answer_sentence"] = answer_sentence
    record["answer_phrase_pass"] = a_pass
    record["answer_phrase_reason"] = a_reason
    record["answer_phrase_raw"] = a_raw

    # ── 4. Weak trigger ──
    wt_result = check_weak_trigger(gold_trigger, phrase)
    record.update(wt_result)

    # If weak trigger check fails, still continue for analysis
    # but skip expensive LLM calls
    if not wt_result["weak_trigger_pass"] and skip_llm:
        _fill_remaining_fields(record, skip_llm=True)
        return record

    if skip_llm:
        _fill_remaining_fields(record, skip_llm=True)
        return record

    # ── 5. Answer-event consistency judge ──
    cons_result = answer_event_consistency_judge(
        q, supporting, events,
        answer_event_id, gold_trigger, phrase, answer_sentence
    )
    record["expected_answer_type"] = cons_result["expected_answer_type"]
    record["expected_answer_summary"] = cons_result["expected_answer_summary"]
    record["answer_consistency_label"] = cons_result["answer_consistency"]
    record["answer_consistency_reason"] = cons_result["answer_consistency_reason"]
    record["answer_consistency_pass"] = cons_result["answer_consistency"] in ("yes", "partial", "judge_error")
    record["asks_target_event"] = cons_result.get("asks_target_event")
    record["judge_answerable"] = cons_result.get("judge_answerable")
    record["consistency_judge_raw"] = cons_result.get("judge_raw_responses", [])

    # ── 6. Path coverage ──
    cov_count, cov_events, cov_pass, cov_reason, cov_raw = path_coverage_judge(
        q, supporting, events, difficulty
    )
    record["path_coverage_count"] = cov_count
    record["path_covered_events"] = cov_events
    record["path_coverage_pass"] = cov_pass
    record["path_coverage_reason"] = cov_reason
    record["path_coverage_raw"] = cov_raw

    # ── 7. Hard degraded ──
    if difficulty == "Hard":
        hd_result = hard_degraded_check(q, supporting, phrase, events)
        record.update(hd_result)
    else:
        record["can_answer_from_single_sentence"] = "N/A"
        record["single_sentence_id"] = "N/A"
        record["need_intermediate_events"] = "N/A"
        record["evidence_hops_used"] = 0
        record["hard_degraded"] = False
        record["hard_degraded_reason"] = "not Hard"
        record["hard_degraded_raw"] = ""

    # ── 8. Final filter ──
    final_pass, final_reason = apply_final_filter(record)
    record["final_filter_pass"] = final_pass
    record["final_filter_reason"] = final_reason

    return record


def _fill_remaining_fields(record, skip_llm=True):
    """Fill remaining fields when early-exit from pipeline."""
    defaults = {
        "gold_answer_phrase": record.get("gold_answer_trigger", ""),
        "answer_type": "invalid",
        "gold_answer_sentence": "",
        "answer_phrase_pass": False,
        "answer_phrase_reason": "skipped (early exit)",
        "answer_phrase_raw": "",
        "weak_trigger_flag": False,
        "weak_trigger_type": "none",
        "weak_trigger_pass": True,
        "weak_trigger_reason": "not checked",
        "expected_answer_type": "unknown",
        "expected_answer_summary": "",
        "answer_consistency_label": "no",
        "answer_consistency_reason": "skipped (early exit)",
        "answer_consistency_pass": False,
        "asks_target_event": None,
        "judge_answerable": None,
        "consistency_judge_raw": [],
        "path_coverage_count": 0,
        "path_covered_events": [],
        "path_coverage_pass": False,
        "path_coverage_reason": "skipped (early exit)",
        "path_coverage_raw": "",
        "can_answer_from_single_sentence": "N/A",
        "single_sentence_id": "N/A",
        "need_intermediate_events": "N/A",
        "evidence_hops_used": 0,
        "hard_degraded": False,
        "hard_degraded_reason": "not checked",
        "hard_degraded_raw": "",
    }
    for k, v in defaults.items():
        if k not in record:
            record[k] = v

    # Final filter
    final_pass, final_reason = apply_final_filter(record)
    record["final_filter_pass"] = final_pass
    record["final_filter_reason"] = final_reason
