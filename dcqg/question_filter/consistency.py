"""
Answer-event consistency checking.

- extract_gold_answer_phrase: LLM-based answer phrase extraction with local fallback
- answer_event_consistency_judge: semantic final-event judge
"""
import json
import re
from collections import Counter

from dcqg.utils.api_client import call_api
from dcqg.utils.text import simple_stem


# Backward-compatible alias used internally
_simple_stem = simple_stem


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

ANSWER_TYPES = [
    "event_phrase", "action", "outcome", "state_change",
    "location_time", "entity", "invalid",
]


# ═══════════════════════════════════════════════════════════════
# GOLD ANSWER PHRASE GENERATION
# ═══════════════════════════════════════════════════════════════

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

    resp = call_api(prompt, temperature=0.0, max_tokens=80)
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
# ANSWER-EVENT CONSISTENCY JUDGE
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
        resp = call_api(prompt, temperature=0.0, max_tokens=60, timeout=90)

        if not resp:
            all_raw_responses.append("")
            continue
        all_raw_responses.append(resp)
        if _detect_judge_degradation(resp):
            continue

        result = _parse_judge_json(resp)

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

    # Last resort: try simplified key=value extraction from all responses
    for resp in all_raw_responses:
        if not resp:
            continue
        result = _extract_key_value_pairs(resp)
        if result:
            asks = result.get("asks_target", "")
            ans = result.get("answerable", "")
            cons = result.get("consistent", "")
            if asks in ("yes", "no") and ans in ("yes", "no") and cons in ("yes", "no", "partial"):
                consistency = cons if asks == "yes" else "no"
                return {
                    "expected_answer_type": "unknown",
                    "expected_answer_summary": "",
                    "answer_consistency": consistency,
                    "answer_consistency_reason": f"extracted from text: asks={asks} ans={ans} cons={cons}",
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


def _parse_judge_json(resp):
    """Parse JSON from judge response with robust error handling."""
    # 1. Direct JSON parse
    try:
        return json.loads(resp)
    except json.JSONDecodeError:
        pass

    # 2. Extract first {...}
    try:
        s_idx = resp.index("{")
        e_idx = resp.rindex("}") + 1
        return json.loads(resp[s_idx:e_idx])
    except (ValueError, json.JSONDecodeError):
        pass

    # 3. Fix common JSON format errors
    cleaned = resp.strip()
    # Remove markdown code fences
    cleaned = re.sub(r'^```(?:json)?', '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'```$', '', cleaned).strip()
    # Fix missing quotes around keys: {asks_target:yes} -> {"asks_target":"yes"}
    cleaned = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', cleaned)
    # Fix single quotes to double quotes
    cleaned = cleaned.replace("'", '"')
    # Fix trailing commas
    cleaned = re.sub(r',\s*}', '}', cleaned)
    # Fix duplicated quotes
    cleaned = re.sub(r'""+', '"', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    return None


def _extract_key_value_pairs(resp):
    """Extract key-value pairs from malformed judge text."""
    result = {}
    patterns = [
        (r'"asks_target"\s*:\s*"(yes|no|partial)"', "asks_target"),
        (r'asks_target["\s:]+(yes|no|partial)', "asks_target"),
        (r'"answerable"\s*:\s*"(yes|no|partial)"', "answerable"),
        (r'answerable["\s:]+(yes|no|partial)', "answerable"),
        (r'"consistent"\s*:\s*"(yes|no|partial)"', "consistent"),
        (r'consistent["\s:]+(yes|no|partial)', "consistent"),
    ]
    for pattern, key in patterns:
        m = re.search(pattern, resp, re.IGNORECASE)
        if m:
            result[key] = m.group(1).lower()
    return result if len(result) >= 2 else None
