"""PathQG-HardAware generator with retry and validation.

generate_with_retry_hardaware: main generation function with difficulty-aware
prompts, grammar filtering, path binding check, and repair retries.
"""
import re
from dcqg.question_filter.grammar import grammar_filter
from dcqg.path.direction import check_path_binding, validate_hard_question
from dcqg.generation.prompts import (
    prompt_pathqg_easy, prompt_pathqg_medium, prompt_pathqg_hard, prompt_pathqg_hard_implicit,
    prompt_hidden_endpoint, prompt_relation_composition, prompt_contrastive, prompt_missing_bridge,
    fmt_ctx,
)
from dcqg.generation.parser import generate_one
from dcqg.generation.repair import build_repair_prompt, build_alignment_repair_prompt, REPAIRABLE_REASONS
from dcqg.question_filter.hard_implicitness import count_explicit_prior_triggers
from dcqg.utils.api_client import call_api


# ── Double question ban ─────────────────────────────────────
_DOUBLE_Q_PATTERNS = [
    r'(?i)^How\s+did\s+.+,\s*and\s+what\b',       # "How did X, and what..."
    r'(?i)^Why\s+did\s+.+,\s*(ultimately|and)\b',   # "Why did X, ultimately..."
    r'(?i)^[Ww]hat\s+led\s+to\s+.+,\s*and\s+how\b', # "what led to X, and how..."
    r'(?i)^How\s+did\s+.+,\s*and\s+how\b',         # "How did X, and how..."
    r'(?i)^What\s+.+,\s*and\s+how\s+did\b',         # "What X, and how did..."
    r'(?i)^What\s+.+,\s*and\s+what\b',              # "What X, and what..."
    r'(?i)^How\s+.+,\s*and\s+why\b',                # "How X, and why..."
    r'(?i)\band\s+(what|how|why|when|where|who)\b.*\?', # "... and what/how/why..."
]
_DOUBLE_Q_RE = [re.compile(p) for p in _DOUBLE_Q_PATTERNS]


def _is_double_question(question):
    """Reject questions containing two question clauses."""
    q = question.strip()
    for pat in _DOUBLE_Q_RE:
        if pat.search(q):
            return True, "double_question"
    return False, ""


# ── Rule-based drift checks ─────────────────────────────────

# Answer type inference from gold_answer_phrase / gold_event_type
_ANSWER_TYPE_PATTERNS = {
    "preventing_or_letting": [
        r'(?i)\b(forbade|forbidden|prohibit|restrict|prevent|block|denied|banned|limited|constraint)\b',
        r'(?i)\b(not allowed|not permitted|no longer|would not)\b',
    ],
    "sign_agreement": [
        r'(?i)\b(signed|agreed|treaty|resolution|accord|pact|ceasefire|armistice|formal)\b',
        r'(?i)\b(agreement|settlement|convention|protocol)\b',
    ],
    "criminal_investigation": [
        r'(?i)\b(closed|investigated|indicted|convicted|sentenced|charged|acquitted|dropped)\b',
        r'(?i)\b(investigation|inquiry|case|trial|verdict|prosecution)\b',
        r'(?i)\b(without indictment|no indictment|sole suspect)\b',
    ],
    "death_injury_damage": [
        r'(?i)\b(killed|died|injured|wounded|damaged|destroyed|casualties|victims|dead)\b',
        r'(?i)\b(death|injury|damage|destruction|harm|fatality)\b',
    ],
    "transfer_ownership": [
        r'(?i)\b(transferred|acquired|purchased|sold|bought|owned|seized|confiscated)\b',
        r'(?i)\b(transfer|acquisition|purchase|sale|ownership|possession)\b',
    ],
}
_ANSWER_TYPE_RES = {k: [re.compile(p) for p in v] for k, v in _ANSWER_TYPE_PATTERNS.items()}

# Allowed question heads per answer type
_ALLOWED_HEADS = {
    "preventing_or_letting": [
        "what restriction", "what limitation", "what was forbidden", "what constraint",
        "what was prohibited", "what was denied", "what ban", "what measure",
    ],
    "sign_agreement": [
        "what agreement", "what treaty", "what formal resolution", "what accord",
        "what pact", "what ceasefire", "what settlement", "what was signed",
    ],
    "criminal_investigation": [
        "what was the final outcome of the investigation", "how did the investigation conclude",
        "what happened to the case", "what was the verdict", "what legal outcome",
        "what was the result of the investigation", "how did the case end",
    ],
    "death_injury_damage": [
        "what harm", "what damage", "what casualties", "what final outcome",
        "what happened to", "what was the toll", "what destruction",
    ],
    "transfer_ownership": [
        "what transfer", "what change of ownership", "what acquisition",
        "what happened to the ownership", "what was seized",
    ],
    "other": [
        "what resulted", "what was the outcome", "what consequence",
        "what happened as a result", "what was the result",
    ],
}

# Banned drift nouns/frames (when not the target answer type)
_DRIFT_NOUNS = [
    "outcry", "inquiry", "campaign", "decision", "influence", "response",
    "reason", "protest", "reaction", "backlash", "controversy", "debate",
    "discussion", "deliberation", "consideration", "evaluation", "assessment",
]
_DRIFT_FRAMES = [
    r'(?i)^how\s+did\s+.+\s+influence\b',
    r'(?i)^how\s+did\s+.+\s+affect\b',
    r'(?i)^how\s+did\s+.+\s+impact\b',
    r'(?i)^how\s+did\s+.+\s+lead\s+to\b',
    r'(?i)^how\s+did\s+.+\s+contribute\s+to\b',
    r'(?i)^why\s+did\b',
    r'(?i)^what\s+led\s+to\b',
    r'(?i)^what\s+caused\b',
    r'(?i)^what\s+influenced\b',
    r'(?i)^what\s+prompted\b',
    r'(?i)^what\s+motivated\b',
]
_DRIFT_FRAME_RES = [re.compile(p) for p in _DRIFT_FRAMES]


def _infer_answer_type(gold_answer_phrase, gold_event_type):
    """Infer expected answer type from gold_answer_phrase and gold_event_type."""
    phrase = gold_answer_phrase or ""
    etype = gold_event_type or ""

    # Check event type first
    etype_lower = etype.lower()
    if any(kw in etype_lower for kw in ("prevent", "letting", "permission", "restriction", "prohibit", "forbid")):
        return "preventing_or_letting"
    if any(kw in etype_lower for kw in ("sign_agreement", "agreement", "treaty", "resolution")):
        return "sign_agreement"
    if any(kw in etype_lower for kw in ("criminal", "investigation", "arrest", "convict", "sentence", "charge")):
        return "criminal_investigation"
    if any(kw in etype_lower for kw in ("death", "injury", "damage", "destroy", "kill", "harm")):
        return "death_injury_damage"
    if any(kw in etype_lower for kw in ("transfer", "acqui", "ownership", "purchase", "buy", "sell")):
        return "transfer_ownership"

    # Check answer phrase
    for atype, patterns in _ANSWER_TYPE_RES.items():
        for pat in patterns:
            if pat.search(phrase):
                return atype

    return "other"


def _get_allowed_heads(answer_type):
    """Get allowed question heads for the given answer type."""
    return _ALLOWED_HEADS.get(answer_type, _ALLOWED_HEADS["other"])


def _check_question_answer_drift(question, answer_type, gold_answer_phrase):
    """Check if question drifts from expected answer type.

    Returns (drifted: bool, drift_info: str).
    """
    q_lower = question.lower().strip()

    # Check for banned drift frames (always reject these)
    for pat in _DRIFT_FRAME_RES:
        if pat.search(q_lower):
            return True, f"drift_frame: {pat.pattern}"

    # Check for banned drift nouns (only if not matching the target answer type)
    if answer_type not in ("other",):
        for noun in _DRIFT_NOUNS:
            if noun in q_lower:
                # Check if this noun is part of the answer phrase
                if gold_answer_phrase and noun in gold_answer_phrase.lower():
                    continue
                return True, f"drift_noun: {noun}"

    # Check if question head matches allowed heads for answer type
    allowed = _get_allowed_heads(answer_type)
    head_match = False
    for head in allowed:
        if q_lower.startswith(head):
            head_match = True
            break

    # If no allowed head matches, check for generic acceptable patterns
    if not head_match:
        # Accept "what" questions that ask for outcomes/results
        generic_ok = [
            r'^what\s+(final\s+)?(restriction|limitation|constraint|prohibition)',
            r'^what\s+(final\s+)?(outcome|result|consequence|conclusion)',
            r'^what\s+(final\s+)?(action|measure|step|decision)\s+was\s+taken',
            r'^what\s+(final\s+)?(agreement|treaty|resolution|accord)',
            r'^what\s+(final\s+)?(harm|damage|destruction|casualties)',
            r'^what\s+was\s+(the\s+)?(final\s+)?(outcome|result|consequence)',
            r'^what\s+was\s+(forbidden|prohibited|restricted|denied|banned)',
            r'^what\s+was\s+(signed|agreed|concluded|resolved)',
        ]
        for gp in generic_ok:
            if re.match(gp, q_lower):
                head_match = True
                break

    if not head_match:
        return True, f"head_mismatch: question does not start with allowed head for {answer_type}"

    return False, ""


def build_drift_repair_prompt(item, failed_question, drift_info, answer_type):
    """Build repair prompt for drift failures.

    Very specific instruction: rewrite so the expected answer is naturally the gold_answer_phrase.
    """
    events = item["events"]
    path_str = " -> ".join(f'"{e["trigger"]}"' for e in events)
    final = events[-1]["trigger"]
    prior_events = [e["trigger"] for e in events[:-1]]
    prior_list = ", ".join(f'"{t}"' for t in prior_events)
    ctx = fmt_ctx(item.get("supporting_sentences", []))
    answer_phrase = item.get("gold_answer_phrase", final)
    start = events[0]["trigger"]
    middle_triggers = [e["trigger"] for e in events[1:-1]]
    allowed = _get_allowed_heads(answer_type)
    allowed_str = "\n".join(f'  - "{h}"' for h in allowed[:5])

    return f"""Your question was rejected because it drifts away from the expected answer.

Rejected question: "{failed_question}"
Problem: {drift_info}

The expected answer is: "{answer_phrase}"

Rewrite the question so the expected answer is naturally: "{answer_phrase}".

The question must begin with one of these allowed openings:
{allowed_str}

Context:
{ctx}

Event path (reference): {path_str}

=== RULES ===
1. You MAY mention the starting event "{start}" or describe it in other words.
2. Do NOT mention intermediate events ({", ".join(f'"{t}"' for t in middle_triggers)}) or the final event "{final}".
3. Ask about the FINAL RESULT/OUTCOME — NOT about intermediate causes or reactions.
4. The question must require reading 3+ context sentences to answer.
5. Do NOT use double questions (no "How did X, and what Y?").
6. Use a SINGLE "What" question focused on the final answer.
7. Do NOT copy the answer phrase into the question.
8. BANNED words: outcry, inquiry, campaign, decision, influence, response, reason, why, how did, led to, contributed to

GOOD: "What [specific result] resulted from [entity]'s {start}?"  (answer = "{answer_phrase}")
BAD: "How did {start} influence [intermediate event]?"  (asks about intermediate, not final answer)
BAD: "What outcry followed the destruction?"  (asks about intermediate event)

Output: {{"question": "...", "answer": "{answer_phrase}", "reasoning_type": "drift_repair", "hidden_path_events": ["event_id", ...], "expected_steps": "3+"}}"""


# ── Hard path suitability filter ────────────────────────────
def check_hard_path_suitability(item):
    """Check if a Hard path is suitable for Hard question generation.

    Rejects paths with:
    - date-only gold_answer_phrase
    - weak/truncated phrase (< 3 words)
    - phrase that looks answerable from a single local sentence
    - phrase that is not a natural answer to an outcome/restriction/action question

    Returns (suitable: bool, reason: str).
    """
    answer_phrase = item.get("gold_answer_phrase", "")
    if not answer_phrase:
        return True, "no phrase to check"

    phrase_lower = answer_phrase.lower().strip()
    words = phrase_lower.split()

    # Date-only phrase: "signed on 30 March", "January 2003", etc.
    date_patterns = [
        r'^(signed|agreed|concluded|ended)\s+on\s+\d',
        r'^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}$',
        r'^\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)',
        r'^\d{4}$',
    ]
    for dp in date_patterns:
        if re.match(dp, phrase_lower):
            return False, f"date_fragment: {answer_phrase}"

    # Truncated phrase: too short or incomplete
    if len(words) < 2:
        return False, f"truncated_phrase: {answer_phrase}"

    # Phrase ending with preposition (incomplete)
    if phrase_lower.endswith((" of ", " in ", " on ", " at ", " to ", " for ", " with ")):
        return False, f"incomplete_phrase: {answer_phrase}"

    # Phrase starting with lowercase (likely mid-sentence fragment)
    if answer_phrase[0].islower() and not answer_phrase.startswith(("signed", "agreed", "closed", "opened")):
        return False, f"fragment_starts_lowercase: {answer_phrase}"

    return True, "suitable"


def generate_with_retry_hardaware(item, max_attempts=5):
    """Generate with difficulty-aware prompt + hard post-validation.
    Retries up to max_attempts times on empty/parse-fail. Checks path binding.
    Returns (result_dict, num_attempts).
    """
    diff = item["difficulty"]
    gold = item.get("answer_trigger", "")
    events = item.get("events", [])

    if diff == "Easy":
        prompt_fn = prompt_pathqg_easy
    elif diff == "Medium":
        prompt_fn = prompt_pathqg_medium
    else:
        prompt_fn = prompt_pathqg_hard_implicit

    question = ""
    rt = "error"
    g_ok, g_reason = False, "not attempted"
    covered_indices = []
    attempts = 0
    generation_error = False
    all_attempt_prompts = []
    all_attempt_raws = []
    last_align_check = "not_checked"

    for attempt in range(max_attempts):
        attempts = attempt + 1

        if attempt == 0:
            prompt = prompt_fn(item)
            temp = 0.1
        else:
            prompt = build_repair_prompt(item, question, g_reason, diff, covered_indices)
            temp = 0.1 + min(attempt * 0.1, 0.3)

        gen, raw = generate_one(prompt, temperature=temp)
        all_attempt_prompts.append(prompt)
        all_attempt_raws.append(raw or "")

        if gen is None:
            question = ""
            rt = "error"
            g_ok, g_reason = False, "parse error"
            continue

        question = gen.get("question", "") if isinstance(gen, dict) else ""
        rt = gen.get("reasoning_type", "unknown") if isinstance(gen, dict) else "error"

        # Validate hidden_path_events for implicit chain
        if diff == "Hard" and isinstance(gen, dict) and "hidden_path_events" in gen:
            valid_ids = {e.get("id", "") for e in events}
            raw_ids = gen.get("hidden_path_events", [])
            if isinstance(raw_ids, list):
                valid_hidden = [eid for eid in raw_ids if eid in valid_ids]
            else:
                valid_hidden = []
            gen["hidden_path_events"] = valid_hidden

        if not question:
            g_ok, g_reason = False, "empty"
            continue

        g_ok, g_reason = grammar_filter(question)

        if g_ok and gold and gold.lower() in question.lower():
            g_ok, g_reason = False, "trigger leakage"

        # Hard implicitness check at generation time: only catch extremely over-explicit (3+)
        # The filter pipeline enforces the real constraint (max 1)
        if g_ok and diff == "Hard":
            explicit_count = count_explicit_prior_triggers(question, events)
            if explicit_count >= 3:
                g_ok, g_reason = False, f"too_explicit: {explicit_count} prior triggers in question"

        if g_ok and diff == "Hard":
            g_ok, g_reason = validate_hard_question(question, events, gold)

        if g_ok:
            # For Hard (implicit chain), relax path_binding to 1 trigger
            # since the design intentionally avoids naming triggers.
            # The LLM path_coverage_judge in the filter pipeline enforces real coverage.
            effective_diff = diff
            if diff == "Hard":
                effective_diff = "Medium"  # Medium requires 1 prior event
            pb_ok, covered_indices, pb_reason = check_path_binding(question, events, effective_diff)
            if not pb_ok:
                g_ok, g_reason = False, f"path_binding: {pb_reason}"

        if g_ok:
            break

        if g_reason not in REPAIRABLE_REASONS:
            if not any(g_reason.startswith(r) for r in ["only ", "banned", "path_binding"]):
                break

    if not g_ok and not question:
        generation_error = True

    pb_method = "lexical_pass" if g_ok else "fail"

    return {
        "item_id": item.get("_item_id", 0),
        "doc_id": item.get("doc_id", ""),
        "difficulty": diff,
        "method": "PathQG-HardAware",
        "generated_question": question,
        "gold_answer_trigger": gold,
        "gold_answer_phrase": item.get("gold_answer_phrase", ""),
        "gold_answer_sentence": item.get("gold_answer_sentence", ""),
        "gold_event_type": item.get("gold_event_type", ""),
        "answer_phrase_status": item.get("answer_phrase_status", "unknown"),
        "reasoning_type": rt,
        "grammar_pass": g_ok,
        "grammar_reason": g_reason,
        "retry_attempts": attempts,
        "generation_error": generation_error,
        "covered_event_indices": covered_indices,
        "path_binding_method": pb_method,
        "events": events,
        "supporting_sentences": item.get("supporting_sentences", []),
        "relation_subtypes": item.get("relation_subtypes", []),
        "generation_prompts": all_attempt_prompts,
        "generation_raw_responses": all_attempt_raws,
        "hidden_path_events": gen.get("hidden_path_events", []) if isinstance(gen, dict) else [],
        "expected_steps": gen.get("expected_steps", "") if isinstance(gen, dict) else "",
        "answer_alignment_check": last_align_check,
    }, attempts


# ── Multi-strategy Hard generation ─────────────────────────

STRATEGY_PROMPT_MAP = {
    "hidden_endpoint": prompt_hidden_endpoint,
    "relation_composition": prompt_relation_composition,
    "contrastive": prompt_contrastive,
    "missing_bridge": prompt_missing_bridge,
    "implicit_chain": prompt_pathqg_hard_implicit,
}


def _check_answer_alignment(question, gold_answer_phrase):
    """Quick sanity check: reject only CLEARLY incompatible question-answer pairs.
    Returns (aligned: bool, raw_response: str).
    Hard questions naturally have indirect framing — this check only catches
    blatant type mismatches (e.g., "when" question with person-name answer).
    """
    if not gold_answer_phrase or not question:
        return True, "skipped"
    q_lower = question.lower().strip()
    a_lower = gold_answer_phrase.lower().strip()
    # Reject "when" questions with non-date answers
    if q_lower.startswith("when ") and not any(c.isdigit() for c in a_lower):
        return False, "when question, non-date answer"
    # Reject "who" questions with non-person answers
    if q_lower.startswith("who ") and len(a_lower.split()) > 6:
        return False, "who question, long answer"
    # Reject "where" questions with non-location answers
    if q_lower.startswith("where ") and any(kw in a_lower for kw in ["signed", "agreed", "forbade", "conducted"]):
        return False, "where question, action answer"
    return True, "compatible"


def generate_multi_strategy(item, strategy_name, max_attempts=5, model_config=None):
    """Generate a Hard question using a specific strategy.
    Same validation chain as generate_with_retry_hardaware.
    Returns (result_dict, num_attempts).

    model_config is accepted but not used for generation-time alignment checks.
    LLM alignment judge runs post-generation in the pilot script.
    """
    prompt_fn = STRATEGY_PROMPT_MAP.get(strategy_name)
    if prompt_fn is None:
        raise ValueError(f"Unknown strategy: {strategy_name}. Choose from {list(STRATEGY_PROMPT_MAP)}")

    gold = item.get("answer_trigger", "")
    events = item.get("events", [])
    diff = "Hard"
    gold_answer_phrase = item.get("gold_answer_phrase", "")
    gold_event_type = item.get("gold_event_type", events[-1].get("type", "") if events else "")

    # Infer answer type for drift checks
    answer_type = _infer_answer_type(gold_answer_phrase, gold_event_type)

    question = ""
    rt = "error"
    g_ok, g_reason = False, "not attempted"
    covered_indices = []
    attempts = 0
    generation_error = False
    all_attempt_prompts = []
    all_attempt_raws = []
    last_align_check = "not_checked"
    drift_check_fail = 0
    drift_repaired = 0

    for attempt in range(max_attempts):
        attempts = attempt + 1

        if attempt == 0:
            prompt = prompt_fn(item)
            temp = 0.2
        else:
            # Use drift repair if drift detected
            if g_reason and "drift" in g_reason:
                prompt = build_drift_repair_prompt(
                    item, question, drift_info, answer_type
                )
            elif g_reason and "misaligned" in g_reason:
                prompt = build_alignment_repair_prompt(
                    item, question, g_reason, diff
                )
            else:
                prompt = build_repair_prompt(item, question, g_reason, diff, covered_indices)
            temp = 0.2 + min(attempt * 0.1, 0.3)

        gen, raw = generate_one(prompt, temperature=temp)
        all_attempt_prompts.append(prompt)
        all_attempt_raws.append(raw or "")

        if gen is None:
            question = ""
            rt = "error"
            g_ok, g_reason = False, "parse error"
            continue

        question = gen.get("question", "") if isinstance(gen, dict) else ""
        rt = gen.get("reasoning_type", strategy_name) if isinstance(gen, dict) else "error"

        if diff == "Hard" and isinstance(gen, dict) and "hidden_path_events" in gen:
            valid_ids = {e.get("id", "") for e in events}
            raw_ids = gen.get("hidden_path_events", [])
            if isinstance(raw_ids, list):
                valid_hidden = [eid for eid in raw_ids if eid in valid_ids]
            else:
                valid_hidden = []
            gen["hidden_path_events"] = valid_hidden

        if not question:
            g_ok, g_reason = False, "empty"
            continue

        # ── Grammar check ──
        g_ok, g_reason = grammar_filter(question)

        # ── Trigger leakage ──
        if g_ok and gold and gold.lower() in question.lower():
            g_ok, g_reason = False, "trigger leakage"

        # ── Double question ban ──
        if g_ok:
            is_dq, dq_reason = _is_double_question(question)
            if is_dq:
                g_ok, g_reason = False, dq_reason

        # ── Drift check (rule-based) ──
        if g_ok:
            drifted, drift_info = _check_question_answer_drift(question, answer_type, gold_answer_phrase)
            if drifted:
                drift_check_fail += 1
                g_ok, g_reason = False, f"drift: {drift_info}"
                # Track if this was repaired on a subsequent attempt
                if attempt > 0 and drift_repaired < drift_check_fail:
                    drift_repaired += 1

        # ── Explicitness check ──
        if g_ok:
            explicit_count = count_explicit_prior_triggers(question, events)
            if explicit_count >= 3:
                g_ok, g_reason = False, f"too_explicit: {explicit_count} prior triggers in question"

        # ── Hard validation ──
        if g_ok:
            g_ok, g_reason = validate_hard_question(question, events, gold)

        # ── Path binding ──
        if g_ok:
            effective_diff = "Medium"  # relax path_binding for Hard
            pb_ok, covered_indices, pb_reason = check_path_binding(question, events, effective_diff)
            if not pb_ok:
                g_ok, g_reason = False, f"path_binding: {pb_reason}"

        # ── Rule-based alignment check ──
        if g_ok:
            aligned, align_resp = _check_answer_alignment(question, gold)
            last_align_check = "aligned" if aligned else f"misaligned: {align_resp}"
            if not aligned:
                g_ok, g_reason = False, f"answer_misaligned: {align_resp}"

        if g_ok:
            break

        if g_reason not in REPAIRABLE_REASONS:
            if not any(g_reason.startswith(r) for r in ["only ", "banned", "path_binding", "answer_misaligned", "drift"]):
                break

    if not g_ok and not question:
        generation_error = True

    pb_method = "lexical_pass" if g_ok else "fail"

    return {
        "item_id": item.get("_item_id", 0),
        "doc_id": item.get("doc_id", ""),
        "difficulty": diff,
        "method": "PathQG-HardAware",
        "hard_strategy": strategy_name,
        "generated_question": question,
        "gold_answer_trigger": gold,
        "gold_answer_phrase": item.get("gold_answer_phrase", ""),
        "gold_answer_sentence": item.get("gold_answer_sentence", ""),
        "gold_event_type": item.get("gold_event_type", ""),
        "answer_phrase_status": item.get("answer_phrase_status", "unknown"),
        "inferred_answer_type": answer_type,
        "reasoning_type": rt,
        "grammar_pass": g_ok,
        "grammar_reason": g_reason,
        "retry_attempts": attempts,
        "generation_error": generation_error,
        "covered_event_indices": covered_indices,
        "path_binding_method": pb_method,
        "drift_check_fail": drift_check_fail,
        "drift_repaired": drift_repaired,
        "events": events,
        "supporting_sentences": item.get("supporting_sentences", []),
        "relation_subtypes": item.get("relation_subtypes", []),
        "generation_prompts": all_attempt_prompts,
        "generation_raw_responses": all_attempt_raws,
        "hidden_path_events": gen.get("hidden_path_events", []) if isinstance(gen, dict) else [],
        "expected_steps": gen.get("expected_steps", "") if isinstance(gen, dict) else [],
        "answer_alignment_check": last_align_check,
    }, attempts
