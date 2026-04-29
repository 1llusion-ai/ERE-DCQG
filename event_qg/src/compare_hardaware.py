"""
PathQG Hard-Aware: difficulty-controlled question generation with monotonicity enforcement.

Changes from compare.py:
- Difficulty-specific few-shot examples and constraints
- Hard: requires binding 2+ prior events, bans shortcut phrases
- Medium: requires start event + 1 intermediate event mention
- Easy: 1-hop OK, simple questions
- New path_faithfulness_judge: NeedIntermediateEvents, EvidenceHopsUsed, CanAnswerFromSingleSentence
- Difficulty monotonicity: Easy > Medium > Hard on solver_correct
"""
import json
import time
import random
import os
import sys
import re
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent))

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in open(env_path):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

SILICONFLOW_API_URL = os.environ.get("SILICONFLOW_API_URL", "")
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-32B-Instruct")

from evaluator_v2 import grammar_filter, evaluate_item, _call_api, _normalize, _fuzzy_match


# ═══════════════════════════════════════════════════════════════
# ENHANCED FEW-SHOT EXAMPLES (difficulty-specific)
# ═══════════════════════════════════════════════════════════════

FEW_SHOT_EASY = """Example 1 (Easy — 1-hop question):
Events: "attack" → "destroy"
Context: [S0] The army launched an attack on the city. [S1] The city was destroyed.
Question: "What happened to the city after the army attacked it?"
- This is Easy: asks about one event directly following another.
- Uses a simple "after X, what happened to Y?" pattern."""

FEW_SHOT_MEDIUM = """Example 2 (Medium — must reference start event + one intermediate event):
Events: "resign" → "appoint" → "implement"
Context: [S0] The CEO resigned. [S1] The board appointed a new leader. [S2] The new CEO implemented reforms.
Question: "What did the new CEO implement after the board appointed a replacement following the CEO's resignation?"
- This is Medium: the question mentions both "resignation" and "appointed", requiring the solver to connect two steps.
- The answer event ("implement") is the final event, not mentioned in the question."""

FEW_SHOT_HARD = """Example 3 (Hard — must bind 2+ prior events in the question):
Events: "announce" → "protest" → "cancel"
Context: [S0] The government announced budget cuts. [S1] Citizens protested the decision. [S2] Officials canceled the policy.
Question: "After the government announced budget cuts and citizens protested, what did officials do regarding the policy?"
- This is Hard: the question EXPLICITLY names two prior events ("announced budget cuts" AND "citizens protested").
- The solver must understand the causal chain: announcement → protest → cancellation.
- The answer cannot be found by reading only the last sentence — you need to understand the full chain.
- Note: the gold answer "cancel" is NEVER mentioned in the question."""

# ═══════════════════════════════════════════════════════════════
# BANNED SHORTCUT PHRASES (Hard only)
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# DIFFICULTY-AWARE PROMPTS
# ═══════════════════════════════════════════════════════════════

def fmt_ctx(supporting):
    return "\n".join(s if isinstance(s, str) else f"[S{s[0]}] {s[1]}" for s in supporting)


def prompt_pathqg_easy(item):
    """Easy: 1-hop, simple question. Use the path but only require one connection."""
    events = item["events"]
    path_str = " -> ".join(f'"{e["trigger"]}"' for e in events)
    final = events[-1]["trigger"]
    first = events[0]["trigger"]
    ctx = fmt_ctx(item.get("supporting_sentences", []))

    return f"""{FEW_SHOT_EASY}

Now generate an Easy question:
Difficulty: Easy
Events: {path_str}
Context: {ctx}

Requirements for Easy:
- Ask about what happened to someone/something after a single event.
- A 1-hop question is acceptable.
- The answer is the final event: "{final}".
- You MAY mention the first event "{first}" or another event as context.
- Question must start with a question word (What/Who/When/Where/Why/How) and end with "?".
- Do NOT use the word "{final}" or its direct synonyms in the question.
- Output ONLY: {{"question": "...", "reasoning_type": "direct"}}"""


def prompt_pathqg_medium(item):
    """Medium: must reference start event + at least one intermediate event."""
    events = item["events"]
    path_str = " -> ".join(f'"{e["trigger"]}"' for e in events)
    final = events[-1]["trigger"]

    # Identify key events to reference
    if len(events) >= 3:
        start_event = events[0]["trigger"]
        mid_event = events[1]["trigger"]
        event_hint = f'Your question MUST reference both the start event "{start_event}" AND an intermediate event like "{mid_event}".'
    else:
        start_event = events[0]["trigger"]
        event_hint = f'Your question MUST reference the start event "{start_event}".'

    ctx = fmt_ctx(item.get("supporting_sentences", []))

    return f"""{FEW_SHOT_MEDIUM}

Now generate a Medium question:
Difficulty: Medium
Events: {path_str}
Context: {ctx}

Requirements for Medium:
- Ask about the final event in a way that requires understanding how it connects to earlier events.
- {event_hint}
- Your question MUST reference at least 2 different path events from the list above.
- The solver should need to understand the relationship between at least two events.
- The answer is the final event: "{final}". Do NOT use this word in the question.
- Question must start with a question word and end with "?".
- Output ONLY: {{"question": "...", "reasoning_type": "chain"}}"""


def prompt_pathqg_hard(item):
    """Hard: must bind 2+ prior events, cannot be answered from a single sentence."""
    events = item["events"]
    path_str = " -> ".join(f'"{e["trigger"]}"' for e in events)
    final = events[-1]["trigger"]

    # List all prior events for binding
    prior_events = [e["trigger"] for e in events[:-1]]
    prior_list = ", ".join(f'"{t}"' for t in prior_events)

    ctx = fmt_ctx(item.get("supporting_sentences", []))

    return f"""{FEW_SHOT_HARD}

Now generate a Hard question:
Difficulty: Hard
Events: {path_str}
Context: {ctx}

CRITICAL Requirements for Hard (you MUST follow ALL of these):
1. Your question MUST EXPLICITLY mention at least TWO events from: {prior_list}.
   - Don't just say "after the incident" — use the SPECIFIC event names or descriptions.
   - Example of GOOD: "After X announced cuts and Y protested, what did officials do?"
   - Example of BAD: "What was the final outcome after the incident?" ← NEVER DO THIS.

2. FORBIDDEN phrases (do NOT use ANY of these):
   - "final outcome" / "final result"
   - "what happened after the incident/event/crash"
   - "what action was taken"
   - "as a result" / "what was the result"

3. The question must NOT be answerable by reading only ONE sentence.
   - The solver must need to connect at least two pieces of information from different parts of the context.

4. The answer is the final event: "{final}". Do NOT use this word or its direct synonyms in the question.

5. Question must start with a question word and end with "?".

6. Output ONLY: {{"question": "...", "reasoning_type": "cross_sentence"}}"""


# ═══════════════════════════════════════════════════════════════
# HARD VALIDATION (post-generation check)
# ═══════════════════════════════════════════════════════════════

def validate_hard_question(question, events, gold_trigger):
    """Post-generation checks specific to Hard questions. Returns (passed, reason)."""
    # Check banned phrases
    has_banned, banned_phrase = check_banned_phrases(question)
    if has_banned:
        return False, f"banned phrase: {banned_phrase}"

    # Check that question mentions at least 2 prior events
    prior_triggers = [e["trigger"].lower() for e in events[:-1]]
    q_lower = question.lower()
    mentioned = [t for t in prior_triggers if t in q_lower]
    # Also check lemmatized/stem matches (words with length >= 4 share first 4 chars)
    for t in prior_triggers:
        if t in mentioned or len(t) < 4:
            continue
        for qw in q_lower.split():
            if len(qw) >= 4 and len(t) >= 4 and (qw[:4] == t[:4] or t in qw or qw in t):
                mentioned.append(t)
                break
    if len(set(mentioned)) < 2:
        return False, f"only {len(set(mentioned))} prior events mentioned, need >=2 from {prior_triggers}"

    # Check gold trigger not leaked
    if gold_trigger.lower() in q_lower:
        return False, "trigger leakage"

    return True, "pass"


# ═══════════════════════════════════════════════════════════════
# GENERATOR (reusing compare.py infrastructure)
# ═══════════════════════════════════════════════════════════════

def generate_one(prompt, temperature=0.1):
    """Single API call, return (json_dict_or_None, raw_text)."""
    import urllib.request
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {SILICONFLOW_API_KEY}"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Output ONLY a valid JSON object. Follow the question requirements EXACTLY."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 250,
        "stop": ["\n\n"],
    }
    try:
        req = urllib.request.Request(
            SILICONFLOW_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"]
    except Exception as e:
        return None, f"ERROR: {e}"

    gen = None
    try:
        gen = json.loads(text)
    except json.JSONDecodeError:
        try:
            s = text.index("{")
            e = text.rindex("}") + 1
            gen = json.loads(text[s:e])
        except (ValueError, json.JSONDecodeError):
            pass
    return gen, text


# ═══════════════════════════════════════════════════════════════
# REPAIR PROMPTS (hard-aware)
# ═══════════════════════════════════════════════════════════════

def _build_repair_prompt_hardaware(item, failed_question, failure_reason, difficulty):
    """Build repair prompt for hard-aware generation."""
    events = item["events"]
    path_str = " -> ".join(f'"{e["trigger"]}"' for e in events)
    final = events[-1]["trigger"]
    prior_events = [e["trigger"] for e in events[:-1]]
    prior_list = ", ".join(f'"{t}"' for t in prior_events)
    ctx = fmt_ctx(item.get("supporting_sentences", []))

    # Specific fix instructions
    fix_hints = {
        "no question mark": "End the question with a question mark (?)",
        "bad start": "Start with What/Who/When/Where/Why/How/Did/Was/Were/Is/Are",
        "word repetition": "Remove repeated words, write grammatically correct English",
        "trigger leakage": f'Do NOT use the word "{final}" or its synonyms anywhere in the question',
        "empty": "Write a complete question",
        "parse error": "Output ONLY valid JSON with 'question' and 'reasoning_type' keys",
        "too short": "Write a longer, more complete question that includes event details",
        "excessive repetition": "Remove repeated words, write naturally",
        "looping trigram": "Write naturally, avoid repeating the same phrase patterns",
    }

    # Add Hard-specific fixes
    if "banned phrase" in failure_reason:
        fix_hints["banned phrase"] = "Avoid template phrases like 'final outcome' or 'what happened after the incident'. Instead, use SPECIFIC event names from the path."
    if "only" in failure_reason and "prior events mentioned" in failure_reason:
        fix_hints["insufficient events"] = f"Mention at least TWO specific events from: {prior_list}. Name them explicitly in the question."
    if "path_binding" in failure_reason:
        min_req = {"Easy": 1, "Medium": 2, "Hard": 2}.get(difficulty, 1)
        fix_hints["path_binding"] = f"Your question must explicitly mention at least {min_req} prior events from the path: {path_str}. Use the specific event trigger words in your question."

    fix = fix_hints.get(failure_reason, f"Fix this issue: {failure_reason}")

    hard_extra = ""
    if difficulty == "Hard":
        hard_extra = f"""
HARD-SPECIFIC:
- Mention at least 2 prior events explicitly: {prior_list}
- Do NOT use: "final outcome", "what happened after the incident", "what action was taken"
- Question must require connecting info from multiple sentences"""

    return f"""Your previous output was rejected.
Rejected: "{failed_question}"
Issue: {failure_reason}
-> {fix}

Generate a corrected {difficulty} question:
Events: {path_str}
Context: {ctx}
- Answer is "{final}", do NOT mention it
{hard_extra}
- Question must start with a question word and end with ?
- Output ONLY: {{"question": "...", "reasoning_type": "..."}}"""


REPAIRABLE_REASONS = {
    "no question mark", "bad start", "word repetition", "trigger leakage",
    "empty", "parse error", "too short", "excessive repetition", "looping trigram",
    "not a dict", "no common English words",
    "banned phrase", "only 0 prior events mentioned, need >=2",
    "only 1 prior events mentioned, need >=2",
    "path_binding",
}


def _check_path_binding(question, events, difficulty):
    """Check if question text mentions enough path event triggers.
    Uses lexical matching (trigger in question, stem matching).
    Easy: 1, Medium: 2, Hard: 2 (prior events, not counting answer event).
    Returns (pass: bool, covered_indices: list, reason: str).
    """
    # For Hard, we check prior events only (exclude answer event = last)
    check_events = events
    if difficulty == "Hard":
        check_events = events[:-1]  # prior events only
    min_required = {"Easy": 1, "Medium": 2, "Hard": 2}.get(difficulty, 1)

    q_lower = question.lower()
    q_words = set(q_lower.split())
    covered = []

    for i, e in enumerate(check_events):
        trigger = e["trigger"].lower()
        # Direct match
        if trigger in q_lower:
            covered.append(i)
            continue
        # Stem match (first 4 chars for words >= 4)
        for qw in q_words:
            if len(qw) >= 4 and len(trigger) >= 4:
                if qw[:4] == trigger[:4] or trigger in qw or qw in trigger:
                    covered.append(i)
                    break

    covered = list(set(covered))
    if len(covered) >= min_required:
        return True, covered, f"covers {len(covered)}/{len(check_events)} events, need >= {min_required}"
    return False, covered, f"covers {len(covered)}/{len(check_events)} events, need >= {min_required}"


def generate_with_retry_hardaware(item, max_attempts=5):
    """Generate with difficulty-aware prompt + hard post-validation.
    Retries up to 5 times on empty/parse-fail. Checks path binding.
    """
    diff = item["difficulty"]
    gold = item.get("answer_trigger", "")
    num_events = len(item.get("events", []))

    if diff == "Easy":
        prompt_fn = prompt_pathqg_easy
    elif diff == "Medium":
        prompt_fn = prompt_pathqg_medium
    else:
        prompt_fn = prompt_pathqg_hard

    question = ""
    rt = "error"
    g_ok, g_reason = False, "not attempted"
    covered_indices = []
    attempts = 0
    generation_error = False
    events = item.get("events", [])

    for attempt in range(max_attempts):
        attempts = attempt + 1

        if attempt == 0:
            prompt = prompt_fn(item)
            temp = 0.1
        else:
            prompt = _build_repair_prompt_hardaware(item, question, g_reason, diff)
            temp = 0.1 + min(attempt * 0.1, 0.3)

        gen, raw = generate_one(prompt, temperature=temp)

        if gen is None:
            question = ""
            rt = "error"
            g_ok, g_reason = False, "parse error"
            continue

        question = gen.get("question", "") if isinstance(gen, dict) else ""
        rt = gen.get("reasoning_type", "unknown") if isinstance(gen, dict) else "error"

        if not question:
            g_ok, g_reason = False, "empty"
            continue

        # Grammar filter
        g_ok, g_reason = grammar_filter(question)

        # Trigger leakage
        if g_ok and gold and gold.lower() in question.lower():
            g_ok, g_reason = False, "trigger leakage"

        # Hard-specific validation
        if g_ok and diff == "Hard":
            g_ok, g_reason = validate_hard_question(question, item["events"], gold)

        # Path binding check (only if grammar passed)
        if g_ok:
            pb_ok, covered_indices, pb_reason = _check_path_binding(question, events, diff)
            if not pb_ok:
                g_ok, g_reason = False, f"path_binding: {pb_reason}"

        if g_ok:
            break

        if g_reason not in REPAIRABLE_REASONS:
            if not any(g_reason.startswith(r) for r in ["only ", "banned", "path_binding"]):
                break

    # If still no valid question after all attempts, mark generation_error
    if not g_ok and not question:
        generation_error = True

    return {
        "item_id": item.get("_item_id", 0),
        "doc_id": item.get("doc_id", ""),
        "difficulty": diff,
        "method": "PathQG-HardAware",
        "generated_question": question,
        "gold_answer_trigger": gold,
        "reasoning_type": rt,
        "grammar_pass": g_ok,
        "grammar_reason": g_reason,
        "retry_attempts": attempts,
        "generation_error": generation_error,
        "covered_event_indices": covered_indices,
        "events": events,
        "supporting_sentences": item.get("supporting_sentences", []),
        "relation_subtypes": item.get("relation_subtypes", []),
        "difficulty_score": item.get("difficulty_score", 0),
    }, attempts


# ═══════════════════════════════════════════════════════════════
# PATH-FAITHFULNESS JUDGE (Task 3)
# ═══════════════════════════════════════════════════════════════

def path_faithfulness_judge(question, path_events, supporting_sentences, difficulty):
    """
    Judge whether a question genuinely requires multi-hop reasoning across path events.
    Returns dict with:
        - need_intermediate_events: 0.0/0.5/1.0 (yes=1.0)
        - evidence_hops_used: 0.33/0.67/1.0 (1/2/3+)
        - can_answer_single_sentence: 0.0/0.5/1.0 (yes=1.0 = BAD for Hard)
        - hard_pass: bool (all 3 conditions met for Hard)
        - raw_judgment: string
    """
    path_str = " -> ".join(e["trigger"] for e in path_events)
    path_trigger_list = ", ".join(f'"{e["trigger"]}"' for e in path_events)
    final_trigger = path_events[-1]["trigger"] if path_events else "?"

    # Format context
    ctx_lines = []
    for i, s in enumerate(supporting_sentences):
        if isinstance(s, (list, tuple)):
            ctx_lines.append(f"[S{s[0]}] {s[1]}")
        else:
            ctx_lines.append(f"[S{i}] {s}")
    ctx_text = "\n".join(ctx_lines[:8])  # truncate

    prompt = f"""Context:
{ctx_text}

Question: "{question}"
Path events: {path_str}
Gold answer: "{final_trigger}"
Difficulty: {difficulty}

Answer yes/no for each question:
1. Does the question require understanding intermediate events (not just the last one) to answer correctly? (yes/no)
2. How many context sentences must the solver read to find the answer? (1/2/3+)
3. Can the question be answered correctly by reading only ONE sentence? (yes/no)

Reply: NEED= EVIDENCE= SINGLE= (e.g., "NEED=yes EVIDENCE=3+ SINGLE=no")"""

    resp = _call_api(prompt, temperature=0.0, max_tokens=30, model=JUDGE_MODEL, timeout=120)

    need_ie = 0.5
    hops = 0.67
    can_single = 0.5
    raw = resp or "NO_RESPONSE"

    if resp:
        resp_upper = resp.upper().replace(',', ' ')
        for part in resp_upper.split():
            if part.startswith('NEED='):
                need_ie = 1.0 if 'YES' in part else (0.0 if 'NO' in part else 0.5)
            elif part.startswith('EVIDENCE='):
                val = part.split('=', 1)[1].strip()
                if '3' in val and '+' in val:
                    hops = 1.0
                elif '2' in val:
                    hops = 0.67
                elif '1' in val:
                    hops = 0.33
            elif part.startswith('SINGLE='):
                can_single = 1.0 if 'YES' in part else (0.0 if 'NO' in part else 0.5)

    hard_pass = (need_ie >= 0.5 and hops >= 0.67 and can_single <= 0.5)

    return {
        "need_intermediate_events": round(need_ie, 2),
        "evidence_hops_used": round(hops, 2),
        "can_answer_single_sentence": round(can_single, 2),
        "hard_pass": hard_pass,
        "raw_judgment": raw,
    }


# ═══════════════════════════════════════════════════════════════
# FULL EVALUATION WITH PATH-FAITHFULNESS
# ═══════════════════════════════════════════════════════════════

def evaluate_item_with_faithfulness(r, skip_quality=False):
    """Evaluate one item with v2 metrics + path faithfulness."""
    # Standard v2 evaluation
    r = evaluate_item(r, skip_judge=skip_quality)

    # Path faithfulness (for all items)
    q = r["generated_question"]
    path_events = r.get("events", [])
    sents = r.get("supporting_sentences", [])
    diff = r["difficulty"]

    faith = path_faithfulness_judge(q, path_events, sents, diff)
    r["faith_need_intermediate"] = faith["need_intermediate_events"]
    r["faith_evidence_hops"] = faith["evidence_hops_used"]
    r["faith_can_answer_single"] = faith["can_answer_single_sentence"]
    r["faith_hard_pass"] = faith["hard_pass"]
    r["faith_raw"] = faith["raw_judgment"]

    return r


def evaluate_file_with_faithfulness(input_path, output_path, max_items=None):
    """Evaluate all grammar-passed items with v2 + faith judge."""
    with open(input_path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    passed = [r for r in items if r.get("grammar_pass", False)]
    print(f"  {len(passed)}/{len(items)} grammar-passed")

    if max_items:
        random.seed(42)
        passed = random.sample(passed, min(max_items, len(passed)))

    if Path(output_path).exists():
        Path(output_path).unlink()

    scored = []
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, r in enumerate(passed):
            r = evaluate_item_with_faithfulness(r)
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out_f.flush()
            scored.append(r)

            if (i + 1) % 20 == 0:
                n = len(scored)
                avg_com = sum(s["composite"] for s in scored) / n
                avg_need = sum(s["faith_need_intermediate"] for s in scored) / n
                avg_single = sum(s["faith_can_answer_single"] for s in scored) / n
                hard_pass = sum(1 for s in scored if s.get("faith_hard_pass", False))
                print(f"  [{i+1}/{len(passed)}] comp={avg_com:.3f} need={avg_need:.2f} single={avg_single:.2f} hard_pass={hard_pass}", flush=True)

            time.sleep(0.15)

    return scored


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", default="event_qg/outputs/stage2_generation_results.jsonl")
    parser.add_argument("--output_dir", default="event_qg/outputs")
    parser.add_argument("--n_per_level", type=int, default=100)
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    random.seed(42)

    # Step 1: Sample items (same seed as compare.py for fair comparison)
    print("=" * 70)
    print("Step 1: Sampling items (seed=42, 100/level)")
    print("=" * 70)
    with open(args.results_file, encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f]

    valid = [r for r in all_items if r.get("filter_pass", False)]
    by_level = defaultdict(list)
    for item in valid:
        by_level[item["difficulty"]].append(item)

    sampled = []
    for level in ["Easy", "Medium", "Hard"]:
        pool = by_level[level]
        n = min(args.n_per_level, len(pool))
        sampled.extend(random.sample(pool, n))
    random.shuffle(sampled)

    print(f"Sampled {len(sampled)} items")
    level_counts = Counter(r["difficulty"] for r in sampled)
    print(f"  By level: {dict(level_counts)}")

    # Step 2: Generate PathQG-HardAware
    gen_path = output_dir / "compare_PathQG_generated_retry_hardaware.jsonl"
    if not args.skip_generation:
        print(f"\n{'='*70}")
        print("Step 2: Generating PathQG-HardAware")
        print(f"{'='*70}")

        if gen_path.exists():
            gen_path.unlink()

        results = []
        raw_pass = 0
        total_attempts = 0

        with open(gen_path, "w", encoding="utf-8") as out_f:
            for i, item in enumerate(sampled):
                item["_item_id"] = i

                r, attempts = generate_with_retry_hardaware(item, max_attempts=3)
                total_attempts += attempts
                if attempts == 1 and r["grammar_pass"]:
                    raw_pass += 1

                results.append(r)
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                out_f.flush()

                if (i + 1) % 30 == 0:
                    n_pass = sum(1 for r in results if r["grammar_pass"])
                    print(f"  [{i+1}/{len(sampled)}] pass={n_pass} ({n_pass/(i+1)*100:.0f}%)", flush=True)

                time.sleep(0.1)

        n_pass = sum(1 for r in results if r["grammar_pass"])
        print(f"\nPathQG-HardAware: {n_pass}/{len(results)} passed ({n_pass/len(results)*100:.0f}%)")
        print(f"  Raw pass: {raw_pass} ({raw_pass/len(results)*100:.0f}%)")
        print(f"  Avg attempts: {total_attempts/len(results):.2f}")

        # Show fail reasons
        fails = Counter(r["grammar_reason"] for r in results if not r["grammar_pass"])
        print("  Top fail reasons:")
        for reason, count in fails.most_common(8):
            print(f"    {reason}: {count}")

    else:
        with open(gen_path, encoding="utf-8") as f:
            results = [json.loads(line) for line in f]
        print(f"Loaded {len(results)} existing results")

    # Step 3: Evaluate
    if not args.skip_evaluation:
        eval_path = output_dir / "compare_PathQG_evaluated_retry_hardaware_v2.jsonl"
        print(f"\n{'='*70}")
        print("Step 3: Evaluating PathQG-HardAware (v2 + faith judge)")
        print(f"{'='*70}")

        scored = evaluate_file_with_faithfulness(gen_path, eval_path)
        n = len(scored)

        print(f"\n{'='*70}")
        print(f"PATHQG-HARDAWARE RESULTS ({n} evaluated)")
        print(f"{'='*70}")

        # Overall
        hit = sum(s["target_event_hit"] for s in scored) / n
        ans = sum(s["judge_answerable"] for s in scored) / n
        cor = sum(s["judge_solver_correct"] for s in scored) / n
        sup = sum(s["judge_support_covered"] for s in scored) / n
        flu = sum(s["quality_fluency"] for s in scored) / n
        pat = sum(s["quality_path_relevance"] for s in scored) / n
        dif = sum(s["quality_difficulty_alignment"] for s in scored) / n
        comp = sum(s["composite"] for s in scored) / n
        need = sum(s["faith_need_intermediate"] for s in scored) / n
        single = sum(s["faith_can_answer_single"] for s in scored) / n
        hops = sum(s["faith_evidence_hops"] for s in scored) / n
        hard_pass = sum(1 for s in scored if s.get("faith_hard_pass", False))

        print(f"  target_event_hit:         {hit:.3f}")
        print(f"  judge_solver_correct:     {cor:.3f}")
        print(f"  judge_answerable:         {ans:.3f}")
        print(f"  judge_support_covered:    {sup:.3f}")
        print(f"  quality_fluency:          {flu:.3f}")
        print(f"  quality_path_relevance:   {pat:.3f}")
        print(f"  quality_difficulty:       {dif:.3f}")
        print(f"  composite:                {comp:.3f}")
        print(f"  faith_need_intermediate:  {need:.3f}")
        print(f"  faith_can_answer_single:  {single:.3f}")
        print(f"  faith_evidence_hops:      {hops:.3f}")
        print(f"  faith_hard_pass (Hard):   {hard_pass}")

        # By difficulty
        print(f"\n{'Difficulty':<10} {'N':>5} {'Hit':>6} {'SolCorr':>7} {'Ans':>6} {'Comp':>6} {'NeedIE':>6} {'Single':>6} {'Hops':>6}")
        print("-" * 72)
        for level in ["Easy", "Medium", "Hard"]:
            items_l = [s for s in scored if s["difficulty"] == level]
            if not items_l:
                continue
            n_l = len(items_l)
            hit_l = sum(s["target_event_hit"] for s in items_l) / n_l
            cor_l = sum(s["judge_solver_correct"] for s in items_l) / n_l
            ans_l = sum(s["judge_answerable"] for s in items_l) / n_l
            comp_l = sum(s["composite"] for s in items_l) / n_l
            need_l = sum(s["faith_need_intermediate"] for s in items_l) / n_l
            single_l = sum(s["faith_can_answer_single"] for s in items_l) / n_l
            hops_l = sum(s["faith_evidence_hops"] for s in items_l) / n_l
            hp_l = sum(1 for s in items_l if s.get("faith_hard_pass", False))
            print(f"{level:<10} {n_l:>5} {hit_l:>6.3f} {cor_l:>7.3f} {ans_l:>6.3f} {comp_l:>6.3f} {need_l:>6.3f} {single_l:>6.3f} {hops_l:>6.3f}")

        # Monotonicity check
        print(f"\n{'='*70}")
        print("DIFFICULTY MONOTONICITY CHECK")
        print(f"{'='*70}")
        easy_cor = sum(s["judge_solver_correct"] for s in scored if s["difficulty"] == "Easy")
        easy_n = sum(1 for s in scored if s["difficulty"] == "Easy")
        med_cor = sum(s["judge_solver_correct"] for s in scored if s["difficulty"] == "Medium")
        med_n = sum(1 for s in scored if s["difficulty"] == "Medium")
        hard_cor = sum(s["judge_solver_correct"] for s in scored if s["difficulty"] == "Hard")
        hard_n = sum(1 for s in scored if s["difficulty"] == "Hard")

        e = easy_cor / easy_n if easy_n else 0
        m = med_cor / med_n if med_n else 0
        h = hard_cor / hard_n if hard_n else 0
        print(f"  solver_correct: Easy={e:.3f} > Medium={m:.3f} > Hard={h:.3f}")
        if e >= m >= h:
            print("  MONOTONICITY: PASS (Easy >= Medium >= Hard)")
        elif e >= h:
            print("  MONOTONICITY: PARTIAL (Easy >= Hard, but Medium not monotonic)")
        else:
            print("  MONOTONICITY: FAIL (Easy < Hard)")

        # Faith metrics by difficulty
        print(f"\n  faith_need_intermediate:")
        e_n = sum(s["faith_need_intermediate"] for s in scored if s["difficulty"] == "Easy") / max(easy_n, 1)
        m_n = sum(s["faith_need_intermediate"] for s in scored if s["difficulty"] == "Medium") / max(med_n, 1)
        h_n = sum(s["faith_need_intermediate"] for s in scored if s["difficulty"] == "Hard") / max(hard_n, 1)
        print(f"    Easy={e_n:.3f} Medium={m_n:.3f} Hard={h_n:.3f} (want: Hard > Medium > Easy)")

        print(f"  faith_can_answer_single:")
        e_s = sum(s["faith_can_answer_single"] for s in scored if s["difficulty"] == "Easy") / max(easy_n, 1)
        m_s = sum(s["faith_can_answer_single"] for s in scored if s["difficulty"] == "Medium") / max(med_n, 1)
        h_s = sum(s["faith_can_answer_single"] for s in scored if s["difficulty"] == "Hard") / max(hard_n, 1)
        print(f"    Easy={e_s:.3f} Medium={m_s:.3f} Hard={h_s:.3f} (want: Easy > Medium > Hard)")

        print(f"  hard_pass rate:")
        for level in ["Easy", "Medium", "Hard"]:
            items_l = [s for s in scored if s["difficulty"] == level]
            hp = sum(1 for s in items_l if s.get("faith_hard_pass", False)) / max(len(items_l), 1)
            print(f"    {level}: {hp:.3f}")

    print(f"\n=== PathQG-HardAware Complete ===")


if __name__ == "__main__":
    main()
