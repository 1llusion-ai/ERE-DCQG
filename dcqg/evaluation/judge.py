"""Evaluator v2: improved evaluation with 3-way LLM judge (from evaluator_v2.py).

Changes from v1:
- eval_answerability -> target_event_hit (text similarity)
- New LLM judge: answerable / solver_correct / support_covered
- Fluency + path_relevance + difficulty_alignment (32B judge)
- Fixed random seeds, no file append issues
"""
import json
import re
import time
import random
from pathlib import Path

from dcqg.utils.api_client import call_api, call_openai_compatible
from dcqg.utils.text import normalize, fuzzy_match, detect_loop
from dcqg.utils.config import get_api_config
from dcqg.question_filter.grammar import grammar_filter


# ── Solver ──────────────────────────────────────────────────
def solve(question, context):
    """Answer question from context. Returns short answer string."""
    ctx_lines = context.split("\n")
    if len(ctx_lines) > 8:
        context = "\n".join(ctx_lines[:8])
    prompt = f"""Context:
{context}

Question: {question}

Answer in 1-5 words only."""
    for _ in range(2):
        resp = call_api(prompt,
                        system="Answer questions briefly. Output ONLY the answer.",
                        temperature=0.0, max_tokens=40, timeout=60)
        if resp:
            cleaned = detect_loop(resp)
            if cleaned and len(cleaned) < 100 and len(cleaned.split()) <= 10:
                q_words = set(normalize(question).split()) - {'the', 'a', 'an', 'is', 'was', 'were', 'did', 'what', 'who', 'when', 'where', 'why', 'how', 'after', 'in', 'to', 'of'}
                a_words = set(normalize(cleaned).split()) - {'the', 'a', 'an', 'is', 'was', 'were', 'did', 'what', 'who', 'when', 'where', 'why', 'how', 'after', 'in', 'to', 'of'}
                if len(a_words) >= 1 and not (a_words and a_words == (a_words & q_words)):
                    return cleaned
        time.sleep(0.2)
    return resp if resp else ""


# ── Target event hit (renamed from answerability) ────────────
def target_event_hit(solver_answer, gold_trigger):
    """Score 0-1: how well solver answer matches gold trigger."""
    match = fuzzy_match(solver_answer, gold_trigger)
    if match == 'exact':
        return 1.0, match
    if match == 'fuzzy':
        return 0.7, match
    if match == 'stem':
        return 0.4, match
    return 0.0, match


# ── LLM Judge v2: answerable / solver_correct / support_covered
def llm_judge_v2(question, context, gold_trigger, solver_answer):
    """
    3-way LLM judge using 32B model.
    Returns (answerable, solver_correct, support_covered) each 0-1.
    """
    judge_model = get_api_config()["JUDGE_MODEL"]

    # Truncate context
    ctx_lines = context.split("\n")
    if len(ctx_lines) > 10:
        ctx_short = "\n".join(ctx_lines[:10])
    else:
        ctx_short = context

    prompt = f"""Context:
{ctx_short}

Question: {question}
Gold answer: "{gold_trigger}"
Solver answer: "{solver_answer}"

Answer yes/no for each:
1. Answerable: Can someone answer the question using ONLY the context? (yes/no)
2. SolverCorrect: Does the solver answer match the gold answer semantically? (yes/no)
3. SupportCovered: Does the context contain the gold answer or direct evidence for it? (yes/no)

Reply: A= S= U= (e.g., "A=yes S=yes U=yes")"""
    resp = call_api(prompt, temperature=0.0, max_tokens=30, model=judge_model, timeout=120)

    answerable, solver_correct, support_covered = 0.5, 0.5, 0.5
    if resp:
        for part in resp.upper().replace(',', ' ').split():
            if part.startswith('A='):
                answerable = 1.0 if 'YES' in part else (0.0 if 'NO' in part else 0.5)
            elif part.startswith('S='):
                solver_correct = 1.0 if 'YES' in part else (0.0 if 'NO' in part else 0.5)
            elif part.startswith('U='):
                support_covered = 1.0 if 'YES' in part else (0.0 if 'NO' in part else 0.5)

    return answerable, solver_correct, support_covered


# ── Quality judge: fluency, path relevance, difficulty ──────
def quality_judge(question, path_events, difficulty):
    """32B judge: fluency, path relevance, difficulty alignment."""
    judge_model = get_api_config()["JUDGE_MODEL"]
    path_str = " → ".join(e["trigger"] for e in path_events)

    prompt = f"""Question: "{question}"
Path: {path_str}
Difficulty: {difficulty}

Score 1-3:
1. Fluency: 3=natural 2=minor errors 1=broken
2. Path: 3=uses multiple events 2=uses two 1=single event
3. Difficulty: 3=perfect {difficulty} 2=close 1=wrong

Reply: F= P= D= (e.g., "F=3 P=2 D=3")"""
    resp = call_api(prompt, temperature=0.0, max_tokens=30, model=judge_model, timeout=120)

    fluency, relevance, diff_align = 1/3, 1/3, 1/3
    if resp:
        for part in resp.replace(',', ' ').split():
            part = part.strip().upper()
            try:
                if part.startswith('F='):
                    fluency = int(re.findall(r'(\d)', part)[0]) / 3.0
                elif part.startswith('P='):
                    relevance = int(re.findall(r'(\d)', part)[0]) / 3.0
                elif part.startswith('D='):
                    diff_align = int(re.findall(r'(\d)', part)[0]) / 3.0
            except (ValueError, IndexError):
                pass

    return fluency, relevance, diff_align


# ── Full evaluation ─────────────────────────────────────────
def evaluate_item(r, skip_judge=False):
    """
    Evaluate one result item. Returns updated dict with scores.
    r must have: generated_question, supporting_sentences, gold_answer_trigger, events, difficulty
    """
    q = r["generated_question"]
    ctx = "\n".join(
        s if isinstance(s, str) else s[1]
        for s in r.get("supporting_sentences", [])
    )
    gold = r["gold_answer_trigger"]
    path_events = r.get("events", [])
    diff = r["difficulty"]

    # 1. Solver
    solver_ans = solve(q, ctx)
    r["solver_answer"] = solver_ans

    # 2. Target event hit (text similarity)
    hit_score, hit_method = target_event_hit(solver_ans, gold)
    r["target_event_hit"] = round(hit_score, 2)
    r["hit_method"] = hit_method

    # 3. LLM judge v2
    answerable, solver_correct, support_covered = llm_judge_v2(q, ctx, gold, solver_ans)
    r["judge_answerable"] = round(answerable, 2)
    r["judge_solver_correct"] = round(solver_correct, 2)
    r["judge_support_covered"] = round(support_covered, 2)

    # 4. Quality judge
    if not skip_judge:
        fluency, relevance, diff_align = quality_judge(q, path_events, diff)
        r["quality_fluency"] = round(fluency, 2)
        r["quality_path_relevance"] = round(relevance, 2)
        r["quality_difficulty_alignment"] = round(diff_align, 2)
        r["composite"] = round(
            0.25 * solver_correct +
            0.20 * answerable +
            0.15 * support_covered +
            0.15 * fluency +
            0.10 * relevance +
            0.15 * diff_align,
            3
        )
    else:
        r["quality_fluency"] = 0
        r["quality_path_relevance"] = 0
        r["quality_difficulty_alignment"] = 0
        r["composite"] = round(
            0.30 * solver_correct +
            0.25 * answerable +
            0.20 * support_covered +
            0.25 * hit_score,
            3
        )

    return r


def evaluate_file(input_path, output_path, max_items=None, skip_quality=False):
    """Evaluate all grammar-passed items in a file, save incrementally."""
    with open(input_path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    # Filter to grammar-passed items
    passed = [r for r in items if r.get("grammar_pass", r.get("filter_pass", False))]
    print(f"  {len(passed)}/{len(items)} grammar-passed")

    if max_items:
        random.seed(42)
        passed = random.sample(passed, min(max_items, len(passed)))

    # Delete output to prevent append duplication
    if Path(output_path).exists():
        Path(output_path).unlink()

    scored = []
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, r in enumerate(passed):
            r = evaluate_item(r, skip_judge=skip_quality)
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out_f.flush()
            scored.append(r)

            if (i + 1) % 30 == 0:
                n = len(scored)
                avg_hit = sum(s["target_event_hit"] for s in scored) / n
                avg_ans = sum(s["judge_answerable"] for s in scored) / n
                avg_cor = sum(s["judge_solver_correct"] for s in scored) / n
                avg_sup = sum(s["judge_support_covered"] for s in scored) / n
                avg_com = sum(s["composite"] for s in scored) / n
                print(f"  [{i+1}/{len(passed)}] hit={avg_hit:.2f} ans={avg_ans:.2f} cor={avg_cor:.2f} sup={avg_sup:.2f} comp={avg_com:.3f}")

            time.sleep(0.15)

    # Summary
    if scored:
        n = len(scored)
        print(f"\n  Summary ({n} items):")
        print(f"  target_event_hit:     {sum(s['target_event_hit'] for s in scored)/n:.3f}")
        print(f"  judge_answerable:     {sum(s['judge_answerable'] for s in scored)/n:.3f}")
        print(f"  judge_solver_correct: {sum(s['judge_solver_correct'] for s in scored)/n:.3f}")
        print(f"  judge_support_covered:{sum(s['judge_support_covered'] for s in scored)/n:.3f}")
        if not skip_quality:
            print(f"  quality_fluency:      {sum(s['quality_fluency'] for s in scored)/n:.3f}")
            print(f"  quality_path_relevance:{sum(s['quality_path_relevance'] for s in scored)/n:.3f}")
            print(f"  quality_difficulty:   {sum(s['quality_difficulty_alignment'] for s in scored)/n:.3f}")
        print(f"  composite:            {sum(s['composite'] for s in scored)/n:.3f}")

    return scored


# ═══════════════════════════════════════════════════════════════
# Independent Difficulty & Path Dependency Judges
# (ported from scripts/run_independent_difficulty_eval.py)
# Uses AIHUBMIX (GPT-4o-mini). Does NOT see target difficulty.
# ═══════════════════════════════════════════════════════════════

def _fmt_supporting(supporting_sentences, max_sents=6):
    """Format supporting_sentences as [S{id}] text lines.
    If max_sents is None, include all sentences."""
    sliced = supporting_sentences if max_sents is None else supporting_sentences[:max_sents]
    lines = []
    for i, sent in enumerate(sliced):
        if isinstance(sent, (list, tuple)) and len(sent) >= 2:
            lines.append(f"[S{sent[0]}] {sent[1]}")
        elif isinstance(sent, str):
            lines.append(f"[S{i}] {sent}")
    return "\n".join(lines) if lines else "No context available."


def _fmt_events(events):
    """Format events as numbered list."""
    lines = []
    for i, ev in enumerate(events):
        trigger = ev.get("trigger", "?")
        etype = ev.get("type", "?")
        lines.append(f"  {i+1}. {trigger} ({etype})")
    return "\n".join(lines) if lines else "No events."


def _fmt_events_with_roles(events):
    """Format events with PRIOR/FINAL labels."""
    lines = []
    for i, ev in enumerate(events):
        trigger = ev.get("trigger", "?")
        etype = ev.get("type", "?")
        eid = ev.get("id", "?")
        role = "FINAL" if i == len(events) - 1 else "PRIOR"
        lines.append(f"  {i+1}. id={eid} trigger=\"{trigger}\" type={etype} role={role}")
    return "\n".join(lines) if lines else "No events."


def _parse_judge_json(resp, expected_keys):
    """Robust JSON parser for judge responses. Returns dict or None."""
    if not resp:
        return None

    def _check(parsed):
        if isinstance(parsed, dict) and all(k in parsed for k in expected_keys):
            return parsed
        return None

    try:
        result = _check(json.loads(resp))
        if result:
            return result
    except json.JSONDecodeError:
        pass

    try:
        s_idx = resp.index("{")
        e_idx = resp.rindex("}") + 1
        result = _check(json.loads(resp[s_idx:e_idx]))
        if result:
            return result
    except (ValueError, json.JSONDecodeError):
        pass

    cleaned = resp.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'\s*```$', '', cleaned).strip()
    cleaned = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', cleaned)
    cleaned = cleaned.replace("'", '"')
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    try:
        result = _check(json.loads(cleaned))
        if result:
            return result
    except json.JSONDecodeError:
        pass

    partial = {}
    for key in expected_keys:
        m = re.search(rf'"{key}"\s*:\s*(".*?"|[\d.]+|true|false|\[.*?\])', resp, re.DOTALL)
        if m:
            val_str = m.group(1)
            try:
                partial[key] = json.loads(val_str)
            except json.JSONDecodeError:
                partial[key] = val_str.strip('"')
    if partial and all(k in partial for k in expected_keys):
        return partial

    return None


def _call_judge_api(prompt, model_config):
    """Call AIHUBMIX LLM for independent judging. Returns (response_text, error_or_None)."""
    try:
        resp = call_openai_compatible(
            prompt,
            api_url=model_config["api_url"],
            api_key=model_config["api_key"],
            model=model_config["model"],
            temperature=0.0,
            max_tokens=300,
            json_mode=True,
        )
        return resp, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _build_difficulty_prompt(item):
    """Build difficulty-only judge prompt. Does NOT include target difficulty."""
    question = item.get("generated_question", "")
    answer = item.get("gold_answer_phrase", "") or item.get("gold_answer_trigger", "")
    context = _fmt_supporting(item.get("supporting_sentences", []))
    events = _fmt_events(item.get("events", []))

    return f"""You are an expert difficulty evaluator for reading comprehension questions.

## Context
{context}

## Event path
{events}

## Question
"{question}"

## Expected answer
"{answer}"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace through a chain of events/facts to reach the answer

The key question: HOW MANY sequential reasoning steps must the solver make?

Reply as a single JSON object:
{{
  "predicted_difficulty": "Easy",
  "required_steps": "1",
  "single_sentence_answerable": "yes",
  "answerable": "yes",
  "final_event_consistent": "yes",
  "reason": "short explanation"
}}

Guidelines:
- predicted_difficulty: Easy, Medium, or Hard
- required_steps: "1", "2", or "3+"
- single_sentence_answerable: can the answer be found in a single sentence? "yes", "partial", or "no"
- answerable: is the question answerable from the context? "yes", "partial", or "no"
- final_event_consistent: does the question ask about the final event in the path? "yes", "partial", or "no"

Difficulty definitions (focus on REASONING STEPS, not just information count):

- Easy (1 step): The solver reads ONE sentence and extracts the answer directly. No chain reasoning.
  Example: "What did the army do?" → read the sentence about the army.

- Medium (2 steps): The solver connects 2 pieces of information from 2 different sentences. Simple A→B link.
  Example: "What happened after X?" → find X in sentence 1, find result in sentence 2.

- Hard (3+ steps): The solver must trace a CHAIN of 3+ events/facts where each step depends on the previous.
  The solver cannot shortcut by reading just the first and last sentences — they must follow the intermediate steps.
  Example: "What was the ultimate consequence after X?" where the chain is X→Y→Z→answer.
  The solver must: find X → discover Y → discover Z → find the answer. This is 3+ reasoning steps.

CRITICAL: A question asking "What was the consequence after X?" is Hard if the chain from X to the answer involves 3+ intermediate events that the solver must discover. Do NOT rate it Easy just because the answer appears in one sentence — the solver must TRACE the chain to find that sentence."""


def _build_difficulty_prompt_short(item):
    """Simplified difficulty prompt for retry."""
    question = item.get("generated_question", "")
    answer = item.get("gold_answer_phrase", "") or item.get("gold_answer_trigger", "")
    context = _fmt_supporting(item.get("supporting_sentences", []), max_sents=3)

    return f"""Context: {context}

Question: "{question}"
Answer: "{answer}"

Rate difficulty as JSON:
{{"predicted_difficulty":"Easy|Medium|Hard","required_steps":"1|2|3+","single_sentence_answerable":"yes|partial|no","answerable":"yes|partial|no","final_event_consistent":"yes|partial|no","reason":"brief"}}"""


def _build_path_dependency_prompt(item):
    """Build path-dependency judge prompt. Does NOT include target difficulty."""
    question = item.get("generated_question", "")
    answer = item.get("gold_answer_phrase", "") or item.get("gold_answer_trigger", "")
    context = _fmt_supporting(item.get("supporting_sentences", []), max_sents=None)
    events = _fmt_events_with_roles(item.get("events", []))
    relations = " -> ".join(item.get("relation_subtypes", [])) or "N/A"

    return f"""You are an expert evaluator for event-path question generation.

## Event path
{events}

## Relation sequence
{relations}

## Context sentences
{context}

## Question
"{question}"

## Expected answer
"{answer}"

## Task
Evaluate whether this question requires understanding the EVENT PATH (the chain of prior events leading to the final event). Consider:

1. Does the question IMPLICITLY require knowing about prior events?
   - Even if the question doesn't name prior events, does the solver need to trace
     the chain to understand what the question is asking about?
   - Example: "What formal resolution ended the conflict after X?" requires knowing
     that X led to Y led to Z led to the resolution — the solver must trace the chain.

2. Could someone answer this question by reading ONLY the sentence containing the final event?
   - If the question asks about the CONSEQUENCE of a chain, reading only the final
     sentence gives the answer text but doesn't tell the solver that THIS is the
     answer they're looking for. They need the chain to identify it.

3. How many prior events must be understood to answer correctly?

Reply as a single JSON object with exactly these fields:
{{
  "path_dependency": "none",
  "covered_prior_events": [],
  "num_required_prior_events": 0,
  "can_answer_without_path": "yes",
  "reason": "short explanation"
}}

Guidelines:
- path_dependency: "none", "partial", or "strong"
  * "strong": the solver MUST trace the event chain to find/identify the answer
  * "partial": the chain helps but isn't strictly necessary
  * "none": the answer is directly findable from one sentence with no chain needed
- covered_prior_events: list of prior event IDs that the question references or requires
- num_required_prior_events: how many prior events are needed to answer
- can_answer_without_path: "yes", "partial", or "no"
- reason: one sentence explanation"""


def _build_path_dependency_prompt_short(item):
    """Simplified path-dependency prompt for retry."""
    question = item.get("generated_question", "")
    answer = item.get("gold_answer_phrase", "") or item.get("gold_answer_trigger", "")
    events = _fmt_events_with_roles(item.get("events", []))

    return f"""Events: {events}
Question: "{question}"
Answer: "{answer}"

Does this question require prior events to answer? JSON:
{{"path_dependency":"none|partial|strong","covered_prior_events":[],"num_required_prior_events":0,"can_answer_without_path":"yes|partial|no","reason":"brief"}}"""


def independent_difficulty_judge(item, model_config):
    """Run independent difficulty judge. Does NOT see target difficulty.
    Returns dict with difficulty_judge fields merged into item.
    """
    expected_keys = {"predicted_difficulty", "required_steps", "single_sentence_answerable",
                     "answerable", "final_event_consistent", "reason"}

    prompt = _build_difficulty_prompt(item)
    result = {
        "difficulty_judge_prompt": prompt,
        "difficulty_judge_raw": "",
        "difficulty_judge": {},
        "difficulty_judge_status": "ok",
    }

    resp, err = _call_judge_api(prompt, model_config)
    if err:
        result["difficulty_judge_raw"] = err
        result["difficulty_judge_status"] = "api_error"
        result["difficulty_judge"] = {
            "predicted_difficulty": "Medium", "required_steps": "2",
            "single_sentence_answerable": "partial", "answerable": "partial",
            "final_event_consistent": "partial", "reason": "api_error"
        }
        return result

    result["difficulty_judge_raw"] = resp or ""
    parsed = _parse_judge_json(resp, expected_keys)

    if not parsed:
        short_prompt = _build_difficulty_prompt_short(item)
        result["difficulty_judge_prompt"] = short_prompt
        resp2, err2 = _call_judge_api(short_prompt, model_config)
        if err2:
            result["difficulty_judge_raw"] = resp2 or err2
            result["difficulty_judge_status"] = "api_error"
            result["difficulty_judge"] = {
                "predicted_difficulty": "Medium", "required_steps": "2",
                "single_sentence_answerable": "partial", "answerable": "partial",
                "final_event_consistent": "partial", "reason": "api_error_retry"
            }
            return result
        result["difficulty_judge_raw"] = resp2 or ""
        parsed = _parse_judge_json(resp2, expected_keys)

    if not parsed:
        result["difficulty_judge_status"] = "parse_error"
        result["difficulty_judge"] = {
            "predicted_difficulty": "Medium", "required_steps": "2",
            "single_sentence_answerable": "partial", "answerable": "partial",
            "final_event_consistent": "partial", "reason": "parse_error"
        }
        return result

    pred = parsed.get("predicted_difficulty", "Medium")
    if pred not in ("Easy", "Medium", "Hard"):
        pred = "Medium"
    steps = str(parsed.get("required_steps", "2"))
    if steps not in ("1", "2", "3+"):
        steps = "2"
    for key in ("single_sentence_answerable", "answerable", "final_event_consistent"):
        val = parsed.get(key, "partial")
        if val not in ("yes", "partial", "no"):
            parsed[key] = "partial"

    parsed["predicted_difficulty"] = pred
    parsed["required_steps"] = steps
    result["difficulty_judge"] = parsed
    return result


def independent_path_dependency_judge(item, model_config):
    """Run independent path-dependency judge. Returns dict with path_dependency fields merged into item."""
    expected_keys = {"path_dependency", "covered_prior_events", "num_required_prior_events",
                     "can_answer_without_path", "reason"}

    prompt = _build_path_dependency_prompt(item)
    result = {
        "path_dependency_judge_prompt": prompt,
        "path_dependency_judge_raw": "",
        "path_dependency_judge": {},
        "path_dependency_judge_status": "ok",
    }

    resp, err = _call_judge_api(prompt, model_config)
    if err:
        result["path_dependency_judge_raw"] = err
        result["path_dependency_judge_status"] = "api_error"
        result["path_dependency_judge"] = {
            "path_dependency": "partial", "covered_prior_events": [],
            "num_required_prior_events": 0, "can_answer_without_path": "partial",
            "reason": "api_error"
        }
        return result

    result["path_dependency_judge_raw"] = resp or ""
    parsed = _parse_judge_json(resp, expected_keys)

    if not parsed:
        short_prompt = _build_path_dependency_prompt_short(item)
        result["path_dependency_judge_prompt"] = short_prompt
        resp2, err2 = _call_judge_api(short_prompt, model_config)
        if err2:
            result["path_dependency_judge_raw"] = resp2 or err2
            result["path_dependency_judge_status"] = "api_error"
            result["path_dependency_judge"] = {
                "path_dependency": "partial", "covered_prior_events": [],
                "num_required_prior_events": 0, "can_answer_without_path": "partial",
                "reason": "api_error_retry"
            }
            return result
        result["path_dependency_judge_raw"] = resp2 or ""
        parsed = _parse_judge_json(resp2, expected_keys)

    if not parsed:
        result["path_dependency_judge_status"] = "parse_error"
        result["path_dependency_judge"] = {
            "path_dependency": "partial", "covered_prior_events": [],
            "num_required_prior_events": 0, "can_answer_without_path": "partial",
            "reason": "parse_error"
        }
        return result

    dep = parsed.get("path_dependency", "partial")
    if dep not in ("none", "partial", "strong"):
        dep = "partial"
    parsed["path_dependency"] = dep

    for key in ("can_answer_without_path",):
        val = parsed.get(key, "partial")
        if val not in ("yes", "partial", "no"):
            parsed[key] = "partial"

    if not isinstance(parsed.get("covered_prior_events"), list):
        parsed["covered_prior_events"] = []

    try:
        parsed["num_required_prior_events"] = int(parsed.get("num_required_prior_events", 0))
    except (ValueError, TypeError):
        parsed["num_required_prior_events"] = 0

    result["path_dependency_judge"] = parsed
    return result


# ═══════════════════════════════════════════════════════════════
# Blind Difficulty Judge
# Shows ONLY context + question + expected answer.
# Does NOT show: event path, relation sequence, target difficulty,
# method, strategy, hard_strategy, expected_steps.
# ═══════════════════════════════════════════════════════════════

def _build_blind_difficulty_prompt(item):
    """Build blind difficulty judge prompt. Context + question + answer ONLY."""
    question = item.get("generated_question", "")
    answer = item.get("gold_answer_phrase", "") or item.get("gold_answer_trigger", "")
    context = _fmt_supporting(item.get("supporting_sentences", []), max_sents=None)

    return f"""You are an expert difficulty evaluator for reading comprehension questions.

## Context
{context}

## Question
"{question}"

## Expected answer
"{answer}"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace through a chain of events/facts to reach the answer

The key question: HOW MANY sequential reasoning steps must the solver make to answer this question, given ONLY the context above?

Reply as a single JSON object:
{{
  "predicted_difficulty": "Easy",
  "required_steps": "1",
  "single_sentence_answerable": "yes",
  "answerable": "yes",
  "final_event_consistent": "yes",
  "reason": "short explanation"
}}

Guidelines:
- predicted_difficulty: Easy, Medium, or Hard
- required_steps: "1", "2", or "3+"
- single_sentence_answerable: can the answer be found in a single sentence? "yes", "partial", or "no"
- answerable: is the question answerable from the context? "yes", "partial", or "no"
- final_event_consistent: does the question ask for the expected answer? "yes", "partial", or "no"

Difficulty definitions — judge ONLY from what the question and context require:

- Easy (1 step): The solver reads ONE sentence and extracts the answer directly. No chain reasoning.
  Example: "What did the army do?" → read the sentence about the army.

- Medium (2 steps): The solver connects 2 pieces of information from 2 different sentences. Simple A→B link.
  Example: "What happened after X?" → find X in sentence 1, find result in sentence 2.

- Hard (3+ steps): The question and context REQUIRE the solver to trace a CHAIN of 3+ events/facts where each step depends on the previous. The solver cannot answer correctly without following the full intermediate chain."""


def _build_blind_difficulty_prompt_short(item):
    """Simplified blind difficulty prompt for retry."""
    question = item.get("generated_question", "")
    answer = item.get("gold_answer_phrase", "") or item.get("gold_answer_trigger", "")
    context = _fmt_supporting(item.get("supporting_sentences", []), max_sents=3)

    return f"""Context: {context}

Question: "{question}"
Answer: "{answer}"

Rate difficulty as JSON (final_event_consistent = does the question ask for the expected answer?):
{{"predicted_difficulty":"Easy|Medium|Hard","required_steps":"1|2|3+","single_sentence_answerable":"yes|partial|no","answerable":"yes|partial|no","final_event_consistent":"yes|partial|no","reason":"brief"}}"""


def blind_difficulty_judge(item, model_config):
    """Run blind difficulty judge. Shows ONLY context + question + answer.
    Does NOT see event path, target difficulty, strategy, or method.
    Returns dict with blind_ fields merged into item.
    """
    expected_keys = {"predicted_difficulty", "required_steps", "single_sentence_answerable",
                     "answerable", "final_event_consistent", "reason"}

    prompt = _build_blind_difficulty_prompt(item)
    # Audit: does the blind context contain the gold answer sentence?
    answer_text = (item.get("gold_answer_phrase", "") or item.get("gold_answer_trigger", "")).lower()
    supporting = item.get("supporting_sentences", [])
    answer_in_context = False
    if answer_text:
        for sent in supporting:
            sent_text = (sent[1] if isinstance(sent, (list, tuple)) and len(sent) >= 2 else str(sent)).lower()
            if answer_text in sent_text:
                answer_in_context = True
                break
    result = {
        "blind_difficulty_judge_prompt": prompt,
        "blind_difficulty_judge_raw": "",
        "blind_difficulty_judge": {},
        "blind_difficulty_judge_status": "ok",
        "blind_context_contains_answer_sentence": answer_in_context,
    }

    resp, err = _call_judge_api(prompt, model_config)
    if err:
        result["blind_difficulty_judge_raw"] = err
        result["blind_difficulty_judge_status"] = "api_error"
        result["blind_difficulty_judge"] = {
            "predicted_difficulty": "Medium", "required_steps": "2",
            "single_sentence_answerable": "partial", "answerable": "partial",
            "final_event_consistent": "partial", "reason": "api_error"
        }
        return result

    result["blind_difficulty_judge_raw"] = resp or ""
    parsed = _parse_judge_json(resp, expected_keys)

    if not parsed:
        short_prompt = _build_blind_difficulty_prompt_short(item)
        result["blind_difficulty_judge_prompt"] = short_prompt
        resp2, err2 = _call_judge_api(short_prompt, model_config)
        if err2:
            result["blind_difficulty_judge_raw"] = resp2 or err2
            result["blind_difficulty_judge_status"] = "api_error"
            result["blind_difficulty_judge"] = {
                "predicted_difficulty": "Medium", "required_steps": "2",
                "single_sentence_answerable": "partial", "answerable": "partial",
                "final_event_consistent": "partial", "reason": "api_error_retry"
            }
            return result
        result["blind_difficulty_judge_raw"] = resp2 or ""
        parsed = _parse_judge_json(resp2, expected_keys)

    if not parsed:
        result["blind_difficulty_judge_status"] = "parse_error"
        result["blind_difficulty_judge"] = {
            "predicted_difficulty": "Medium", "required_steps": "2",
            "single_sentence_answerable": "partial", "answerable": "partial",
            "final_event_consistent": "partial", "reason": "parse_error"
        }
        return result

    pred = parsed.get("predicted_difficulty", "Medium")
    if pred not in ("Easy", "Medium", "Hard"):
        pred = "Medium"
    steps = str(parsed.get("required_steps", "2"))
    if steps not in ("1", "2", "3+"):
        steps = "2"
    for key in ("single_sentence_answerable", "answerable", "final_event_consistent"):
        val = parsed.get(key, "partial")
        if val not in ("yes", "partial", "no"):
            parsed[key] = "partial"

    parsed["predicted_difficulty"] = pred
    parsed["required_steps"] = steps
    result["blind_difficulty_judge"] = parsed
    return result


# ═══════════════════════════════════════════════════════════════
# Hard Answer-Alignment Judge
# Evaluates whether a Hard question actually asks for the expected answer.
# ═══════════════════════════════════════════════════════════════

def _build_hard_alignment_prompt(item):
    """Build hard answer-alignment judge prompt."""
    question = item.get("generated_question", "")
    answer = item.get("gold_answer_phrase", "") or item.get("gold_answer_trigger", "")
    context = _fmt_supporting(item.get("supporting_sentences", []), max_sents=None)

    return f"""You are an expert evaluator for reading comprehension questions.

## Context
{context}

## Question
"{question}"

## Expected answer
"{answer}"

## Task
Evaluate whether this question is well-aligned with its expected answer.

1. asks_expected_answer: Does the question ask for information that the expected answer provides?
   - "yes": the question clearly asks for what the answer gives
   - "partial": the question is somewhat aligned but indirect or vague
   - "no": the question asks for something fundamentally different from the answer

2. expected_answer_natural: Would a reader, seeing only the question, naturally expect an answer like "{answer}"?
   - "yes": the answer type matches what the question asks for
   - "partial": the answer could fit but isn't the most natural type
   - "no": the answer type doesn't match (e.g., question asks "why" but answer is a date)

3. target_drift: Has the question drifted away from the target answer to ask about something else?
   - "yes": the question asks about a different event/concept than the answer addresses
   - "no": the question stays on target

Reply as a single JSON object:
{{"asks_expected_answer":"yes","expected_answer_natural":"yes","target_drift":"no","reason":"brief explanation"}}"""


def hard_answer_alignment_judge(item, model_config):
    """Run hard answer-alignment judge.
    Returns dict with hard_alignment fields merged into item.
    """
    expected_keys = {"asks_expected_answer", "expected_answer_natural", "target_drift", "reason"}

    prompt = _build_hard_alignment_prompt(item)
    result = {
        "hard_alignment_prompt": prompt,
        "hard_alignment_raw": "",
        "hard_alignment": {},
        "hard_alignment_status": "ok",
    }

    resp, err = _call_judge_api(prompt, model_config)
    if err:
        result["hard_alignment_raw"] = err
        result["hard_alignment_status"] = "api_error"
        result["hard_alignment"] = {
            "asks_expected_answer": "partial", "expected_answer_natural": "partial",
            "target_drift": "no", "reason": "api_error"
        }
        return result

    result["hard_alignment_raw"] = resp or ""
    parsed = _parse_judge_json(resp, expected_keys)

    if not parsed:
        short_prompt = f"""Question: "{item.get('generated_question', '')}"
Answer: "{item.get('gold_answer_phrase', '') or item.get('gold_answer_trigger', '')}"

Does this question ask for this answer? JSON:
{{"asks_expected_answer":"yes|partial|no","expected_answer_natural":"yes|partial|no","target_drift":"yes|no","reason":"brief"}}"""
        result["hard_alignment_prompt"] = short_prompt
        resp2, err2 = _call_judge_api(short_prompt, model_config)
        if err2:
            result["hard_alignment_raw"] = resp2 or err2
            result["hard_alignment_status"] = "api_error"
            result["hard_alignment"] = {
                "asks_expected_answer": "partial", "expected_answer_natural": "partial",
                "target_drift": "no", "reason": "api_error_retry"
            }
            return result
        result["hard_alignment_raw"] = resp2 or ""
        parsed = _parse_judge_json(resp2, expected_keys)

    if not parsed:
        result["hard_alignment_status"] = "parse_error"
        result["hard_alignment"] = {
            "asks_expected_answer": "partial", "expected_answer_natural": "partial",
            "target_drift": "no", "reason": "parse_error"
        }
        return result

    for key in ("asks_expected_answer", "expected_answer_natural"):
        val = parsed.get(key, "partial")
        if val not in ("yes", "partial", "no"):
            parsed[key] = "partial"

    td = parsed.get("target_drift", "no")
    if td not in ("yes", "no"):
        parsed["target_drift"] = "no"

    result["hard_alignment"] = parsed
    return result
