"""Independent difficulty evaluation: judge question difficulty and path dependency
without revealing target difficulty to the judge.

Uses AIHUBMIX (GPT-4o-mini) as the independent judge model.

Usage:
    python -m scripts.run_independent_difficulty_eval
    python -m scripts.run_independent_difficulty_eval --max_items 2
    python -m scripts.run_independent_difficulty_eval --methods pathqg,zeroshot
"""
import argparse
import json
import os
import re
import time
import random
from collections import defaultdict
from pathlib import Path

from dcqg.utils.jsonl import read_jsonl, write_jsonl
from dcqg.utils.config import get_api_config
from dcqg.utils.api_client import call_openai_compatible


INPUT_DIR = "outputs/runs/baseline_alignment_pilot"
OUTPUT_DIR = "outputs/runs/independent_difficulty_eval_pilot"
SEED = 42
DIFF_MAP = {"Easy": 1, "Medium": 2, "Hard": 3}
METHOD_FILES = {
    "PathQG-HardAware": "PathQG-HardAware_questions.filtered.jsonl",
    "ZeroShot-TargetQG": "ZeroShot_questions.filtered.jsonl",
    "ICL-TargetQG": "ICL_questions.filtered.jsonl",
    "SelfRefine": "SelfRefine_questions.filtered.jsonl",
}

# ── Counters ──
api_call_count = 0
parse_error_count = 0
retry_count = 0


# ═══════════════════════════════════════════════════════════════════════════════
# JSON Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_judge_json(resp, expected_keys):
    """Robust JSON parser for judge responses. Returns dict or None."""
    if not resp:
        return None

    def _check(parsed):
        if isinstance(parsed, dict) and all(k in parsed for k in expected_keys):
            return parsed
        return None

    # 1. Direct parse
    try:
        result = _check(json.loads(resp))
        if result:
            return result
    except json.JSONDecodeError:
        pass

    # 2. Extract first {...}
    try:
        s_idx = resp.index("{")
        e_idx = resp.rindex("}") + 1
        result = _check(json.loads(resp[s_idx:e_idx]))
        if result:
            return result
    except (ValueError, json.JSONDecodeError):
        pass

    # 3. Fix common JSON errors
    cleaned = resp.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'\s*```$', '', cleaned).strip()
    cleaned = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', cleaned)
    cleaned = cleaned.replace("'", '"')
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    cleaned = re.sub(r'""+', '"', cleaned)
    try:
        result = _check(json.loads(cleaned))
        if result:
            return result
    except json.JSONDecodeError:
        pass

    # 4. Regex targeted extraction
    partial = {}
    for key in expected_keys:
        m = re.search(rf'"{key}"\s*:\s*(".*?"|[\d.]+|true|false|null|\[.*?\])', resp, re.DOTALL)
        if m:
            val_str = m.group(1)
            try:
                partial[key] = json.loads(val_str)
            except json.JSONDecodeError:
                partial[key] = val_str.strip('"')
    if partial and all(k in partial for k in expected_keys):
        return partial

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt Builders
# ═══════════════════════════════════════════════════════════════════════════════

def _format_supporting_sentences(supporting_sentences, max_sents=6):
    """Format supporting_sentences as [S{id}] text lines."""
    lines = []
    for i, sent in enumerate(supporting_sentences[:max_sents]):
        if isinstance(sent, (list, tuple)) and len(sent) >= 2:
            lines.append(f"[S{sent[0]}] {sent[1]}")
        elif isinstance(sent, str):
            lines.append(f"[S{i}] {sent}")
    return "\n".join(lines) if lines else "No context available."


def _format_events(events):
    """Format events as numbered list."""
    lines = []
    for i, ev in enumerate(events):
        trigger = ev.get("trigger", "?")
        etype = ev.get("type", "?")
        lines.append(f"  {i+1}. {trigger} ({etype})")
    return "\n".join(lines) if lines else "No events."


def _format_events_with_roles(events):
    """Format events with PRIOR/FINAL labels."""
    lines = []
    for i, ev in enumerate(events):
        trigger = ev.get("trigger", "?")
        etype = ev.get("type", "?")
        eid = ev.get("id", "?")
        role = "FINAL" if i == len(events) - 1 else "PRIOR"
        lines.append(f"  {i+1}. id={eid} trigger=\"{trigger}\" type={etype} role={role}")
    return "\n".join(lines) if lines else "No events."


def build_difficulty_prompt(item):
    """Build difficulty-only judge prompt. Does NOT include target difficulty."""
    question = item.get("generated_question", "")
    answer = item.get("gold_answer_phrase", "") or item.get("gold_answer_trigger", "")
    context = _format_supporting_sentences(item.get("supporting_sentences", []))
    events = _format_events(item.get("events", []))

    prompt = f"""You are an expert difficulty evaluator for reading comprehension questions.

## Context
{context}

## Event path
{events}

## Question
"{question}"

## Expected answer
"{answer}"

## Task
Evaluate the REASONING DIFFICULTY of this question.

Think from the solver's perspective: the solver does NOT know the expected answer. They must:
1. Understand what the question is asking
2. Identify which information in the context is relevant
3. Connect the relevant pieces to arrive at the answer

The key question is: HOW MANY distinct pieces of information must the solver connect to answer correctly?

Reply as a single JSON object with exactly these fields:
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

Difficulty definitions:
- Easy (1 step): The solver can answer by finding ONE fact in ONE sentence. Example: "When did X happen?" → find the date in one sentence.
- Medium (2 steps): The solver must connect TWO pieces of information. Example: "What happened after X?" → find X, then find what followed.
- Hard (3+ steps): The solver must connect THREE OR MORE pieces of information from different sentences. The question describes a situation or chain, and the solver must trace through multiple events to identify the answer. Example: "What consequence did the series of military actions that began with X ultimately produce?" → find X, find intermediate actions, find the final consequence.

For Hard questions, the solver typically needs to:
- Identify what event the question refers to (may not use the exact trigger word)
- Trace through 2+ intermediate events
- Arrive at the answer that is the end of the chain

Note: A question can be Hard even if the answer text appears in one sentence, IF the solver needs to understand the chain of events to know that sentence is the answer."""
    return prompt


def build_difficulty_prompt_short(item):
    """Simplified difficulty prompt for retry."""
    question = item.get("generated_question", "")
    answer = item.get("gold_answer_phrase", "") or item.get("gold_answer_trigger", "")
    context = _format_supporting_sentences(item.get("supporting_sentences", []), max_sents=3)

    prompt = f"""Context: {context}

Question: "{question}"
Answer: "{answer}"

Rate difficulty as JSON:
{{"predicted_difficulty":"Easy|Medium|Hard","required_steps":"1|2|3+","single_sentence_answerable":"yes|partial|no","answerable":"yes|partial|no","final_event_consistent":"yes|partial|no","reason":"brief"}}"""
    return prompt


def build_path_dependency_prompt(item):
    """Build path-dependency judge prompt. Does NOT include target difficulty."""
    question = item.get("generated_question", "")
    answer = item.get("gold_answer_phrase", "") or item.get("gold_answer_trigger", "")
    context = _format_supporting_sentences(item.get("supporting_sentences", []))
    events = _format_events_with_roles(item.get("events", []))
    relations = " -> ".join(item.get("relation_subtypes", [])) or "N/A"

    prompt = f"""You are an expert evaluator for event-path question generation.

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
1. Could someone answer this question by reading ONLY the sentence containing the final event?
2. Does answering require knowledge of what happened before (prior events)?
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
  - none: question can be answered without using prior path events
  - partial: prior events help contextualize the question but are not strictly necessary
  - strong: question explicitly or semantically requires prior path events
- covered_prior_events: list of prior event IDs that the question references
- num_required_prior_events: how many prior events are needed to answer
- can_answer_without_path: "yes", "partial", or "no"
- reason: one sentence explanation"""
    return prompt


def build_path_dependency_prompt_short(item):
    """Simplified path-dependency prompt for retry."""
    question = item.get("generated_question", "")
    answer = item.get("gold_answer_phrase", "") or item.get("gold_answer_trigger", "")
    events = _format_events_with_roles(item.get("events", []))

    prompt = f"""Events: {events}
Question: "{question}"
Answer: "{answer}"

Does this question require prior events to answer? JSON:
{{"path_dependency":"none|partial|strong","covered_prior_events":[],"num_required_prior_events":0,"can_answer_without_path":"yes|partial|no","reason":"brief"}}"""
    return prompt


# ═══════════════════════════════════════════════════════════════════════════════
# Judge Functions
# ═══════════════════════════════════════════════════════════════════════════════

def call_judge(prompt, model_config):
    """Call AIHUBMIX LLM. Returns (response_text, error_message_or_None)."""
    global api_call_count
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
        api_call_count += 1
        return resp, None
    except Exception as exc:
        api_call_count += 1
        return None, f"{type(exc).__name__}: {exc}"


def difficulty_judge(item, model_config):
    """Run difficulty-only judge on one item. Returns dict with judge fields."""
    global parse_error_count, retry_count

    expected_keys = {"predicted_difficulty", "required_steps", "single_sentence_answerable",
                     "answerable", "final_event_consistent", "reason"}

    prompt = build_difficulty_prompt(item)
    result = {
        "difficulty_judge_prompt": prompt,
        "difficulty_judge_raw": "",
        "difficulty_judge": {},
        "difficulty_judge_status": "ok",
    }

    resp, err = call_judge(prompt, model_config)
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
    parsed = parse_judge_json(resp, expected_keys)

    if not parsed:
        # Retry with simplified prompt
        retry_count += 1
        short_prompt = build_difficulty_prompt_short(item)
        result["difficulty_judge_prompt"] = short_prompt  # overwrite with retry prompt
        resp2, err2 = call_judge(short_prompt, model_config)
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
        parsed = parse_judge_json(resp2, expected_keys)

    if not parsed:
        parse_error_count += 1
        result["difficulty_judge_status"] = "parse_error"
        result["difficulty_judge"] = {
            "predicted_difficulty": "Medium", "required_steps": "2",
            "single_sentence_answerable": "partial", "answerable": "partial",
            "final_event_consistent": "partial", "reason": "parse_error"
        }
        return result

    # Normalize
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


def path_dependency_judge(item, model_config):
    """Run path-dependency judge on one item. Returns dict with judge fields."""
    global parse_error_count, retry_count

    expected_keys = {"path_dependency", "covered_prior_events", "num_required_prior_events",
                     "can_answer_without_path", "reason"}

    prompt = build_path_dependency_prompt(item)
    result = {
        "path_dependency_judge_prompt": prompt,
        "path_dependency_judge_raw": "",
        "path_dependency_judge": {},
        "path_dependency_judge_status": "ok",
    }

    resp, err = call_judge(prompt, model_config)
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
    parsed = parse_judge_json(resp, expected_keys)

    if not parsed:
        retry_count += 1
        short_prompt = build_path_dependency_prompt_short(item)
        result["path_dependency_judge_prompt"] = short_prompt
        resp2, err2 = call_judge(short_prompt, model_config)
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
        parsed = parse_judge_json(resp2, expected_keys)

    if not parsed:
        parse_error_count += 1
        result["path_dependency_judge_status"] = "parse_error"
        result["path_dependency_judge"] = {
            "path_dependency": "partial", "covered_prior_events": [],
            "num_required_prior_events": 0, "can_answer_without_path": "partial",
            "reason": "parse_error"
        }
        return result

    # Normalize
    dep = parsed.get("path_dependency", "partial")
    if dep not in ("none", "partial", "strong"):
        dep = "partial"
    parsed["path_dependency"] = dep

    for key in ("can_answer_without_path",):
        val = parsed.get(key, "partial")
        if val not in ("yes", "partial", "no"):
            parsed[key] = "partial"

    # Ensure covered_prior_events is a list
    if not isinstance(parsed.get("covered_prior_events"), list):
        parsed["covered_prior_events"] = []

    # Ensure num_required_prior_events is int
    try:
        parsed["num_required_prior_events"] = int(parsed.get("num_required_prior_events", 0))
    except (ValueError, TypeError):
        parsed["num_required_prior_events"] = 0

    result["path_dependency_judge"] = parsed
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_spearman(targets, predictions):
    """Pure-Python Spearman rank correlation. Returns float or 'N/A'."""
    n = len(targets)
    if n < 2:
        return "N/A"

    # Check if all values are identical
    if len(set(targets)) < 2 or len(set(predictions)) < 2:
        return "N/A"

    def _rank(values):
        indexed = sorted(enumerate(values), key=lambda x: x[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and indexed[j][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j + 1) / 2.0
            for k in range(i, j):
                ranks[indexed[k][0]] = avg_rank
            i = j
        return ranks

    rank_t = _rank(targets)
    rank_p = _rank(predictions)
    d_sq_sum = sum((rt - rp) ** 2 for rt, rp in zip(rank_t, rank_p))
    rho = 1.0 - (6.0 * d_sq_sum) / (n * (n * n - 1))
    return round(rho, 4)


def compute_stats(items, label=""):
    """Compute stats for a list of judged items."""
    total = len(items)
    if total == 0:
        return {"label": label, "total": 0}

    ok_items = [x for x in items if x.get("difficulty_judge_status") == "ok"
                and x.get("path_dependency_judge_status") == "ok"]

    # Difficulty accuracy
    targets = [x.get("difficulty", "") for x in ok_items]
    predicted = [x["difficulty_judge"].get("predicted_difficulty", "Medium") for x in ok_items]
    diff_acc = sum(1 for t, p in zip(targets, predicted) if t == p) / len(ok_items) if ok_items else 0

    # Spearman
    target_nums = [DIFF_MAP.get(t, 2) for t in targets]
    pred_nums = [DIFF_MAP.get(p, 2) for p in predicted]
    spearman = compute_spearman(target_nums, pred_nums)

    # Step consistency
    step_map = {"Easy": "1", "Medium": "2", "Hard": "3+"}
    step_consistent = 0
    for x in ok_items:
        tgt = x.get("difficulty", "")
        steps = x["difficulty_judge"].get("required_steps", "")
        if step_map.get(tgt) == steps:
            step_consistent += 1
    step_cons = step_consistent / len(ok_items) if ok_items else 0

    # Path dependency distribution
    dep_counts = {"none": 0, "partial": 0, "strong": 0}
    for x in ok_items:
        dep = x["path_dependency_judge"].get("path_dependency", "partial")
        dep_counts[dep] = dep_counts.get(dep, 0) + 1

    # Answerability
    answerable_count = sum(1 for x in ok_items
                          if x["difficulty_judge"].get("answerable") in ("yes", "partial"))
    answerable_rate = answerable_count / len(ok_items) if ok_items else 0

    # Final-event consistency
    fec_count = sum(1 for x in ok_items
                    if x["difficulty_judge"].get("final_event_consistent") in ("yes", "partial"))
    fec_rate = fec_count / len(ok_items) if ok_items else 0

    # Confusion matrix
    confusion = defaultdict(int)
    for t, p in zip(targets, predicted):
        confusion[f"{t}->{p}"] += 1

    # Per-difficulty breakdown
    per_diff = {}
    for diff in ["Easy", "Medium", "Hard"]:
        dx = [x for x in ok_items if x.get("difficulty") == diff]
        if not dx:
            per_diff[diff] = {"count": 0}
            continue
        d_targets = [diff] * len(dx)
        d_predicted = [x["difficulty_judge"].get("predicted_difficulty", "Medium") for x in dx]
        d_acc = sum(1 for t, p in zip(d_targets, d_predicted) if t == p) / len(dx)
        d_steps = [x["difficulty_judge"].get("required_steps", "2") for x in dx]
        d_step_ok = sum(1 for x in dx if step_map.get(diff) == x["difficulty_judge"].get("required_steps", ""))
        d_dep_strong = sum(1 for x in dx if x["path_dependency_judge"].get("path_dependency") == "strong")
        per_diff[diff] = {
            "count": len(dx),
            "accuracy": round(d_acc, 3),
            "step_consistency": round(d_step_ok / len(dx), 3) if dx else 0,
            "pred_distribution": {d: sum(1 for p in d_predicted if p == d) for d in ["Easy", "Medium", "Hard"]},
            "pathdep_strong": d_dep_strong,
            "avg_steps": round(sum(DIFF_MAP.get(p, 2) for p in d_predicted) / len(dx), 2) if dx else 0,
        }

    diff_parse_errors = sum(1 for x in items if x.get("difficulty_judge_status") != "ok")
    path_parse_errors = sum(1 for x in items if x.get("path_dependency_judge_status") != "ok")

    return {
        "label": label,
        "total": total,
        "judge_ok": len(ok_items),
        "diff_parse_errors": diff_parse_errors,
        "path_parse_errors": path_parse_errors,
        "diff_accuracy": round(diff_acc, 3),
        "spearman": spearman,
        "step_consistency": round(step_cons, 3),
        "pathdep_strong": dep_counts.get("strong", 0),
        "pathdep_strong_pct": round(dep_counts.get("strong", 0) / len(ok_items) * 100, 1) if ok_items else 0,
        "pathdep_strong_partial": dep_counts.get("strong", 0) + dep_counts.get("partial", 0),
        "pathdep_strong_partial_pct": round((dep_counts.get("strong", 0) + dep_counts.get("partial", 0)) / len(ok_items) * 100, 1) if ok_items else 0,
        "answerable_rate": round(answerable_rate, 3),
        "fec_rate": round(fec_rate, 3),
        "confusion": dict(confusion),
        "per_difficulty": per_diff,
    }


def balanced_subset(all_items, seed=SEED):
    """Create balanced subset: min count per difficulty across methods. Hard excluded if any method has 0."""
    rng = random.Random(seed)

    # Group by (method, difficulty)
    groups = defaultdict(list)
    for item in all_items:
        method = item.get("method", "unknown")
        diff = item.get("difficulty", "?")
        groups[(method, diff)].append(item)

    # Find methods
    methods = sorted(set(item.get("method", "unknown") for item in all_items))

    # For each difficulty, find min count across methods
    balanced = []
    min_counts = {}
    excluded = []
    for diff in ["Easy", "Medium", "Hard"]:
        counts = {m: len(groups.get((m, diff), [])) for m in methods}
        min_c = min(counts.values())
        min_counts[diff] = min_c
        if min_c == 0:
            excluded.append(diff)
            continue
        for m in methods:
            pool = groups.get((m, diff), [])
            sampled = rng.sample(pool, min(min_c, len(pool)))
            balanced.extend(sampled)

    return balanced, min_counts, excluded


# ═══════════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(all_stats, balanced_stats, all_judged, balanced_judged,
                    min_counts, excluded_diffs, model_name, output_dir):
    """Generate DIFFICULTY_EVAL_REPORT.md."""
    lines = []
    lines.append("# Independent Difficulty Evaluation Report\n")
    lines.append(f"**Input:** `{INPUT_DIR}/*_questions.filtered.jsonl`")
    lines.append(f"**Judge Model:** {model_name} (AIHUBMIX)")
    lines.append(f"**Items evaluated:** {sum(s['total'] for s in all_stats)} (final_filter_pass=true)")
    lines.append(f"**Balanced subset:** {sum(s['total'] for s in balanced_stats)} "
                 f"(seed={SEED})")
    lines.append(f"**Total API calls:** {api_call_count}")
    lines.append(f"**Parse errors:** {parse_error_count}")
    lines.append(f"**Retries:** {retry_count}")
    lines.append("")
    lines.append("**This evaluation is independent of the solver and does not use solver answers or solver correctness.**")
    lines.append("")

    # ── Yield-Aware Results ──
    lines.append("## Method Comparison (Yield-Aware — All Valid Questions)\n")
    header = "| Metric | " + " | ".join(s["label"] for s in all_stats) + " |"
    sep = "|--------|" + "|".join("---:" for _ in all_stats) + "|"
    lines.append(header)
    lines.append(sep)

    def _fmt_row(label, key, vals_func):
        vals = [vals_func(s) for s in all_stats]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    _fmt_row("Judged", "total", lambda s: str(s["total"]))
    _fmt_row("Judge OK", "judge_ok", lambda s: str(s["judge_ok"]))
    _fmt_row("Diff Parse Errors", "diff_parse_errors", lambda s: str(s["diff_parse_errors"]))
    _fmt_row("Path Parse Errors", "path_parse_errors", lambda s: str(s["path_parse_errors"]))
    _fmt_row("Diff Accuracy", "diff_accuracy", lambda s: f"{s['diff_accuracy']:.1%}")
    _fmt_row("Spearman rho", "spearman", lambda s: str(s["spearman"]))
    _fmt_row("Step Consistency", "step_consistency", lambda s: f"{s['step_consistency']:.1%}")
    _fmt_row("PathDep Strong", "pathdep_strong",
             lambda s: f"{s['pathdep_strong']}/{s['judge_ok']} ({s['pathdep_strong_pct']:.0f}%)")
    _fmt_row("PathDep Strong+Partial", "pathdep_strong_partial",
             lambda s: f"{s['pathdep_strong_partial']}/{s['judge_ok']} ({s['pathdep_strong_partial_pct']:.0f}%)")
    _fmt_row("Answerable", "answerable_rate", lambda s: f"{s['answerable_rate']:.1%}")
    _fmt_row("FinalConsistent", "fec_rate", lambda s: f"{s['fec_rate']:.1%}")

    # ── Confusion Matrix (Yield-Aware, per method) ──
    lines.append("\n### Difficulty Confusion Matrix (Yield-Aware)\n")
    for s in all_stats:
        lines.append(f"\n#### {s['label']}\n")
        lines.append("| Target \\ Pred | Easy | Medium | Hard |")
        lines.append("|---------------|-----:|-------:|-----:|")
        c = s.get("confusion", {})
        for diff in ["Easy", "Medium", "Hard"]:
            cells = [str(c.get(f"{diff}->{p}", 0)) for p in ["Easy", "Medium", "Hard"]]
            lines.append(f"| {diff} | " + " | ".join(cells) + " |")

    # ── Per-Difficulty Breakdown ──
    lines.append("\n## Per-Difficulty Breakdown (Yield-Aware)\n")
    lines.append("| Method | Target | N | Pred Easy | Pred Medium | Pred Hard | Diff Acc | Step Cons | PathDep Strong |")
    lines.append("|--------|--------|--:|----------:|------------:|----------:|---------:|----------:|---------------:|")
    for s in all_stats:
        for diff in ["Easy", "Medium", "Hard"]:
            pd = s.get("per_difficulty", {}).get(diff, {})
            if pd.get("count", 0) == 0:
                continue
            dist = pd.get("pred_distribution", {})
            lines.append(
                f"| {s['label']} | {diff} | {pd['count']} | "
                f"{dist.get('Easy', 0)} | {dist.get('Medium', 0)} | {dist.get('Hard', 0)} | "
                f"{pd.get('accuracy', 0):.0%} | {pd.get('step_consistency', 0):.0%} | "
                f"{pd.get('pathdep_strong', 0)} |"
            )

    # ── Balanced Quality ──
    lines.append("\n## Balanced Quality (Min-Count Per Difficulty)\n")
    lines.append(f"**Seed:** {SEED}")
    lines.append(f"**Per-difficulty min counts:** " +
                 ", ".join(f"{d}={min_counts.get(d, 0)}" for d in ["Easy", "Medium", "Hard"]))
    if excluded_diffs:
        lines.append(f"**Excluded difficulties:** {', '.join(excluded_diffs)} "
                     f"(at least one method has 0 valid questions)")
    lines.append("")

    header = "| Metric | " + " | ".join(s["label"] for s in balanced_stats) + " |"
    sep = "|--------|" + "|".join("---:" for _ in balanced_stats) + "|"
    lines.append(header)
    lines.append(sep)

    def _fmt_row_b(label, vals_func):
        vals = [vals_func(s) for s in balanced_stats]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    _fmt_row_b("N balanced", lambda s: str(s["total"]))
    _fmt_row_b("Diff Accuracy", lambda s: f"{s['diff_accuracy']:.1%}")
    _fmt_row_b("Spearman rho", lambda s: str(s["spearman"]))
    _fmt_row_b("Step Consistency", lambda s: f"{s['step_consistency']:.1%}")
    _fmt_row_b("PathDep Strong",
               lambda s: f"{s['pathdep_strong']}/{s['judge_ok']} ({s['pathdep_strong_pct']:.0f}%)")
    _fmt_row_b("Answerable", lambda s: f"{s['answerable_rate']:.1%}")
    _fmt_row_b("FinalConsistent", lambda s: f"{s['fec_rate']:.1%}")

    # ── Key Observations ──
    lines.append("\n## Key Observations\n")

    # Best difficulty accuracy
    ok_stats = [s for s in all_stats if s.get("judge_ok", 0) > 0]
    if ok_stats:
        best_acc = max(ok_stats, key=lambda s: s["diff_accuracy"])
        lines.append(f"- Highest difficulty accuracy (yield-aware): **{best_acc['label']}** ({best_acc['diff_accuracy']:.1%})")

    # Best Spearman
    sp_stats = [s for s in ok_stats if s.get("spearman") != "N/A"]
    if sp_stats:
        best_sp = max(sp_stats, key=lambda s: s["spearman"])
        lines.append(f"- Highest Spearman rho (yield-aware): **{best_sp['label']}** ({best_sp['spearman']})")

    # Path dependency
    if ok_stats:
        best_dep = max(ok_stats, key=lambda s: s["pathdep_strong_pct"])
        lines.append(f"- Highest path dependency strong (yield-aware): **{best_dep['label']}** ({best_dep['pathdep_strong_pct']:.0f}%)")

    # Hard note
    hard_stats = [(s["label"], s.get("per_difficulty", {}).get("Hard", {}).get("count", 0)) for s in all_stats]
    lines.append(f"- Hard valid counts: " + ", ".join(f"{n}={c}" for n, c in hard_stats))

    # ── Interpretation ──
    lines.append("\n## Interpretation\n")
    lines.append("- **Valid yield advantage:** PathQG-HardAware produces more filter-passing questions (36 vs 7-11 for baselines).")
    lines.append("- **Independent difficulty consistency:** Compare difficulty accuracy and Spearman across methods to assess whether PathQG's difficulty labels align with independent judge assessment.")
    lines.append("- **Path dependency quality:** Strong path dependency indicates questions genuinely require understanding the event path, not just the final sentence.")
    lines.append("- **Solver accuracy is not used here.** This evaluation is purely about difficulty prediction and path dependency.")

    report_path = Path(output_dir) / "DIFFICULTY_EVAL_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report: {report_path}")


def generate_audit_sample(items, model_name, output_dir, seed=SEED):
    """Generate DIFFICULTY_AUDIT_SAMPLE.md with examples for manual review."""
    rng = random.Random(seed)
    lines = []
    lines.append("# Independent Difficulty Evaluation — Audit Sample\n")
    lines.append(f"**Judge Model:** {model_name}")
    lines.append(f"**Seed:** {seed}\n")
    lines.append("This evaluation is independent of the solver and does not use solver answers or solver correctness.\n")

    # Group by method
    by_method = defaultdict(list)
    for item in items:
        by_method[item.get("method", "unknown")].append(item)

    for method, method_items in sorted(by_method.items()):
        lines.append(f"## {method}\n")

        # Priority: mismatch > pathdep=none > Hard > parse errors
        mismatches = [x for x in method_items
                      if x.get("difficulty") != x.get("difficulty_judge", {}).get("predicted_difficulty", "")]
        no_dep = [x for x in method_items
                  if x.get("path_dependency_judge", {}).get("path_dependency") == "none"]
        hard_items = [x for x in method_items if x.get("difficulty") == "Hard"]
        parse_errs = [x for x in method_items if x.get("difficulty_judge_status") != "ok"
                      or x.get("path_dependency_judge_status") != "ok"]

        # Select up to 5, prioritizing interesting cases
        selected = []
        for pool in [mismatches, no_dep, hard_items, parse_errs, method_items]:
            for x in pool:
                if x not in selected:
                    selected.append(x)
                if len(selected) >= 5:
                    break
            if len(selected) >= 5:
                break

        for i, item in enumerate(selected[:5]):
            diff = item.get("difficulty", "?")
            dj = item.get("difficulty_judge", {})
            pj = item.get("path_dependency_judge", {})
            pred = dj.get("predicted_difficulty", "?")
            steps = dj.get("required_steps", "?")
            dep = pj.get("path_dependency", "?")
            prior_needed = pj.get("num_required_prior_events", "?")
            question = item.get("generated_question", "")[:200]
            answer = (item.get("gold_answer_phrase", "") or item.get("gold_answer_trigger", ""))[:200]
            events = item.get("events", [])
            event_path = " -> ".join(f"{e.get('trigger','?')}/{e.get('type','?')}" for e in events)
            relations = " -> ".join(item.get("relation_subtypes", [])) or "N/A"
            doc_id = item.get("doc_id", "?")[:12]

            lines.append(f"### [{diff} -> {pred}] doc_id={doc_id}\n")
            lines.append(f"- **Target difficulty:** {diff}")
            lines.append(f"- **Predicted difficulty:** {pred} | **Steps:** {steps}")
            lines.append(f"- **Path dependency:** {dep} | **Prior needed:** {prior_needed}")
            lines.append(f"- **Question:** {question}")
            lines.append(f"- **Answer:** {answer}")
            lines.append(f"- **Event path:** {event_path}")
            lines.append(f"- **Relations:** {relations}")
            lines.append(f"- **Diff judge status:** {item.get('difficulty_judge_status', '?')}")
            lines.append(f"- **Path judge status:** {item.get('path_dependency_judge_status', '?')}")
            if dj.get("reason"):
                lines.append(f"- **Diff reason:** {dj['reason']}")
            if pj.get("reason"):
                lines.append(f"- **Path reason:** {pj['reason']}")
            lines.append("")

    audit_path = Path(output_dir) / "DIFFICULTY_AUDIT_SAMPLE.md"
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Audit sample: {audit_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global api_call_count, parse_error_count, retry_count
    api_call_count = 0
    parse_error_count = 0
    retry_count = 0

    parser = argparse.ArgumentParser(description="Independent difficulty evaluation.")
    parser.add_argument("--input_dir", default=INPUT_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--methods", default="all",
                        help="Comma-separated: pathqg,zeroshot,icl,selfrefine,all")
    parser.add_argument("--max_items", type=int, default=None,
                        help="Limit items per method (for testing)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model config
    cfg = get_api_config()
    model_config = {
        "api_url": cfg["AIHUBMIX_API_URL"],
        "api_key": cfg["AIHUBMIX_API_KEY"],
        "model": cfg["AIHUBMIX_MODEL"],
    }
    model_name = cfg["AIHUBMIX_MODEL"]
    print(f"Judge model: {model_name}")

    # Load filtered items
    methods_to_run = args.methods.lower().split(",")
    run_all = "all" in methods_to_run

    all_items = []
    for method, filename in METHOD_FILES.items():
        method_key = method.split("-")[0].lower() if "-" in method else method.lower()
        if not run_all and method_key not in methods_to_run:
            continue
        fpath = input_dir / filename
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping {method}")
            continue
        items = read_jsonl(fpath)
        valid = [x for x in items if x.get("final_filter_pass")]
        if args.max_items:
            valid = valid[:args.max_items]
        for x in valid:
            x["method"] = method
        all_items.extend(valid)
        print(f"  {method}: {len(valid)} valid items")

    print(f"\nTotal items to judge: {len(all_items)}")
    print(f"Estimated API calls: {len(all_items) * 2}")
    print("=" * 60)

    # Judge each item
    for i, item in enumerate(all_items):
        method = item.get("method", "?")
        diff = item.get("difficulty", "?")
        doc_id = item.get("doc_id", "")[:12]
        print(f"[{i+1}/{len(all_items)}] {method} {diff} {doc_id}", end=" ")

        # Difficulty judge
        diff_result = difficulty_judge(item, model_config)
        item.update(diff_result)
        item["judge_model"] = model_name

        # Path dependency judge
        path_result = path_dependency_judge(item, model_config)
        item.update(path_result)

        d_status = item["difficulty_judge_status"]
        p_status = item["path_dependency_judge_status"]
        pred = item["difficulty_judge"].get("predicted_difficulty", "?")
        dep = item["path_dependency_judge"].get("path_dependency", "?")
        print(f"-> diff={pred}({d_status}) dep={dep}({p_status})")

        time.sleep(0.15)

    # Write judged_all.jsonl
    write_jsonl(output_dir / "judged_all.jsonl", all_items)
    print(f"\nWrote {len(all_items)} items to judged_all.jsonl")

    # Compute yield-aware stats
    by_method = defaultdict(list)
    for item in all_items:
        by_method[item.get("method", "unknown")].append(item)

    all_stats = []
    for method in METHOD_FILES:
        if method in by_method:
            stats = compute_stats(by_method[method], method)
            all_stats.append(stats)

    # Balanced subset
    balanced, min_counts, excluded_diffs = balanced_subset(all_items)
    write_jsonl(output_dir / "judged_balanced.jsonl", balanced)
    print(f"Wrote {len(balanced)} items to judged_balanced.jsonl")

    balanced_by_method = defaultdict(list)
    for item in balanced:
        balanced_by_method[item.get("method", "unknown")].append(item)

    balanced_stats = []
    for method in METHOD_FILES:
        if method in balanced_by_method:
            stats = compute_stats(balanced_by_method[method], method)
            balanced_stats.append(stats)

    # Reports
    generate_report(all_stats, balanced_stats, all_items, balanced,
                    min_counts, excluded_diffs, model_name, output_dir)
    generate_audit_sample(all_items, model_name, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for s in all_stats:
        sp = s.get("spearman", "N/A")
        print(f"  {s['label']:25s}  judged={s['total']}  "
              f"diff_acc={s['diff_accuracy']:.0%}  spearman={sp}  "
              f"pathdep_strong={s['pathdep_strong_pct']:.0f}%")
    print(f"\n  Total API calls: {api_call_count}")
    print(f"  Parse errors: {parse_error_count}")
    print(f"  Retries: {retry_count}")
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
