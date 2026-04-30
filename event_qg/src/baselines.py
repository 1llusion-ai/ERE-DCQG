"""
Baseline generators for DCQG evaluation.

Main baselines (target-aware QG comparison):
  - ZeroShotTargetQG: context + target answer + difficulty, no examples
  - ICLTargetQG: context + target answer + difficulty + few-shot examples
  - SelfRefine: ZeroShot + critique + revise (3 API calls)
  - PathQG-HardAware: event path + context + hard-aware constraints (ours)

Ablations (component analysis):
  - PathOnlyQG: event path only, no context
  - RelationTypeQG: context + relation types, no specific path
  - DirectLLM: context only, no path (legacy)

ICL-TargetQG is a target-aware baseline: given the target answer trigger
but not the event path. This matches our task definition of
target-event-grounded question generation.
"""
import json
import time
import random
import os
import re
from pathlib import Path
from collections import defaultdict, Counter

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in open(env_path):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

SILICONFLOW_API_URL = os.environ.get("SILICONFLOW_API_URL", "https://api.siliconflow.cn/v1/chat/completions")
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")

from evaluator import grammar_filter, Solver, Judge, _call_api, _fuzzy_match, _text_similarity
from evaluator_v2 import llm_judge_v2, quality_judge


# ═══════════════════════════════════════════════════════════════
# Fixed sample management
# ═══════════════════════════════════════════════════════════════

def load_or_create_sample(results_file, output_path, n_per_level=100, seed=42):
    """Load or create a fixed sample of items for all baselines.
    Ensures all methods use the same 300 items (100 per difficulty level).
    """
    output_path = Path(output_path)
    if output_path.exists():
        items = [json.loads(l) for l in open(output_path, encoding='utf-8')]
        print(f"Loaded {len(items)} items from {output_path}")
    else:
        all_items = [json.loads(l) for l in open(results_file, encoding='utf-8')]
        by_level = defaultdict(list)
        for item in all_items:
            by_level[item["difficulty"]].append(item)
        random.seed(seed)
        sampled = []
        for level in ["Easy", "Medium", "Hard"]:
            pool = by_level[level]
            n = min(n_per_level, len(pool))
            sampled.extend(random.sample(pool, n))
        random.shuffle(sampled)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in sampled:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        items = sampled
        print(f"Created {len(items)} items -> {output_path}")

    doc_ids = set(item['doc_id'] for item in items)
    diff_counts = Counter(item['difficulty'] for item in items)
    print(f"  Lines: {len(items)}, Unique docs: {len(doc_ids)}, Distribution: {dict(diff_counts)}")
    return items


# ═══════════════════════════════════════════════════════════════
# Difficulty definitions and ICL examples
# ═══════════════════════════════════════════════════════════════

DIFFICULTY_DEFINITIONS = {
    "Easy": "Easy questions are straightforward, answerable from a single sentence in the context.",
    "Medium": "Medium questions require connecting information from 2-3 sentences.",
    "Hard": "Hard questions require synthesizing information across multiple sentences and reasoning chains.",
}

ICL_EXAMPLES = {
    "Easy": """Example 1:
Context: [S0] The army launched an attack on the city. [S1] The city was destroyed.
Target answer: "destroyed"
Difficulty: Easy
Output: {"question": "What happened to the city after the attack?", "answer": "destroyed", "reasoning_type": "direct"}

Example 2:
Context: [S0] The company announced layoffs. [S1] Hundreds of employees lost their jobs.
Target answer: "lost"
Difficulty: Easy
Output: {"question": "What happened to employees after the company announced layoffs?", "answer": "lost", "reasoning_type": "direct"}""",

    "Medium": """Example 1:
Context: [S0] The CEO resigned last Monday. [S1] The board held an emergency meeting. [S2] A new leader was appointed.
Target answer: "appointed"
Difficulty: Medium
Output: {"question": "What was the outcome of the emergency meeting after the CEO resigned?", "answer": "appointed", "reasoning_type": "chain"}

Example 2:
Context: [S0] Heavy rains hit the region. [S1] Rivers overflowed their banks. [S2] Thousands were evacuated.
Target answer: "evacuated"
Difficulty: Medium
Output: {"question": "What happened to residents after the rivers overflowed following heavy rains?", "answer": "evacuated", "reasoning_type": "chain"}""",

    "Hard": """Example 1:
Context: [S0] The government announced austerity measures. [S1] Citizens organized mass protests. [S2] Police deployed tear gas. [S3] Parliament repealed the legislation.
Target answer: "repealed"
Difficulty: Hard
Output: {"question": "What was the final legislative outcome after citizens protested the government's austerity measures and police responded with force?", "answer": "repealed", "reasoning_type": "cross_sentence"}

Example 2:
Context: [S0] The scientist published a controversial paper. [S1] Peer reviewers challenged the methodology. [S2] The journal issued a retraction notice. [S3] The university launched an investigation.
Target answer: "investigation"
Difficulty: Hard
Output: {"question": "What institutional action followed after peer reviewers challenged the controversial paper and the journal retracted it?", "answer": "investigation", "reasoning_type": "cross_sentence"}""",
}


def _format_context(item):
    """Format supporting_sentences as context string."""
    supporting = item.get("supporting_sentences", [])
    return "\n".join(s if isinstance(s, str) else f"[S{s[0]}] {s[1]}" for s in supporting)


def _get_gold_trigger(item):
    """Get gold trigger from item."""
    return item.get("answer_trigger", "") or item.get("gold_answer_trigger", "")


# ═══════════════════════════════════════════════════════════════
# Main baseline prompt builders (target-aware QG)
# ═══════════════════════════════════════════════════════════════

def build_zero_shot_targetqg_prompt(item):
    """ZeroShot-TargetQG: context + target answer + difficulty definition.
    No examples, no event path, no relation subtypes.
    This is a target-aware baseline — it knows the target answer but not the event path."""
    diff = item["difficulty"]
    ctx = _format_context(item)
    gold_trigger = _get_gold_trigger(item)

    return f"""Your task is to generate one question-answer pair according to the following context and target difficulty.

Context:
{ctx}

Target answer:
"{gold_trigger}"

Difficulty Definition:
{DIFFICULTY_DEFINITIONS[diff]}

Requirements:
1. The question must be answerable using only the context.
2. The answer must correspond to the target answer.
3. Do not mention the target answer directly in the question.
4. The question should match the target difficulty ({diff}).
5. The answer field should correspond to the target answer.
6. Output exactly one JSON object.

Output Format:
{{"question": "...", "answer": "...", "reasoning_type": "direct|chain|cross_sentence"}}"""


def build_icl_targetqg_prompt(item):
    """ICL-TargetQG: context + target answer + difficulty + few-shot examples.
    No event path, no relation subtypes.
    This is a target-aware baseline — it knows the target answer but not the event path."""
    diff = item["difficulty"]
    ctx = _format_context(item)
    gold_trigger = _get_gold_trigger(item)
    examples = ICL_EXAMPLES[diff]

    return f"""Your task is to generate one question-answer pair according to the following context and target difficulty.

Context:
{ctx}

Target answer:
"{gold_trigger}"

Difficulty Definition:
{DIFFICULTY_DEFINITIONS[diff]}

Requirements:
1. The question must be answerable using only the context.
2. The answer must correspond to the target answer.
3. Do not mention the target answer directly in the question.
4. The question should match the target difficulty ({diff}).
5. The answer field should correspond to the target answer.
6. Output exactly one JSON object.

Here are several examples:
{examples}

Output Format:
{{"question": "...", "answer": "...", "reasoning_type": "direct|chain|cross_sentence"}}"""


# ═══════════════════════════════════════════════════════════════
# SelfRefine v2 (based on ZeroShot-TargetQG)
# ═══════════════════════════════════════════════════════════════

def build_self_refine_v2_prompt(item):
    """SelfRefine step 1: Generate initial question using ZeroShot-TargetQG.
    Returns (gen_prompt, ctx, diff, gold_trigger) for the 3-step flow."""
    prompt = build_zero_shot_targetqg_prompt(item)
    diff = item["difficulty"]
    ctx = _format_context(item)
    gold_trigger = _get_gold_trigger(item)
    return prompt, ctx, diff, gold_trigger


def self_refine_critique_v2_prompt(question, ctx, diff, gold_trigger):
    """SelfRefine step 2: Critique the generated question."""
    return f"""Context:
{ctx}

Generated question: "{question}"
Target answer: "{gold_trigger}"
Target difficulty: {diff}

Please critique this question on the following criteria:
1. Grammar: Is the question grammatically correct and natural?
2. Difficulty: Does the question match the target difficulty ({diff})?
3. Answerability: Can the question be answered using ONLY the context?
4. Answer leakage: Does the question directly mention "{gold_trigger}"?
5. Completeness: Does the question lead to the target answer?

Reply with ONLY: {{"issues": ["issue1", ...], "overall_quality": "good" | "needs_revision"}}"""


def self_refine_revise_v2_prompt(question, critique, ctx, diff, gold_trigger):
    """SelfRefine step 3: Revise based on critique."""
    return f"""Context:
{ctx}

Original question: "{question}"
Critique: {critique}
Target answer: "{gold_trigger}" (do NOT mention in question)
Target difficulty: {diff}

Revise the question to fix the issues identified in the critique.
- Question must start with a question word and end with "?".
- Do NOT use the word "{gold_trigger}" in the question.
- The question must match the target difficulty ({diff}).
- The answer must correspond to the target answer.

Output ONLY: {{"question": "...", "answer": "...", "reasoning_type": "direct|chain|cross_sentence"}}"""


# ═══════════════════════════════════════════════════════════════
# Ablation prompt builders (legacy, kept for component analysis)
# ═══════════════════════════════════════════════════════════════

GENERIC_FEW_SHOT = """Example 1 (Easy):
Context: [S0] The army launched an attack on the city. [S1] The city was destroyed.
Difficulty: Easy
Output: {"question": "What happened to the city after the attack?", "reasoning_type": "direct"}

Example 2 (Medium):
Context: [S0] The CEO resigned last Monday. [S1] The board appointed a new leader. [S2] The new CEO implemented reforms.
Difficulty: Medium
Output: {"question": "What did the new CEO do after being appointed?", "reasoning_type": "chain"}

Example 3 (Hard):
Context: [S0] The government announced budget cuts. [S1] Citizens protested the decision. [S2] Officials canceled the policy. [S3] The announcement had been controversial.
Difficulty: Hard
Output: {"question": "What was the final outcome after citizens protested the budget announcement?", "reasoning_type": "cross_sentence"}"""


def build_direct_llm_prompt(item):
    """Direct LLM: context only, no path. (Legacy ablation)"""
    diff = item["difficulty"]
    ctx = _format_context(item)

    prompt = f"""{GENERIC_FEW_SHOT}

Now generate:
Context: {ctx}
Difficulty: {diff}

Output ONLY: {{"question": "...", "reasoning_type": "direct|chain|cross_sentence"}}"""
    return prompt


def build_path_only_prompt(item):
    """Path only: event triggers, no context sentences. (Ablation)"""
    diff = item["difficulty"]
    events = item["events"]
    path_str = " → ".join(e["trigger"] for e in events)
    final = events[-1]["trigger"]

    prompt = f"""Example:
Path: "attack" → "destroy"
Difficulty: Easy
Output: {{"question": "What happened to the city after the army attacked it?", "reasoning_type": "direct"}}

Now generate:
Path: {path_str}
Difficulty: {diff}
- The answer is the final event ("{final}"), do NOT mention it in the question
- Output ONLY: {{"question": "...", "reasoning_type": "direct|chain|cross_sentence"}}"""
    return prompt


def build_relation_type_prompt(item):
    """Relation types + context, no specific path triggers. (Ablation)"""
    diff = item["difficulty"]
    rel_types = item.get("relation_subtypes", [])
    ctx = _format_context(item)

    rel_cats = set()
    for rt in rel_types:
        if rt.startswith("CAUSE"):
            rel_cats.add("causal")
        elif rt.startswith("TEMPORAL"):
            rel_cats.add("temporal")
        elif rt.startswith("SUBEVENT"):
            rel_cats.add("subevent")
    rel_str = ", ".join(sorted(rel_cats)) if rel_cats else "unknown"

    prompt = f"""{GENERIC_FEW_SHOT}

Now generate a question where the answer is an event connected through {rel_str} relations:
Context: {ctx}
Difficulty: {diff}

Output ONLY: {{"question": "...", "reasoning_type": "direct|chain|cross_sentence"}}"""
    return prompt


# ═══════════════════════════════════════════════════════════════
# Legacy SelfRefine (kept for backward compatibility)
# ═══════════════════════════════════════════════════════════════

def build_self_refine_prompt(item):
    """Legacy Self-Refine step 1. Use build_self_refine_v2_prompt for new runs."""
    diff = item["difficulty"]
    ctx = _format_context(item)
    gold_trigger = _get_gold_trigger(item)

    gen_prompt = f"""{GENERIC_FEW_SHOT}

Now generate a {diff} question:
Context: {ctx}
Difficulty: {diff}

Output ONLY: {{"question": "...", "reasoning_type": "direct|chain|cross_sentence"}}"""

    return gen_prompt, ctx, diff, gold_trigger


def self_refine_critique_prompt(question, ctx, diff, gold_trigger):
    """Legacy Self-Refine step 2."""
    return f"""Context: {ctx}
Question: "{question}"
Difficulty: {diff}
Gold answer trigger: "{gold_trigger}"

Critique this question:
1. Is it grammatically correct?
2. Does it match the difficulty level ({diff})?
3. Can it be answered from the context?
4. Is the answer "{gold_trigger}" NOT mentioned in the question?

Reply with ONLY: {{"issues": ["issue1", "issue2", ...], "overall_quality": "good/needs_revision"}}"""


def self_refine_revise_prompt(question, critique, ctx, diff, gold_trigger):
    """Legacy Self-Refine step 3."""
    return f"""Original question: "{question}"
Critique: {critique}
Context: {ctx}
Difficulty: {diff}
Gold answer: "{gold_trigger}" (do NOT mention in question)

Revise the question to fix the issues identified in the critique.
- Question must start with a question word and end with "?".
- Do NOT use the word "{gold_trigger}" in the question.

Output ONLY: {{"question": "...", "reasoning_type": "direct|chain|cross_sentence"}}"""


# ═══════════════════════════════════════════════════════════════
# API call helper
# ═══════════════════════════════════════════════════════════════

def _call_llm(prompt, temperature=0.1, max_tokens=200, timeout=90):
    """Single API call to SiliconFlow. Returns raw text."""
    import urllib.request
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {SILICONFLOW_API_KEY}"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Output ONLY a valid JSON object."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": ["\n\n"],
    }
    try:
        req = urllib.request.Request(SILICONFLOW_API_URL, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"


def _parse_json_response(text):
    """Parse JSON from LLM response, with fallback substring extraction."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return None


def _make_result(item, question, reasoning_type, name, grammar_pass, grammar_reason,
                 generation_prompts=None, generation_raw_responses=None):
    """Build a result dict for a generated question."""
    result = {
        "doc_id": item.get("doc_id", ""),
        "difficulty": item["difficulty"],
        "method": name,
        "generated_question": question,
        "gold_answer_trigger": _get_gold_trigger(item),
        "reasoning_type": reasoning_type,
        "grammar_pass": grammar_pass,
        "grammar_reason": grammar_reason,
        "events": item.get("events", []),
        "supporting_sentences": item.get("supporting_sentences", []),
        "relation_subtypes": item.get("relation_subtypes", []),
        "difficulty_score": item.get("difficulty_score", 0),
    }
    if generation_prompts is not None:
        result["generation_prompts"] = generation_prompts
    if generation_raw_responses is not None:
        result["generation_raw_responses"] = generation_raw_responses
    return result


# ═══════════════════════════════════════════════════════════════
# Generators
# ═══════════════════════════════════════════════════════════════

def generate_baseline(items, prompt_builder, name, output_path, n_max=None):
    """Generate questions using a baseline method, save incrementally."""
    if n_max:
        items = items[:n_max]

    results = []
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(items):
            prompt = prompt_builder(item)
            gold_trigger = _get_gold_trigger(item)
            gen_text = _call_llm(prompt)
            gen = _parse_json_response(gen_text)

            question = gen.get("question", "") if isinstance(gen, dict) else ""
            reasoning_type = gen.get("reasoning_type", "unknown") if isinstance(gen, dict) else "error"

            grammar_ok, grammar_reason = grammar_filter(question)
            passed = grammar_ok

            if passed and gold_trigger:
                if gold_trigger.lower() in question.lower():
                    passed = False
                    grammar_reason = "trigger leakage"

            result = _make_result(item, question, reasoning_type, name, passed, grammar_reason,
                                  generation_prompts=[prompt],
                                  generation_raw_responses=[gen_text or ""])
            results.append(result)
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            if (i + 1) % 20 == 0:
                n_pass = sum(1 for r in results if r["grammar_pass"])
                print(f"  [{name}] {i+1}/{len(items)} passed={n_pass}", flush=True)

            time.sleep(0.1)

    n_pass = sum(1 for r in results if r["grammar_pass"])
    print(f"[{name}] Generated {len(results)}, passed={n_pass} ({n_pass/len(results)*100:.0f}%)")
    return results


def generate_self_refine_v2(items, output_path, n_max=None):
    """SelfRefine v2: ZeroShot-TargetQG + critique + revise (3 API calls per item).
    Based on Madaan et al. (2023). No event graph structure."""
    if n_max:
        items = items[:n_max]

    results = []
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(items):
            gen_prompt, ctx, diff, gold_trigger = build_self_refine_v2_prompt(item)

            all_prompts = [gen_prompt]
            all_raws = []

            # Step 1: Generate initial question
            text = _call_llm(gen_prompt)
            all_raws.append(text or "")
            gen = _parse_json_response(text)
            question = gen.get("question", "") if isinstance(gen, dict) else ""

            # Step 2: Critique (only if we have a question)
            critique_text = ""
            if question.strip():
                critique_prompt = self_refine_critique_v2_prompt(question, ctx, diff, gold_trigger)
                all_prompts.append(critique_prompt)
                critique_text = _call_llm(critique_prompt, max_tokens=150)
                all_raws.append(critique_text or "")

            # Step 3: Revise (only if critique says needs_revision)
            if question.strip() and critique_text and "needs_revision" in critique_text.lower():
                revise_prompt = self_refine_revise_v2_prompt(question, critique_text, ctx, diff, gold_trigger)
                all_prompts.append(revise_prompt)
                text = _call_llm(revise_prompt)
                all_raws.append(text or "")
                gen = _parse_json_response(text)
                revised = gen.get("question", "") if isinstance(gen, dict) else ""
                if revised.strip():
                    question = revised

            grammar_ok, grammar_reason = grammar_filter(question)
            passed = grammar_ok

            if passed and gold_trigger:
                if gold_trigger.lower() in question.lower():
                    passed = False
                    grammar_reason = "trigger leakage"

            result = _make_result(item, question, "self_refine", "SelfRefine", passed, grammar_reason,
                                  generation_prompts=all_prompts,
                                  generation_raw_responses=all_raws)
            results.append(result)
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            if (i + 1) % 20 == 0:
                n_pass = sum(1 for r in results if r["grammar_pass"])
                print(f"  [SelfRefine] {i+1}/{len(items)} passed={n_pass}", flush=True)

            time.sleep(0.15)

    n_pass = sum(1 for r in results if r["grammar_pass"])
    print(f"[SelfRefine] Generated {len(results)}, passed={n_pass} ({n_pass/len(results)*100:.0f}%)")
    return results


# Legacy SelfRefine generator (kept for backward compatibility)
def generate_self_refine(items, output_path, n_max=None):
    """Legacy SelfRefine. Use generate_self_refine_v2 for new runs."""
    import urllib.request

    if n_max:
        items = items[:n_max]

    results = []
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(items):
            gen_prompt, ctx, diff, gold_trigger = build_self_refine_prompt(item)

            text = _call_llm(gen_prompt)
            gen = _parse_json_response(text)
            question = gen.get("question", "") if isinstance(gen, dict) else ""

            critique_text = ""
            if question.strip():
                critique_prompt = self_refine_critique_prompt(question, ctx, diff, gold_trigger)
                critique_text = _call_llm(critique_prompt, max_tokens=150)

            if question.strip() and critique_text and "needs_revision" in critique_text.lower():
                revise_prompt = self_refine_revise_prompt(question, critique_text, ctx, diff, gold_trigger)
                text = _call_llm(revise_prompt)
                gen = _parse_json_response(text)
                revised = gen.get("question", "") if isinstance(gen, dict) else ""
                if revised.strip():
                    question = revised

            grammar_ok, grammar_reason = grammar_filter(question)
            passed = grammar_ok

            if passed and gold_trigger:
                if gold_trigger.lower() in question.lower():
                    passed = False
                    grammar_reason = "trigger leakage"

            result = _make_result(item, question, "self_refine", "SelfRefine", passed, grammar_reason)
            results.append(result)
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            if (i + 1) % 20 == 0:
                n_pass = sum(1 for r in results if r["grammar_pass"])
                print(f"  [SelfRefine] {i+1}/{len(items)} passed={n_pass}", flush=True)

            time.sleep(0.15)

    n_pass = sum(1 for r in results if r["grammar_pass"])
    print(f"[SelfRefine] Generated {len(results)}, passed={n_pass} ({n_pass/len(results)*100:.0f}%)")
    return results


# ═══════════════════════════════════════════════════════════════
# Unified evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate_method(results, name, output_dir, solver=None):
    """Evaluate a method's results using unified LLM judge.
    Returns list of scored items."""
    if solver is None:
        solver = Solver()

    eval_path = output_dir / f"{name}_evaluated_llm.jsonl"
    passed = [r for r in results if r["grammar_pass"]]
    print(f"\n{name}: {len(passed)} grammar-passed, evaluating...")

    scored = []
    with open(eval_path, "w", encoding="utf-8") as out_f:
        for i, r in enumerate(passed):
            q = r["generated_question"]
            ctx = "\n".join(
                s if isinstance(s, str) else s[1]
                for s in r.get("supporting_sentences", [])
            )
            gold = r["gold_answer_trigger"]
            path_events = r.get("events", [])
            diff = r["difficulty"]

            solver_ans = solver.answer(q, ctx)
            answerable, solver_correct, support_covered = llm_judge_v2(q, ctx, gold, solver_ans)
            fluency, relevance, diff_align = quality_judge(q, path_events, diff)

            composite = round(
                0.25 * solver_correct +
                0.20 * answerable +
                0.15 * support_covered +
                0.15 * fluency +
                0.10 * relevance +
                0.15 * diff_align,
                3
            )

            r["solver_answer"] = solver_ans
            r["judge_answerable"] = round(answerable, 2)
            r["judge_solver_correct"] = round(solver_correct, 2)
            r["judge_support_covered"] = round(support_covered, 2)
            r["quality_fluency"] = round(fluency, 2)
            r["quality_path_relevance"] = round(relevance, 2)
            r["quality_difficulty_alignment"] = round(diff_align, 2)
            r["composite"] = composite

            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out_f.flush()
            scored.append(r)

            if (i + 1) % 30 == 0:
                avg_com = sum(s["composite"] for s in scored) / len(scored)
                avg_cor = sum(s["judge_solver_correct"] for s in scored) / len(scored)
                print(f"  [{i+1}/{len(passed)}] composite={avg_com:.3f} solver_correct={avg_cor:.3f}", flush=True)

            time.sleep(0.15)

    # Summary
    if scored:
        n = len(scored)
        print(f"\n{name} Summary ({n} questions):")
        print(f"  Judge Answerable:     {sum(s['judge_answerable'] for s in scored)/n:.3f}")
        print(f"  Judge Solver Correct: {sum(s['judge_solver_correct'] for s in scored)/n:.3f}")
        print(f"  Judge Support Covered:{sum(s['judge_support_covered'] for s in scored)/n:.3f}")
        print(f"  Quality Fluency:      {sum(s['quality_fluency'] for s in scored)/n:.3f}")
        print(f"  Quality Path Relevance:{sum(s['quality_path_relevance'] for s in scored)/n:.3f}")
        print(f"  Quality Difficulty:   {sum(s['quality_difficulty_alignment'] for s in scored)/n:.3f}")
        print(f"  Composite:            {sum(s['composite'] for s in scored)/n:.3f}")

        for level in ["Easy", "Medium", "Hard"]:
            items_l = [s for s in scored if s["difficulty"] == level]
            if items_l:
                avg = sum(s["composite"] for s in items_l) / len(items_l)
                cor = sum(s["judge_solver_correct"] for s in items_l) / len(items_l)
                ans = sum(s["judge_answerable"] for s in items_l) / len(items_l)
                print(f"    {level}: composite={avg:.3f} solver_correct={cor:.3f} answerable={ans:.3f} (n={len(items_l)})")

    return scored


def print_comparison_table(all_scored):
    """Print a comparison table of all methods."""
    print("\n" + "=" * 80)
    print("METHOD COMPARISON (Unified LLM Judge)")
    print("=" * 80)

    header = f"{'Method':<25} {'N':>4} {'Pass':>4} {'Pass%':>6} {'Ansble':>7} {'SolCor':>7} {'Comp':>7}"
    print(header)
    print("-" * len(header))

    for name, (results, scored) in all_scored.items():
        n_gen = len(results)
        n_pass = sum(1 for r in results if r["grammar_pass"])
        n_scored = len(scored)
        pass_rate = n_pass / n_gen * 100 if n_gen > 0 else 0
        avg_ans = sum(s["judge_answerable"] for s in scored) / n_scored if scored else 0
        avg_cor = sum(s["judge_solver_correct"] for s in scored) / n_scored if scored else 0
        avg_com = sum(s["composite"] for s in scored) / n_scored if scored else 0
        print(f"{name:<25} {n_gen:>4} {n_pass:>4} {pass_rate:>5.1f}% {avg_ans:>7.3f} {avg_cor:>7.3f} {avg_com:>7.3f}")

    # By difficulty
    for d in ["Easy", "Medium", "Hard"]:
        print(f"\n--- {d} ---")
        print(header)
        print("-" * len(header))
        for name, (results, scored) in all_scored.items():
            d_results = [r for r in results if r["difficulty"] == d]
            d_scored = [s for s in scored if s["difficulty"] == d]
            n_gen = len(d_results)
            n_pass = sum(1 for r in d_results if r["grammar_pass"])
            pass_rate = n_pass / n_gen * 100 if n_gen > 0 else 0
            n_scored = len(d_scored)
            avg_ans = sum(s["judge_answerable"] for s in d_scored) / n_scored if d_scored else 0
            avg_cor = sum(s["judge_solver_correct"] for s in d_scored) / n_scored if d_scored else 0
            avg_com = sum(s["composite"] for s in d_scored) / n_scored if d_scored else 0
            print(f"{name:<25} {n_gen:>4} {n_pass:>4} {pass_rate:>5.1f}% {avg_ans:>7.3f} {avg_cor:>7.3f} {avg_com:>7.3f}")


def compute_fair_metrics(results, scored, n_total=None):
    """Compute macro-average, end-to-end, and monotonicity metrics.
    Returns dict with all fair metrics."""
    if n_total is None:
        n_total = len(results)

    n_scored = len(scored)
    if n_scored == 0:
        return {}

    # Per-difficulty solver_correct
    by_diff = {}
    for level in ["Easy", "Medium", "Hard"]:
        items_l = [s for s in scored if s["difficulty"] == level]
        if items_l:
            by_diff[level] = {
                "n": len(items_l),
                "solver_correct": sum(s["judge_solver_correct"] for s in items_l) / len(items_l),
                "answerable": sum(s["judge_answerable"] for s in items_l) / len(items_l),
                "composite": sum(s["composite"] for s in items_l) / len(items_l),
            }
        else:
            by_diff[level] = {"n": 0, "solver_correct": 0, "answerable": 0, "composite": 0}

    easy_sol = by_diff["Easy"]["solver_correct"]
    med_sol = by_diff["Medium"]["solver_correct"]
    hard_sol = by_diff["Hard"]["solver_correct"]

    # Macro-average: (Easy mean + Medium mean + Hard mean) / 3
    macro_sol = (easy_sol + med_sol + hard_sol) / 3
    macro_ans = (by_diff["Easy"]["answerable"] + by_diff["Medium"]["answerable"] + by_diff["Hard"]["answerable"]) / 3
    macro_comp = (by_diff["Easy"]["composite"] + by_diff["Medium"]["composite"] + by_diff["Hard"]["composite"]) / 3

    # End-to-end: total score / n_total (fail=0)
    e2e_sol = sum(s["judge_solver_correct"] for s in scored) / n_total
    e2e_ans = sum(s["judge_answerable"] for s in scored) / n_total
    e2e_comp = sum(s["composite"] for s in scored) / n_total

    # Monotonicity metrics
    e_h_gap = easy_sol - hard_sol
    dc_score = max(0, easy_sol - med_sol) + max(0, med_sol - hard_sol)
    violation_penalty = max(0, med_sol - easy_sol) + max(0, hard_sol - med_sol)
    violations = 0
    if easy_sol < med_sol:
        violations += 1
    if med_sol < hard_sol:
        violations += 1

    return {
        "by_diff": by_diff,
        "easy_sol": easy_sol, "med_sol": med_sol, "hard_sol": hard_sol,
        "e_h_gap": e_h_gap,
        "dc_score": dc_score,
        "violation_penalty": violation_penalty,
        "violations": violations,
        "macro_sol": macro_sol, "macro_ans": macro_ans, "macro_comp": macro_comp,
        "e2e_sol": e2e_sol, "e2e_ans": e2e_ans, "e2e_comp": e2e_comp,
        "n_total": n_total, "n_scored": n_scored,
    }


def print_fair_metrics_table(all_fair):
    """Print difficulty control and fair metrics tables."""
    print("\n" + "=" * 90)
    print("DIFFICULTY CONTROL (Primary Metrics)")
    print("=" * 90)
    header = f"{'Method':<25} {'Easy SolCor':>11} {'Med SolCor':>10} {'Hard SolCor':>11} {'E-H gap':>7} {'DC Score':>8} {'Violations':>10}"
    print(header)
    print("-" * len(header))
    for name, fair in all_fair.items():
        print(f"{name:<25} {fair['easy_sol']:>11.3f} {fair['med_sol']:>10.3f} {fair['hard_sol']:>11.3f} {fair['e_h_gap']:>7.3f} {fair['dc_score']:>8.3f} {fair['violations']:>10}")

    print("\n" + "=" * 90)
    print("FAIR METRICS (Secondary)")
    print("=" * 90)
    header2 = f"{'Method':<25} {'Pass%':>6} {'Cond SolCor':>11} {'Macro SolCor':>12} {'E2E SolCor':>10} {'Cond Comp':>9} {'Macro Comp':>10} {'E2E Comp':>8}"
    print(header2)
    print("-" * len(header2))
    for name, fair in all_fair.items():
        pass_pct = fair['n_scored'] / fair['n_total'] * 100 if fair['n_total'] > 0 else 0
        print(f"{name:<25} {pass_pct:>5.1f}% {fair['macro_sol']:>11.3f} {fair['macro_sol']:>12.3f} {fair['e2e_sol']:>10.3f} {fair['macro_comp']:>9.3f} {fair['macro_comp']:>10.3f} {fair['e2e_comp']:>8.3f}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run baselines and ablations for DCQG evaluation.")
    parser.add_argument("--results_file", default="event_qg/outputs/stage2_generation_results.jsonl")
    parser.add_argument("--output_dir", default="event_qg/outputs")
    parser.add_argument("--n_per_level", type=int, default=100)
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--skip_ablations", action="store_true")
    parser.add_argument("--skip_pathqg", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load fixed sample ──────────────────────────────
    sample_path = output_dir / "sample_300_seed42.jsonl"
    sampled = load_or_create_sample(args.results_file, sample_path, args.n_per_level)

    # ── Method configs ─────────────────────────────────────────
    # Main baselines (target-aware QG comparison)
    MAIN_BASELINES = [
        ("baseline_ZeroShotTargetQG", build_zero_shot_targetqg_prompt),
        ("baseline_ICLTargetQG", build_icl_targetqg_prompt),
    ]

    # Ablations (component analysis)
    ABLATION_BASELINES = [
        ("ablation_PathOnlyQG", build_path_only_prompt),
        ("ablation_RelationTypeQG", build_relation_type_prompt),
    ]

    # ── Step 2: Generate ───────────────────────────────────────
    all_results = {}

    # Main baselines
    for name, prompt_fn in MAIN_BASELINES:
        out_path = output_dir / f"{name}_generated.jsonl"
        if not args.skip_generation:
            print(f"\n=== Generating {name} ===")
            results = generate_baseline(sampled, prompt_fn, name, out_path)
            all_results[name] = results
        else:
            with open(out_path, encoding="utf-8") as f:
                all_results[name] = [json.loads(line) for line in f]
            print(f"Loaded {len(all_results[name])} existing {name} results")

    # SelfRefine v2
    sr_name = "baseline_SelfRefine"
    sr_out_path = output_dir / f"{sr_name}_generated.jsonl"
    if not args.skip_generation:
        print(f"\n=== Generating {sr_name} ===")
        sr_results = generate_self_refine_v2(sampled, sr_out_path)
        all_results[sr_name] = sr_results
    else:
        with open(sr_out_path, encoding="utf-8") as f:
            all_results[sr_name] = [json.loads(line) for line in f]
        print(f"Loaded {len(all_results[sr_name])} existing {sr_name} results")

    # PathQG-HardAware (ours) — must be regenerated on same sample
    pqg_name = "baseline_PathQGHardAware"
    pqg_out_path = output_dir / f"{pqg_name}_generated.jsonl"
    if not args.skip_generation and not args.skip_pathqg:
        print(f"\n=== Generating {pqg_name} (ours) ===")
        from compare_hardaware import generate_with_retry_hardaware
        pqg_results = []
        with open(pqg_out_path, "w", encoding="utf-8") as out_f:
            for i, item in enumerate(sampled):
                result, attempts = generate_with_retry_hardaware(item, max_attempts=3)
                if result:
                    result["method"] = pqg_name
                    pqg_results.append(result)
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
                if (i + 1) % 20 == 0:
                    n_pass = sum(1 for r in pqg_results if r.get("grammar_pass", False))
                    print(f"  [PathQG-HardAware] {i+1}/{len(sampled)} passed={n_pass}", flush=True)
                time.sleep(0.1)
        all_results[pqg_name] = pqg_results
        n_pass = sum(1 for r in pqg_results if r.get("grammar_pass", False))
        print(f"[PathQG-HardAware] Generated {len(pqg_results)}, passed={n_pass} ({n_pass/len(pqg_results)*100:.0f}%)")
    elif pqg_out_path.exists():
        with open(pqg_out_path, encoding="utf-8") as f:
            all_results[pqg_name] = [json.loads(line) for line in f]
        print(f"Loaded {len(all_results[pqg_name])} existing {pqg_name} results")

    # Ablations
    if not args.skip_ablations:
        for name, prompt_fn in ABLATION_BASELINES:
            out_path = output_dir / f"{name}_generated.jsonl"
            if not args.skip_generation:
                print(f"\n=== Generating {name} (ablation) ===")
                results = generate_baseline(sampled, prompt_fn, name, out_path)
                all_results[name] = results
            else:
                with open(out_path, encoding="utf-8") as f:
                    all_results[name] = [json.loads(line) for line in f]
                print(f"Loaded {len(all_results[name])} existing {name} results")

    # ── Step 3: Evaluate ───────────────────────────────────────
    if not args.skip_evaluation:
        print("\n=== Evaluating all methods (Unified LLM Judge) ===")
        solver = Solver()
        all_scored = {}

        for name, results in all_results.items():
            scored = evaluate_method(results, name, output_dir, solver)
            all_scored[name] = (results, scored)

        # ── Step 4: Comparison table ───────────────────────────
        print_comparison_table(all_scored)

    print("\nDone.")


if __name__ == "__main__":
    main()
