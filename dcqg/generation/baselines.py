"""Baseline generators for DCQG evaluation.

Main baselines (target-aware QG comparison):
  - ZeroShotTargetQG: context + target answer + difficulty, no examples
  - ICLTargetQG: context + target answer + difficulty + few-shot examples
  - SelfRefine: ZeroShot + critique + revise (3 API calls)
  - PathQG-HardAware: event path + context + hard-aware constraints (ours)

Ablations (component analysis):
  - PathOnlyQG: event path only, no context
  - RelationTypeQG: context + relation types, no specific path
  - DirectLLM: context only, no path (legacy)
"""
import json
import time
import random
import re
from pathlib import Path
from collections import defaultdict, Counter

from dcqg.utils.api_client import call_api
from dcqg.utils.text import fuzzy_match, text_similarity
from dcqg.question_filter.grammar import grammar_filter
from dcqg.evaluation.solver import Solver, Judge
from dcqg.evaluation.judge import llm_judge_v2, quality_judge


# ── Fixed sample management ──────────────────────────────────

def load_or_create_sample(results_file, output_path, n_per_level=100, seed=42):
    """Load or create a fixed sample of items for all baselines.
    Ensures all methods use the same 300 items (100 per difficulty level).
    """
    output_path = Path(output_path)
    if output_path.exists():
        items = [json.loads(l) for l in open(output_path, encoding="utf-8")]
        print(f"Loaded {len(items)} items from {output_path}")
    else:
        all_items = [json.loads(l) for l in open(results_file, encoding="utf-8")]
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
        with open(output_path, "w", encoding="utf-8") as f:
            for item in sampled:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        items = sampled
        print(f"Created {len(items)} items -> {output_path}")

    doc_ids = set(item["doc_id"] for item in items)
    diff_counts = Counter(item["difficulty"] for item in items)
    print(f"  Lines: {len(items)}, Unique docs: {len(doc_ids)}, Distribution: {dict(diff_counts)}")
    return items


# ── Difficulty definitions and ICL examples ───────────────────

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


# ── Helpers ───────────────────────────────────────────────────

def _format_context(item):
    """Format supporting_sentences as context string."""
    supporting = item.get("supporting_sentences", [])
    return "\n".join(s if isinstance(s, str) else f"[S{s[0]}] {s[1]}" for s in supporting)


def _get_gold_trigger(item):
    """Get gold trigger from item."""
    return item.get("answer_trigger", "") or item.get("gold_answer_trigger", "")


def _call_llm(prompt, temperature=0.1, max_tokens=200, timeout=90):
    """Single API call to SiliconFlow. Returns raw text or error string."""
    resp = call_api(prompt, system="Output ONLY a valid JSON object.",
                    temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    return resp if resp else "ERROR: empty response"


def _parse_json_response(text):
    """Parse JSON from LLM response, with fallback substring extraction."""
    if not text or text.startswith("ERROR"):
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError, TypeError):
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
    }
    if generation_prompts is not None:
        result["generation_prompts"] = generation_prompts
    if generation_raw_responses is not None:
        result["generation_raw_responses"] = generation_raw_responses
    return result


# ── Main baseline prompt builders (target-aware QG) ──────────

def build_zero_shot_targetqg_prompt(item):
    """ZeroShot-TargetQG: context + target answer + difficulty definition."""
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
    """ICL-TargetQG: context + target answer + difficulty + few-shot examples."""
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


# ── SelfRefine v2 (based on ZeroShot-TargetQG) ───────────────

def build_self_refine_v2_prompt(item):
    """SelfRefine step 1: Generate initial question using ZeroShot-TargetQG."""
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


# ── Ablation prompt builders ─────────────────────────────────

def build_direct_llm_prompt(item):
    """Direct LLM: context only, no path. (Legacy ablation)"""
    diff = item["difficulty"]
    ctx = _format_context(item)

    return f"""{GENERIC_FEW_SHOT}

Now generate:
Context: {ctx}
Difficulty: {diff}

Output ONLY: {{"question": "...", "reasoning_type": "direct|chain|cross_sentence"}}"""


def build_path_only_prompt(item):
    """Path only: event triggers, no context sentences. (Ablation)"""
    diff = item["difficulty"]
    events = item["events"]
    path_str = " -> ".join(e["trigger"] for e in events)
    final = events[-1]["trigger"]

    return f"""Example:
Path: "attack" -> "destroy"
Difficulty: Easy
Output: {{"question": "What happened to the city after the army attacked it?", "reasoning_type": "direct"}}

Now generate:
Path: {path_str}
Difficulty: {diff}
- The answer is the final event ("{final}"), do NOT mention it in the question
- Output ONLY: {{"question": "...", "reasoning_type": "direct|chain|cross_sentence"}}"""


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

    return f"""{GENERIC_FEW_SHOT}

Now generate a question where the answer is an event connected through {rel_str} relations:
Context: {ctx}
Difficulty: {diff}

Output ONLY: {{"question": "...", "reasoning_type": "direct|chain|cross_sentence"}}"""


# ── Legacy SelfRefine (kept for backward compatibility) ──────

def build_self_refine_prompt(item):
    """Legacy Self-Refine step 1."""
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


# ── Generators ────────────────────────────────────────────────

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
    """SelfRefine v2: ZeroShot-TargetQG + critique + revise (3 API calls per item)."""
    if n_max:
        items = items[:n_max]

    results = []
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(items):
            gen_prompt, ctx, diff, gold_trigger = build_self_refine_v2_prompt(item)

            all_prompts = [gen_prompt]
            all_raws = []

            text = _call_llm(gen_prompt)
            all_raws.append(text or "")
            gen = _parse_json_response(text)
            question = gen.get("question", "") if isinstance(gen, dict) else ""

            critique_text = ""
            if question.strip():
                critique_prompt = self_refine_critique_v2_prompt(question, ctx, diff, gold_trigger)
                all_prompts.append(critique_prompt)
                critique_text = _call_llm(critique_prompt, max_tokens=150)
                all_raws.append(critique_text or "")

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


# ── Unified evaluation ────────────────────────────────────────

def evaluate_method(results, name, output_dir, solver=None):
    """Evaluate a method's results using unified LLM judge."""
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
                3,
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
