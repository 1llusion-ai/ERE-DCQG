"""
Baseline generators for DCQG evaluation.
- DirectLLM: context only, no path signal
- PathOnlyQG: path triggers only, no context sentences
- RelationTypeQG: context + relation types, no specific path

All use the same Qwen2.5-7B generator and same evaluator.
"""
import json
import time
import random
import os
from pathlib import Path
from collections import defaultdict

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


# ── Prompt builders for each baseline ────────────────────────

# Generic few-shot example (no path structure revealed)
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
    """Direct LLM: context only, no path."""
    diff = item["difficulty"]
    supporting = item.get("supporting_sentences", [])
    ctx = "\n".join(s if isinstance(s, str) else f"[S{s[0]}] {s[1]}" for s in supporting)

    prompt = f"""{GENERIC_FEW_SHOT}

Now generate:
Context: {ctx}
Difficulty: {diff}

Output ONLY: {{"question": "...", "reasoning_type": "direct|chain|cross_sentence"}}"""
    return prompt


def build_path_only_prompt(item):
    """Path only: event triggers, no context sentences."""
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
    """Relation types + context, no specific path triggers."""
    diff = item["difficulty"]
    rel_types = item.get("relation_subtypes", [])
    supporting = item.get("supporting_sentences", [])
    ctx = "\n".join(s if isinstance(s, str) else f"[S{s[0]}] {s[1]}" for s in supporting)

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


# ── Generator ────────────────────────────────────────────────
def generate_baseline(items, prompt_builder, name, output_path, n_max=None):
    """Generate questions using a baseline method, save incrementally."""
    import urllib.request

    if n_max:
        items = items[:n_max]

    results = []
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(items):
            prompt = prompt_builder(item)
            gold_trigger = item.get("answer_trigger", "")

            # Simple API call (no retries for speed — same as generator)
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {SILICONFLOW_API_KEY}"}
            payload = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "Output ONLY a valid JSON object with 'question' and 'reasoning_type' keys."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 200,
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
                    gen_text = data["choices"][0]["message"]["content"]
            except Exception as e:
                gen_text = f"ERROR: {e}"

            # Parse JSON
            import re
            gen = None
            try:
                gen = json.loads(gen_text)
            except json.JSONDecodeError:
                try:
                    start = gen_text.index("{")
                    end = gen_text.rindex("}") + 1
                    gen = json.loads(gen_text[start:end])
                except (ValueError, json.JSONDecodeError):
                    pass

            question = gen.get("question", "") if isinstance(gen, dict) else ""
            reasoning_type = gen.get("reasoning_type", "unknown") if isinstance(gen, dict) else "error"

            # Apply grammar filter
            grammar_ok, grammar_reason = grammar_filter(question)
            passed = grammar_ok

            # Check trigger leakage
            if passed and gold_trigger:
                q_lower = question.lower()
                if gold_trigger.lower() in q_lower:
                    passed = False
                    grammar_reason = "trigger leakage"

            result = {
                "doc_id": item.get("doc_id", ""),
                "difficulty": item["difficulty"],
                "baseline": name,
                "generated_question": question,
                "gold_answer_trigger": gold_trigger,
                "reasoning_type": reasoning_type,
                "grammar_pass": passed,
                "grammar_reason": grammar_reason,
                "events": item.get("events", []),
                "supporting_sentences": item.get("supporting_sentences", []),
                "relation_subtypes": item.get("relation_subtypes", []),
                "difficulty_score": item.get("difficulty_score", 0),
            }
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


# ── Main: run all baselines ──────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", default="event_qg/outputs/stage2_generation_results.jsonl")
    parser.add_argument("--output_dir", default="event_qg/outputs")
    parser.add_argument("--n_per_level", type=int, default=100)
    parser.add_argument("--skip_generation", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Load items and sample balanced per difficulty
    with open(args.results_file, encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f]

    by_level = defaultdict(list)
    for item in all_items:
        by_level[item["difficulty"]].append(item)

    sampled = []
    for level in ["Easy", "Medium", "Hard"]:
        pool = by_level[level]
        n = min(args.n_per_level, len(pool))
        sampled.extend(random.sample(pool, n))

    random.seed(42)
    random.shuffle(sampled)
    print(f"Sampled {len(sampled)} items for baselines (max {args.n_per_level}/level)")

    baseline_configs = [
        ("DirectLLM", build_direct_llm_prompt),
        ("PathOnlyQG", build_path_only_prompt),
        ("RelationTypeQG", build_relation_type_prompt),
    ]

    # Generate
    baseline_results = {}
    for name, prompt_fn in baseline_configs:
        out_path = output_dir / f"baseline_{name}_generated.jsonl"
        if not args.skip_generation:
            print(f"\n=== Generating {name} ===")
            results = generate_baseline(sampled, prompt_fn, name, out_path)
            baseline_results[name] = results
        else:
            with open(out_path, encoding="utf-8") as f:
                baseline_results[name] = [json.loads(line) for line in f]
            print(f"Loaded {len(baseline_results[name])} existing {name} results")

    # Evaluate each baseline
    print("\n=== Evaluating baselines ===")
    solver = Solver()
    judge = Judge()

    for name, results in baseline_results.items():
        eval_path = output_dir / f"baseline_{name}_evaluated.jsonl"
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
                ans_score, ans_method = judge.score_answerability(solver_ans, gold)
                fluency, relevance, diff_align = judge.score_all(q, solver_ans, gold, path_events, diff)

                composite = 0.35 * ans_score + 0.25 * fluency + 0.20 * relevance + 0.20 * diff_align

                r["solver_answer"] = solver_ans
                r["eval_answerability"] = round(ans_score, 2)
                r["eval_answer_method"] = ans_method
                r["eval_fluency"] = round(fluency, 2)
                r["eval_path_relevance"] = round(relevance, 2)
                r["eval_difficulty_alignment"] = round(diff_align, 2)
                r["eval_composite"] = round(composite, 3)

                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                out_f.flush()
                scored.append(r)

                if (i + 1) % 30 == 0:
                    avg_com = sum(s["eval_composite"] for s in scored) / len(scored)
                    print(f"  [{i+1}/{len(passed)}] composite={avg_com:.3f}", flush=True)

                time.sleep(0.15)

        # Summary
        if scored:
            n = len(scored)
            avg_ans = sum(s["eval_answerability"] for s in scored) / n
            avg_flu = sum(s["eval_fluency"] for s in scored) / n
            avg_rel = sum(s["eval_path_relevance"] for s in scored) / n
            avg_dif = sum(s["eval_difficulty_alignment"] for s in scored) / n
            avg_com = sum(s["eval_composite"] for s in scored) / n
            print(f"\n{name} Summary ({n} questions):")
            print(f"  Answerability: {avg_ans:.3f}")
            print(f"  Fluency:       {avg_flu:.3f}")
            print(f"  Path Relevance:{avg_rel:.3f}")
            print(f"  Difficulty:    {avg_dif:.3f}")
            print(f"  Composite:     {avg_com:.3f}")

            # By level
            for level in ["Easy", "Medium", "Hard"]:
                items_l = [s for s in scored if s["difficulty"] == level]
                if items_l:
                    avg = sum(s["eval_composite"] for s in items_l) / len(items_l)
                    print(f"    {level}: {avg:.3f} (n={len(items_l)})")

    # Final comparison
    print("\n" + "="*60)
    print("METHOD COMPARISON")
    print("="*60)
    our_path = output_dir / "stage2_evaluated.jsonl"
    if our_path.exists():
        with open(our_path, encoding="utf-8") as f:
            our_results = [json.loads(line) for line in f]
        n = len(our_results)
        print(f"Ours (PathQG):")
        print(f"  Answerability: {sum(r['eval_answerability'] for r in our_results)/n:.3f}")
        print(f"  Fluency:       {sum(r['eval_fluency'] for r in our_results)/n:.3f}")
        print(f"  Path Relevance:{sum(r['eval_path_relevance'] for r in our_results)/n:.3f}")
        print(f"  Difficulty:    {sum(r['eval_difficulty_alignment'] for r in our_results)/n:.3f}")
        print(f"  Composite:     {sum(r['eval_composite'] for r in our_results)/n:.3f}")


if __name__ == "__main__":
    main()
