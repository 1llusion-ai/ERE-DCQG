"""
Unified method comparison: all methods on identical items.
- Samples items once with fixed seed
- Generates questions with all 4 methods on same items
- Prevents output duplication (deletes before write)
- Reports generation pass rate separately from quality metrics
- Computes matched-set quality comparison
"""
import json
import time
import random
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Load .env for API
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

from evaluator_v2 import grammar_filter, evaluate_item, _call_api


# ── Generation prompts (4 methods) ──────────────────────────

FEW_SHOT = """Example 1 (Easy):
Path: "attack" → "destroy"
Context: [S0] The army launched an attack on the city. [S1] The city was destroyed.
Difficulty: Easy
Output: {"question": "What happened to the city after the army attacked it?", "reasoning_type": "direct"}

Example 2 (Medium):
Path: "resign" → "appoint" → "implement"
Context: [S0] The CEO resigned. [S1] The board appointed a new leader. [S2] The new CEO implemented reforms.
Difficulty: Medium
Output: {"question": "What did the new CEO do after being appointed?", "reasoning_type": "chain"}

Example 3 (Hard):
Path: "announce" → "protest" → "cancel"
Context: [S0] The government announced budget cuts. [S1] Citizens protested. [S2] Officials canceled the policy. [S3] The announcement had been controversial.
Difficulty: Hard
Output: {"question": "What was the final outcome after citizens protested the budget announcement?", "reasoning_type": "cross_sentence"}"""

GENERIC_FEW_SHOT = """Example 1:
Context: [S0] The army launched an attack. [S1] The city was destroyed.
Difficulty: Easy
Output: {"question": "What happened to the city after the attack?", "reasoning_type": "direct"}

Example 2:
Context: [S0] The CEO resigned. [S1] The board appointed a new leader. [S2] The new CEO implemented reforms.
Difficulty: Medium
Output: {"question": "What did the new CEO do after being appointed?", "reasoning_type": "chain"}"""


def fmt_ctx(supporting):
    return "\n".join(s if isinstance(s, str) else f"[S{s[0]}] {s[1]}" for s in supporting)


def prompt_pathqg(item):
    """Our method: path + context."""
    events = item["events"]
    path_str = " → ".join(f'"{e["trigger"]}"' for e in events)
    final = events[-1]["trigger"]
    ctx = fmt_ctx(item.get("supporting_sentences", []))
    diff = item["difficulty"]
    return f"""{FEW_SHOT}

Now generate:
Difficulty: {diff}
Path: {path_str}
Context: {ctx}

- Answer is "{final}", do NOT mention it
- Output ONLY: {{"question": "...", "reasoning_type": "direct|chain|cross_sentence"}}"""


def prompt_direct_llm(item):
    """Direct LLM: context only."""
    ctx = fmt_ctx(item.get("supporting_sentences", []))
    diff = item["difficulty"]
    return f"""{GENERIC_FEW_SHOT}

Now generate:
Context: {ctx}
Difficulty: {diff}

Output ONLY: {{"question": "...", "reasoning_type": "direct|chain|cross_sentence"}}"""


def prompt_path_only(item):
    """Path only, no context."""
    events = item["events"]
    path_str = " → ".join(f'"{e["trigger"]}"' for e in events)
    final = events[-1]["trigger"]
    diff = item["difficulty"]
    return f"""Example:
Path: "attack" → "destroy"
Difficulty: Easy
Output: {{"question": "What happened to the city after the army attacked it?", "reasoning_type": "direct"}}

Now generate:
Path: {path_str}
Difficulty: {diff}
- Answer is "{final}", do NOT mention it
- Output ONLY: {{"question": "...", "reasoning_type": "direct|chain|cross_sentence"}}"""


def prompt_relation_type(item):
    """Relation types + context."""
    rel_types = item.get("relation_subtypes", [])
    rel_cats = set()
    for rt in rel_types:
        if rt.startswith("CAUSE"): rel_cats.add("causal")
        elif rt.startswith("TEMPORAL"): rel_cats.add("temporal")
        elif rt.startswith("SUBEVENT"): rel_cats.add("subevent")
    rel_str = ", ".join(sorted(rel_cats)) if rel_cats else "unknown"
    ctx = fmt_ctx(item.get("supporting_sentences", []))
    diff = item["difficulty"]
    return f"""{GENERIC_FEW_SHOT}

Now generate a question connected through {rel_str} relations:
Context: {ctx}
Difficulty: {diff}

Output ONLY: {{"question": "...", "reasoning_type": "direct|chain|cross_sentence"}}"""


# ── Generator ───────────────────────────────────────────────
def generate_one(prompt, temperature=0.1):
    """Single API call, return (json_dict_or_None, raw_text)."""
    import urllib.request
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {SILICONFLOW_API_KEY}"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Output ONLY a valid JSON object."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
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
            text = data["choices"][0]["message"]["content"]
    except Exception as e:
        return None, f"ERROR: {e}"

    # Parse JSON
    import re as _re
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


# ── Repair prompts ───────────────────────────────────────────
def _build_repair_prompt(method_name, item, failed_question, failure_reason):
    """Build a repair prompt targeting the specific failure."""
    diff = item["difficulty"]
    gold = item.get("answer_trigger", "")
    events = item.get("events", [])

    # Build method-specific constraint block
    if method_name == "PathQG":
        path_str = " → ".join(f'"{e["trigger"]}"' for e in events)
        ctx = fmt_ctx(item.get("supporting_sentences", []))
        constraint = f"""Path: {path_str}
Context: {ctx}
Difficulty: {diff}
- Answer is "{gold}", do NOT mention it in the question"""
    elif method_name == "PathOnlyQG":
        path_str = " → ".join(f'"{e["trigger"]}"' for e in events)
        constraint = f"""Path: {path_str}
Difficulty: {diff}
- Answer is "{gold}", do NOT mention it in the question"""
    elif method_name == "RelationTypeQG":
        rel_types = item.get("relation_subtypes", [])
        rel_cats = set()
        for rt in rel_types:
            if rt.startswith("CAUSE"): rel_cats.add("causal")
            elif rt.startswith("TEMPORAL"): rel_cats.add("temporal")
            elif rt.startswith("SUBEVENT"): rel_cats.add("subevent")
        rel_str = ", ".join(sorted(rel_cats)) if rel_cats else "unknown"
        ctx = fmt_ctx(item.get("supporting_sentences", []))
        constraint = f"""Context: {ctx}
Relations: {rel_str}
Difficulty: {diff}"""
    else:  # DirectLLM
        ctx = fmt_ctx(item.get("supporting_sentences", []))
        constraint = f"""Context: {ctx}
Difficulty: {diff}"""

    # Map failure reason to specific fix instruction
    fix_map = {
        "no question mark": "End the question with ?",
        "bad start": "Start with What/Who/When/Where/Why/How/Did/Was/Were/Is/Are",
        "word repetition": "Remove repeated words, write grammatically",
        "trigger leakage": f'Do NOT use the word "{gold}" or its synonyms in the question',
        "empty": "Write a complete question",
        "parse error": "Output ONLY valid JSON, no extra text",
        "too short": "Write a longer, more complete question",
        "excessive repetition": "Remove repeated words, write naturally",
        "looping trigram": "Write naturally, avoid repeating phrase patterns",
    }
    fix = fix_map.get(failure_reason, "Fix the grammar and formatting")

    repair = f"""Your previous output was rejected.
Rejected: "{failed_question}"
Issue: {failure_reason}
→ {fix}

Generate a corrected question:
{constraint}
- Question must start with a question word and end with ?
- Output ONLY: {{"question": "...", "reasoning_type": "direct|chain|cross_sentence"}}"""
    return repair


# ── Retry generator ──────────────────────────────────────────
REPAIRABLE_REASONS = {
    "no question mark", "bad start", "word repetition", "trigger leakage",
    "empty", "parse error", "too short", "excessive repetition", "looping trigram",
    "not a dict", "no common English words",
}


def generate_with_retry(method_name, prompt_fn, item, max_attempts=3):
    """
    Generate with fair retry/repair for a single item.
    All methods use identical retry budget.
    Returns (result_dict, num_attempts).
    """
    gold = item.get("answer_trigger", "")
    question = ""
    rt = "error"
    g_ok, g_reason = False, "not attempted"
    attempts = 0

    for attempt in range(max_attempts):
        attempts = attempt + 1

        if attempt == 0:
            prompt = prompt_fn(item)
            temp = 0.1
        else:
            prompt = _build_repair_prompt(method_name, item, question, g_reason)
            temp = 0.1 + attempt * 0.1  # slightly increase temperature on retry

        gen, raw = generate_one(prompt, temperature=temp)

        if gen is None:
            # Parse error — may be repairable
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

        # Trigger leakage check
        if g_ok and gold and gold.lower() in question.lower():
            g_ok, g_reason = False, "trigger leakage"

        if g_ok:
            break

        # If not repairable, stop retrying
        if g_reason not in REPAIRABLE_REASONS:
            break

    result = {
        "item_id": item.get("_item_id", 0),
        "doc_id": item.get("doc_id", ""),
        "difficulty": item["difficulty"],
        "method": method_name,
        "generated_question": question,
        "gold_answer_trigger": gold,
        "reasoning_type": rt,
        "grammar_pass": g_ok,
        "grammar_reason": g_reason,
        "retry_attempts": attempts,
        "events": item.get("events", []),
        "supporting_sentences": item.get("supporting_sentences", []),
        "relation_subtypes": item.get("relation_subtypes", []),
        "difficulty_score": item.get("difficulty_score", 0),
    }
    return result, attempts


def generate_all(items, output_dir, methods=None, use_retry=True):
    """
    Generate questions for all items using all methods.
    Fair retry/repair for all methods when use_retry=True.
    Saves to compare_{Method}_generated_retry.jsonl when retry, else compare_{Method}_generated.jsonl.
    """
    if methods is None:
        methods = {
            "PathQG": prompt_pathqg,
            "DirectLLM": prompt_direct_llm,
            "PathOnlyQG": prompt_path_only,
            "RelationTypeQG": prompt_relation_type,
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    suffix = "_retry" if use_retry else ""

    for method_name, prompt_fn in methods.items():
        out_path = output_dir / f"compare_{method_name}_generated{suffix}.jsonl"
        if out_path.exists():
            out_path.unlink()

        print(f"\n{'='*50}")
        print(f"Generating: {method_name} {'(with retry)' if use_retry else ''}")
        print(f"{'='*50}")
        print(f"Output: {out_path.name}")

        method_results = []
        raw_pass = 0
        total_attempts = 0

        with open(out_path, "w", encoding="utf-8") as out_f:
            for i, item in enumerate(items):
                item["_item_id"] = i  # tag for tracking

                if use_retry:
                    r, attempts = generate_with_retry(method_name, prompt_fn, item, max_attempts=3)
                    total_attempts += attempts
                    if attempts == 1 and r["grammar_pass"]:
                        raw_pass += 1
                else:
                    prompt = prompt_fn(item)
                    gen, raw = generate_one(prompt)
                    question = gen.get("question", "") if isinstance(gen, dict) else ""
                    rt = gen.get("reasoning_type", "unknown") if isinstance(gen, dict) else "error"
                    gold = item.get("answer_trigger", "")
                    g_ok, g_reason = grammar_filter(question)
                    if g_ok and gold and gold.lower() in question.lower():
                        g_ok, g_reason = False, "trigger leakage"
                    if gen is None:
                        g_ok, g_reason = False, "parse error"
                    r = {
                        "item_id": i,
                        "doc_id": item.get("doc_id", ""),
                        "difficulty": item["difficulty"],
                        "method": method_name,
                        "generated_question": question,
                        "gold_answer_trigger": gold,
                        "reasoning_type": rt,
                        "grammar_pass": g_ok,
                        "grammar_reason": g_reason,
                        "retry_attempts": 1,
                        "events": item.get("events", []),
                        "supporting_sentences": item.get("supporting_sentences", []),
                        "relation_subtypes": item.get("relation_subtypes", []),
                        "difficulty_score": item.get("difficulty_score", 0),
                    }
                    attempts = 1

                method_results.append(r)
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                out_f.flush()

                if (i + 1) % 50 == 0:
                    n_pass = sum(1 for r in method_results if r["grammar_pass"])
                    print(f"  [{i+1}/{len(items)}] pass={n_pass} ({n_pass/(i+1)*100:.0f}%)", flush=True)

                time.sleep(0.1)

        n_pass = sum(1 for r in method_results if r["grammar_pass"])
        print(f"[{method_name}] Generated {len(method_results)}, grammar-passed: {n_pass} ({n_pass/len(method_results)*100:.0f}%)")
        if use_retry:
            print(f"  Raw pass (1st attempt): {raw_pass} ({raw_pass/len(method_results)*100:.0f}%)")
            print(f"  Retry pass (2nd/3rd):   {n_pass - raw_pass}")
            print(f"  Avg attempts: {total_attempts/len(method_results):.2f}")
        results[method_name] = method_results

    return results


# ── Main comparison ─────────────────────────────────────────
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

    # ── Step 1: Sample items ONCE with fixed seed ─────────────
    print("=== Step 1: Sampling items (seed=42) ===")
    with open(args.results_file, encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f]

    # Only sample from items that passed our original filter
    valid = [r for r in all_items if r.get("filter_pass", False)]

    by_level = defaultdict(list)
    for item in valid:
        by_level[item["difficulty"]].append(item)

    random.seed(42)
    sampled = []
    for level in ["Easy", "Medium", "Hard"]:
        pool = by_level[level]
        n = min(args.n_per_level, len(pool))
        sampled.extend(random.sample(pool, n))
    random.shuffle(sampled)

    print(f"Sampled {len(sampled)} items ({args.n_per_level}/level)")
    level_counts = Counter(r["difficulty"] for r in sampled)
    print(f"  By level: {dict(level_counts)}")

    # ── Step 2: Generate all methods (with retry) ────────────
    use_retry = not getattr(args, 'no_retry', False)
    if not args.skip_generation:
        print(f"\n=== Step 2: Generating all methods (retry={'on' if use_retry else 'off'}) ===")
        gen_results = generate_all(sampled, output_dir, use_retry=use_retry)
    else:
        print("\n=== Step 2: Loading existing generations ===")
        gen_results = {}
        suffix = "_retry" if use_retry else ""
        for method in ["PathQG", "DirectLLM", "PathOnlyQG", "RelationTypeQG"]:
            path = output_dir / f"compare_{method}_generated{suffix}.jsonl"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    gen_results[method] = [json.loads(line) for line in f]
                print(f"  Loaded {len(gen_results[method])} {method} items")
            else:
                print(f"  WARNING: {path} not found")

    # ── Step 3: Generation pass rate report ──────────────────
    print("\n=== Step 3: Generation pass rate ===")
    print(f"{'Method':<18} {'Total':>6} {'RawPass':>8} {'RetryPass':>10} {'RawRate':>8} {'FinalRate':>10}")
    print("-" * 65)
    for method in ["PathQG", "DirectLLM", "PathOnlyQG", "RelationTypeQG"]:
        if method in gen_results:
            items = gen_results[method]
            n_pass = sum(1 for r in items if r["grammar_pass"])
            n_raw = sum(1 for r in items if r["grammar_pass"] and r.get("retry_attempts", 1) == 1)
            raw_rate = n_raw / len(items) * 100 if items else 0
            final_rate = n_pass / len(items) * 100 if items else 0
            print(f"{method:<18} {len(items):>6} {n_raw:>8} {n_pass:>10} {raw_rate:>7.0f}% {final_rate:>9.0f}%")

    # ── Step 4: Evaluate grammar-passed items ────────────────
    if not args.skip_evaluation:
        print("\n=== Step 4: Evaluating all methods (v2) ===")
        eval_results = {}
        suffix = "_retry" if use_retry else ""

        for method in ["PathQG", "DirectLLM", "PathOnlyQG", "RelationTypeQG"]:
            if method not in gen_results:
                continue
            passed = [r for r in gen_results[method] if r["grammar_pass"]]
            if not passed:
                print(f"\n{method}: no passed items, skipping")
                continue

            eval_path = output_dir / f"compare_{method}_evaluated{suffix}_v2.jsonl"
            if eval_path.exists():
                eval_path.unlink()

            print(f"\n{method}: evaluating {len(passed)} items...")
            scored = []
            with open(eval_path, "w", encoding="utf-8") as out_f:
                for i, r in enumerate(passed):
                    r = evaluate_item(r)
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    out_f.flush()
                    scored.append(r)

                    if (i + 1) % 30 == 0:
                        n = len(scored)
                        avg_com = sum(s["composite"] for s in scored) / n
                        print(f"  [{i+1}/{len(passed)}] composite={avg_com:.3f}", flush=True)
                    time.sleep(0.15)

            eval_results[method] = scored

        # Compute pass rates for effective metrics
        pass_rates = {}
        for method in gen_results:
            items = gen_results[method]
            pass_rates[method] = sum(1 for r in items if r["grammar_pass"]) / len(items) if items else 0

        # ── Step 5: Quality metrics report (with effective) ───
        print("\n" + "="*90)
        print("QUALITY METRICS (on grammar-passed items)")
        print("="*90)
        header = f"{'Method':<18} {'N':>5} {'Hit':>6} {'Answer':>6} {'SolCorr':>7} {'Support':>7} {'Flu':>5} {'Path':>5} {'Diff':>5} {'Comp':>6}"
        print(header)
        print("-" * 90)
        for method in ["PathQG", "DirectLLM", "PathOnlyQG", "RelationTypeQG"]:
            if method not in eval_results:
                continue
            scored = eval_results[method]
            n = len(scored)
            hit = sum(s["target_event_hit"] for s in scored) / n
            ans = sum(s["judge_answerable"] for s in scored) / n
            cor = sum(s["judge_solver_correct"] for s in scored) / n
            sup = sum(s["judge_support_covered"] for s in scored) / n
            flu = sum(s["quality_fluency"] for s in scored) / n
            pat = sum(s["quality_path_relevance"] for s in scored) / n
            dif = sum(s["quality_difficulty_alignment"] for s in scored) / n
            com = sum(s["composite"] for s in scored) / n
            print(f"{method:<18} {n:>5} {hit:>6.3f} {ans:>6.3f} {cor:>7.3f} {sup:>7.3f} {flu:>5.3f} {pat:>5.3f} {dif:>5.3f} {com:>6.3f}")

        # ── Effective metrics (pass_rate * quality) ───────────
        print("\n" + "="*90)
        print("EFFECTIVE METRICS (pass_rate * quality, accounts for generation yield)")
        print("="*90)
        header = f"{'Method':<18} {'PassRt':>7} {'EffHit':>7} {'EffCorr':>8} {'EffComp':>8}"
        print(header)
        print("-" * 55)
        for method in ["PathQG", "DirectLLM", "PathOnlyQG", "RelationTypeQG"]:
            if method not in eval_results:
                continue
            scored = eval_results[method]
            n = len(scored)
            pr = pass_rates[method]
            hit = sum(s["target_event_hit"] for s in scored) / n
            cor = sum(s["judge_solver_correct"] for s in scored) / n
            com = sum(s["composite"] for s in scored) / n
            eff_hit = pr * hit
            eff_cor = pr * cor
            eff_com = pr * com
            print(f"{method:<18} {pr:>7.3f} {eff_hit:>7.3f} {eff_cor:>8.3f} {eff_com:>8.3f}")

        # ── Step 6: Matched-set comparison ────────────────────
        print("\n=== Step 6: Matched-set comparison ===")
        all_passed_ids = None
        for method in ["PathQG", "DirectLLM", "PathOnlyQG", "RelationTypeQG"]:
            if method not in gen_results:
                continue
            passed_ids = {r["item_id"] for r in gen_results[method] if r["grammar_pass"]}
            if all_passed_ids is None:
                all_passed_ids = passed_ids
            else:
                all_passed_ids &= passed_ids

        if all_passed_ids:
            print(f"Items where ALL methods pass: {len(all_passed_ids)}")
        else:
            id_to_methods = defaultdict(set)
            for method in gen_results:
                for r in gen_results[method]:
                    if r["grammar_pass"]:
                        id_to_methods[r["item_id"]].add(method)
            matched_2plus = {iid for iid, methods in id_to_methods.items() if len(methods) >= 2}
            print(f"Items where >=2 methods pass: {len(matched_2plus)}")
            all_passed_ids = matched_2plus

        if all_passed_ids and eval_results:
            print(f"\n{'Method':<18} {'N':>5} {'Hit':>6} {'SolCorrect':>10} {'Support':>7} {'Comp':>6}")
            print("-" * 55)
            for method in ["PathQG", "DirectLLM", "PathOnlyQG", "RelationTypeQG"]:
                if method not in eval_results:
                    continue
                matched = [s for s in eval_results[method] if s["item_id"] in all_passed_ids]
                if not matched:
                    continue
                n = len(matched)
                hit = sum(s["target_event_hit"] for s in matched) / n
                cor = sum(s["judge_solver_correct"] for s in matched) / n
                sup = sum(s["judge_support_covered"] for s in matched) / n
                com = sum(s["composite"] for s in matched) / n
                print(f"{method:<18} {n:>5} {hit:>6.3f} {cor:>10.3f} {sup:>7.3f} {com:>6.3f}")

    print("\n=== Comparison Complete ===")


if __name__ == "__main__":
    main()
