"""
Stage 2 prototype v3: few-shot examples + retry logic for reliable JSON output.
1. Sample 30 Easy/Medium/Hard from valid, filtering generic triggers
2. Generate with few-shot prompt, retry up to 3x (temp 0.1→0.25→0.4)
3. Strict filter: parse_ok, reasoning_type, ?, no trigger leakage
4. answer_trigger always from gold
5. Light solver/judge to verify Easy > Medium > Hard
"""
import json
import argparse
import random
import time
from pathlib import Path
from collections import defaultdict
import os

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

# Generic/vague trigger words to filter out at the source
GENERIC_TRIGGERS = {
    "occurred", "happened", "said", "stated", "mentioned", "reported",
    "made", "did", "took place", "took", "came", "went", "was", "were",
    "become", "became", "began", "begin", "ended", "end", "added", "showed",
    "announced", "revealed", "found", "found", "called", "named", "served",
    "continued", "included", "contained", "provided", "given", "seen", "looked"
}

# Trigger words too short or too generic as answer events
BAD_ANSWER_TRIGGERS = {
    "occurred", "said", "made", "came", "took place", "went", "was", "were",
    "found", "called", "began", "ended", "continued"
}


def is_generic_trigger(trigger):
    """Return True if trigger is too vague/generic to serve as a good answer event."""
    t = trigger.strip().lower()
    if not t or len(t) <= 2:
        return True
    if t in GENERIC_TRIGGERS:
        return True
    # Phrases
    if any(phrase in t for phrase in ["took place", "made a", "was a", "is a"]):
        return True
    return False


def sample_balanced(paths_jsonl, n_per_level=30, output_path=None):
    """Sample n_per_level from each difficulty, filtering generic triggers."""
    by_level = defaultdict(list)
    with open(paths_jsonl, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # Filter out generic triggers at source
            if is_generic_trigger(item.get("answer_trigger", "")):
                continue
            by_level[item["difficulty"]].append(item)

    sampled = []
    for level in ["Easy", "Medium", "Hard"]:
        pool = by_level.get(level, [])
        if len(pool) < n_per_level:
            print(f"WARNING: {level} has only {len(pool)} candidates after trigger filter")
        n = min(n_per_level, len(pool))
        sampled.extend(random.sample(pool, n))

    random.seed(42)
    random.shuffle(sampled)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            for item in sampled:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved balanced sample to {output_path}")

    return sampled


FEW_SHOT_EXAMPLES = """Example 1 (Easy):
Path: "attack" → "destroy"
Context: [S0] The army launched an attack on the city. [S1] The city was destroyed.
Difficulty: Easy
Output: {"question": "What happened to the city after the army attacked it?", "reasoning_type": "direct"}

Example 2 (Medium):
Path: "resign" → "appoint" → "implement"
Context: [S0] The CEO resigned last Monday. [S1] The board appointed a new leader. [S2] The new CEO implemented reforms.
Difficulty: Medium
Output: {"question": "What did the new CEO do after being appointed following the resignation?", "reasoning_type": "chain"}

Example 3 (Hard):
Path: "announce" → "protest" → "cancel"
Context: [S0] The government announced budget cuts. [S1] Citizens protested the decision. [S2] Officials canceled the policy. [S3] The announcement had been controversial.
Difficulty: Hard
Output: {"question": "What was the final outcome after citizens protested the government's budget announcement?", "reasoning_type": "cross_sentence"}"""


def build_qg_prompt(item):
    """Short prompt with few-shot examples for reliable JSON output."""
    events = item["events"]
    difficulty = item["difficulty"]
    supporting = item.get("supporting_sentences", [])

    # Format context
    if supporting:
        if isinstance(supporting[0], tuple):
            context_text = "\n".join(f"[S{sid}] {sent}" for sid, sent in supporting)
        else:
            context_text = "\n".join(f"[S{i}] {s}" for i, s in enumerate(supporting))
    else:
        context_text = "[No context]"

    # Path
    path_str = " → ".join(f'"{e["trigger"]}"' for e in events)
    final_trigger = events[-1]["trigger"] if events else "?"

    prompt = f"""{FEW_SHOT_EXAMPLES}

Now generate a question for this case:
Difficulty: {difficulty}
Path: {path_str}
Context: {context_text}

Requirements:
- The answer is the final event ("{final_trigger}"), do NOT mention it in the question
- End with "?"
- Output ONLY: {{"question": "...", "reasoning_type": "direct|chain|cross_sentence"}}"""
    return prompt


def parse_json_response(text):
    """Extract JSON from model response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    import re
    # Try markdown code blocks first
    for pattern in [r'```json\s*(\{.*?\})\s*```', r'```\s*(\{.*?\})\s*```']:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    # Try finding balanced braces
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        candidate = text[start:end]
        return json.loads(candidate)
    except (ValueError, json.JSONDecodeError):
        pass
    return None


def generate_with_api(prompt, max_retries=3):
    """Generate using SiliconFlow API with retry logic on JSON parse failure."""
    import urllib.request

    if not SILICONFLOW_API_KEY:
        return "ERROR: No API key"

    system_prompt = "Output ONLY a valid JSON object. Format: {\"question\": \"...?\", \"reasoning_type\": \"direct|chain|cross_sentence\"}"

    last_text = ""
    for attempt in range(max_retries):
        temperature = 0.1 + attempt * 0.15
        max_tokens = 200 + attempt * 80

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {SILICONFLOW_API_KEY}"}
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": ["\n\n", "```"],
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
                last_text = text

            # Check if result is valid JSON with a question
            result = parse_json_response(text)
            if result and isinstance(result.get("question"), str) and len(result.get("question", "")) > 5:
                return text

            # If parse fails but we have retries left, retry
            if attempt < max_retries - 1:
                time.sleep(0.3)

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.3)
            else:
                return f"ERROR: {e}"

    return last_text if last_text else "ERROR: All retries failed"


def check_trigger_leakage(question, gold_trigger):
    """Check if gold answer trigger appears in the question."""
    if not gold_trigger:
        return False
    q_lower = question.lower()
    t_lower = gold_trigger.lower()

    # Check full phrase first
    if t_lower in q_lower:
        return True

    # Check individual words (for partial matches)
    words = t_lower.split()
    for w in words:
        if len(w) < 3:
            continue
        if w in q_lower:
            return True
    return False


def strict_filter(item, gen, gold_trigger):
    """
    Strict filter: all must pass:
    - parse_ok: gen is dict with "question"
    - reasoning_type != "error"
    - question does not start with [PARSE ERROR]
    - question ends with ?
    - no gold trigger leakage
    """
    if not isinstance(gen, dict):
        return False, "not a dict"
    question = gen.get("question", "")
    reasoning_type = gen.get("reasoning_type", "")
    if not question:
        return False, "empty question"
    if question.startswith("[PARSE ERROR]") or question.startswith("[MOCK]"):
        return False, "parse error"
    if reasoning_type == "error":
        return False, "reasoning_type is error"
    if not question.strip().endswith("?"):
        return False, "no question mark"
    if check_trigger_leakage(question, gold_trigger):
        return False, "trigger leakage"
    return True, "pass"


def light_solver_judge(question, gold_trigger, context_text):
    """Check if context supports an answer containing gold trigger."""
    prompt = f"""Example:
Question: What happened to the city after the army attacked it?
Context: [S0] The army launched an attack on the city. [S1] The city was destroyed.
Answer phrase: "destroyed"
Output: {{"verdict": "yes", "reasoning": "S1 explicitly states the city was destroyed"}}

Now evaluate:
Question: {question}
Context: {context_text}

Does the context support an answer that includes the specific phrase "{gold_trigger}"?
Output ONLY: {{"verdict": "yes|no|partial", "reasoning": "..."}}"""
    resp = generate_with_api(prompt)
    if resp.startswith("ERROR"):
        return "error", resp

    result = parse_json_response(resp)
    if not isinstance(result, dict):
        return "error", f"Could not parse: {str(result)[:200]}"

    verdict = result.get("verdict", "error")
    return verdict, result.get("reasoning", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths_file", default="event_qg/outputs/sampled_paths_preview.jsonl")
    parser.add_argument("--n_per_level", type=int, default=30)
    parser.add_argument("--output_dir", default="event_qg/outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Sample balanced instances (with generic trigger filter)
    print("=== Step 1: Sampling balanced instances ===")
    sample_path = output_dir / "stage2_balanced_sample.jsonl"
    sampled = sample_balanced(args.paths_file, n_per_level=args.n_per_level, output_path=sample_path)
    print(f"Sampled {len(sampled)} instances (generic triggers removed)")

    from collections import Counter
    level_counts = Counter(item["difficulty"] for item in sampled)
    print(f"By level: {dict(level_counts)}")

    # Step 2: Generate
    print("\n=== Step 2: Generating questions ===", flush=True)
    results = []
    results_path = output_dir / "stage2_generation_results.jsonl"

    # Open file for incremental writing
    with open(results_path, "w", encoding="utf-8") as results_f:
        for i, item in enumerate(sampled):
            gold_trigger = item.get("answer_trigger", "")
            prompt = build_qg_prompt(item)

            gen_text = generate_with_api(prompt)
            gen = parse_json_response(gen_text) if not gen_text.startswith("ERROR") else None

            question = gen.get("question", "") if isinstance(gen, dict) else ""
            reasoning_type = gen.get("reasoning_type", "unknown") if isinstance(gen, dict) else "error"

            # Determine pass/fail
            if gen_text.startswith("ERROR:"):
                passed, reason = False, "api error"
            elif gen is None:
                passed, reason = False, "not a dict"
            else:
                passed, reason = strict_filter(item, gen, gold_trigger)

            result = {
                **item,
                "generated_question": question,
                "gold_answer_trigger": gold_trigger,
                "reasoning_type": reasoning_type,
                "raw_response": gen_text[:300] if not passed else "",
                "filter_pass": passed,
                "filter_reason": reason,
            }
            results.append(result)
            results_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            results_f.flush()

            if (i + 1) % 10 == 0:
                n_passed = sum(1 for r in results if r["filter_pass"])
                print(f"  [{i+1}/{len(sampled)}] passed={n_passed}", flush=True)

    # Step 3: Filter summary
    print("\n=== Step 3: Filtering ===", flush=True)
    passed = [r for r in results if r["filter_pass"]]
    print(f"Passed: {len(passed)} / {len(results)}", flush=True)

    by_level = defaultdict(list)
    for r in passed:
        by_level[r["difficulty"]].append(r)
    for level in ["Easy", "Medium", "Hard"]:
        print(f"  {level}: {len(by_level[level])}")

    # Filter reasons
    reasons = Counter(r.get("filter_reason", "unknown") for r in results)
    print(f"Filter reasons: {dict(reasons)}")

    # Step 4: Solver/judge
    print("\n=== Step 4: Light solver/judge ===")
    solver_results = []
    for r in passed[:min(len(passed), 60)]:
        q = r["generated_question"]
        at = r["gold_answer_trigger"]
        ctx = "\n".join(s if isinstance(s, str) else s[1] for s in r.get("supporting_sentences", []))
        verdict, reasoning = light_solver_judge(q, at, ctx)
        r["solver_verdict"] = verdict
        r["solver_reasoning"] = reasoning
        solver_results.append(r)
        time.sleep(0.3)

    verdict_by_level = {}
    for level in ["Easy", "Medium", "Hard"]:
        items = [r for r in solver_results if r["difficulty"] == level]
        dist = Counter(r.get("solver_verdict", "error") for r in items)
        verdict_by_level[level] = dict(dist)

    print("Solver verdict by difficulty:")
    for level, dist in verdict_by_level.items():
        print(f"  {level}: {dist}")

    # Save solver results
    solver_path = output_dir / "stage2_solver_results.jsonl"
    with open(solver_path, "w", encoding="utf-8") as f:
        for r in solver_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Step 5: Human review sample
    print("\n=== Step 5: Human review sample ===")
    review_items = []
    for level in ["Easy", "Medium", "Hard"]:
        level_items = [r for r in passed if r["difficulty"] == level]
        review_items.extend(level_items[:10])

    review_path = output_dir / "stage2_human_review_sample.json"
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(review_items, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(review_items)} items for human review to {review_path}")

    print("\n=== Stage 2 Prototype v2 Complete ===")
    print(f"Total: {len(results)}, Passed: {len(passed)}, Use rate: {len(passed)/len(results)*100:.0f}%")
    print(f"Solver: {verdict_by_level}")


if __name__ == "__main__":
    main()