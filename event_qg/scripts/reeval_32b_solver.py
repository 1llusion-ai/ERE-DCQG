"""
Re-evaluate PathQG-HardAware with 32B solver + sample 30 for manual review.
"""
import json
import time
import os
import random
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in open(env_path):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from evaluator_v2 import (
    _call_api, _detect_loop, _normalize, _fuzzy_match,
    target_event_hit, llm_judge_v2, quality_judge, grammar_filter
)

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-32B-Instruct")


def solve_32b(question, context):
    """Solve using 32B model instead of 7B."""
    ctx_lines = context.split("\n")
    if len(ctx_lines) > 8:
        context = "\n".join(ctx_lines[:8])
    prompt = f"""Context:
{context}

Question: {question}

Answer in 1-5 words only."""
    for _ in range(2):
        resp = _call_api(prompt,
                         system="Answer questions briefly. Output ONLY the answer.",
                         temperature=0.0, max_tokens=40, timeout=60,
                         model=JUDGE_MODEL)
        if resp:
            cleaned = _detect_loop(resp)
            if cleaned and len(cleaned) < 100 and len(cleaned.split()) <= 10:
                q_words = set(_normalize(question).split()) - {
                    'the', 'a', 'an', 'is', 'was', 'were', 'did', 'what', 'who',
                    'when', 'where', 'why', 'how', 'after', 'in', 'to', 'of'}
                a_words = set(_normalize(cleaned).split()) - {
                    'the', 'a', 'an', 'is', 'was', 'were', 'did', 'what', 'who',
                    'when', 'where', 'why', 'how', 'after', 'in', 'to', 'of'}
                if len(a_words) >= 1 and not (a_words and a_words == (a_words & q_words)):
                    return cleaned
        time.sleep(0.2)
    return resp if resp else ""


def path_faithfulness_judge(question, path_events, supporting_sentences, difficulty):
    """Same as hardaware, using 32B for judging."""
    path_str = " -> ".join(e["trigger"] for e in path_events)
    final_trigger = path_events[-1]["trigger"] if path_events else "?"

    ctx_lines = []
    for i, s in enumerate(supporting_sentences):
        if isinstance(s, (list, tuple)):
            ctx_lines.append(f"[S{s[0]}] {s[1]}")
        else:
            ctx_lines.append(f"[S{i}] {s}")
    ctx_text = "\n".join(ctx_lines[:8])

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

    need_ie, hops, can_single = 0.5, 0.67, 0.5
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


def evaluate_one(r):
    """Evaluate one item with 32B solver + v2 judges + faith judge."""
    q = r["generated_question"]
    ctx = "\n".join(
        s if isinstance(s, str) else s[1]
        for s in r.get("supporting_sentences", [])
    )
    gold = r["gold_answer_trigger"]
    path_events = r.get("events", [])
    diff = r["difficulty"]

    # 32B Solver
    solver_ans = solve_32b(q, ctx)
    r["solver_answer_32b"] = solver_ans

    # Target event hit
    hit_score, hit_method = target_event_hit(solver_ans, gold)
    r["target_event_hit"] = round(hit_score, 2)
    r["hit_method"] = hit_method

    # LLM judge v2
    answerable, solver_correct, support_covered = llm_judge_v2(q, ctx, gold, solver_ans)
    r["judge_answerable"] = round(answerable, 2)
    r["judge_solver_correct"] = round(solver_correct, 2)
    r["judge_support_covered"] = round(support_covered, 2)

    # Quality judge
    fluency, relevance, diff_align = quality_judge(q, path_events, diff)
    r["quality_fluency"] = round(fluency, 2)
    r["quality_path_relevance"] = round(relevance, 2)
    r["quality_difficulty_alignment"] = round(diff_align, 2)
    r["composite"] = round(
        0.25 * solver_correct + 0.20 * answerable + 0.15 * support_covered +
        0.15 * fluency + 0.10 * relevance + 0.15 * diff_align, 3)

    # Path faithfulness judge
    faith = path_faithfulness_judge(q, path_events, r.get("supporting_sentences", []), diff)
    r["faith_need_intermediate"] = faith["need_intermediate_events"]
    r["faith_evidence_hops"] = faith["evidence_hops_used"]
    r["faith_can_answer_single"] = faith["can_answer_single_sentence"]
    r["faith_hard_pass"] = faith["hard_pass"]
    r["faith_raw"] = faith["raw_judgment"]

    return r


def main():
    # Load generation file
    gen_path = Path("event_qg/outputs/compare_PathQG_generated_retry_hardaware.jsonl")
    with open(gen_path, encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f]

    passed = [r for r in all_items if r["grammar_pass"]]
    print(f"Grammar-passed: {len(passed)}/{len(all_items)}")

    # Pass rate by difficulty
    by_level = {"Easy": [], "Medium": [], "Hard": []}
    for r in all_items:
        by_level[r["difficulty"]].append(r)

    print("\n=== PASS RATE BY DIFFICULTY (transparent report) ===")
    for level in ["Easy", "Medium", "Hard"]:
        items_l = by_level[level]
        n_pass = sum(1 for r in items_l if r["grammar_pass"])
        print(f"  {level}: {n_pass}/{len(items_l)} ({n_pass/len(items_l)*100:.0f}%)")
    print(f"  Total: {len(passed)}/{len(all_items)} ({len(passed)/len(all_items)*100:.0f}%)")

    # Evaluate
    output_path = Path("event_qg/outputs/compare_PathQG_evaluated_hardaware_32bsolver.jsonl")
    if output_path.exists():
        output_path.unlink()

    print(f"\n=== Re-evaluating with 32B solver ({len(passed)} items) ===")
    scored = []
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, r in enumerate(passed):
            r = evaluate_one(r)
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out_f.flush()
            scored.append(r)

            if (i + 1) % 20 == 0:
                n = len(scored)
                avg_cor = sum(s["judge_solver_correct"] for s in scored) / n
                avg_com = sum(s["composite"] for s in scored) / n
                print(f"  [{i+1}/{len(passed)}] cor={avg_cor:.3f} comp={avg_com:.3f}", flush=True)

            time.sleep(0.15)

    # Results
    n = len(scored)
    print(f"\n{'='*70}")
    print(f"RESULTS: 32B Solver on PathQG-HardAware (n={n})")
    print(f"{'='*70}")

    print(f"\n{'Level':<10} {'N':>5} {'Hit':>6} {'SolCorr':>7} {'Ans':>6} {'Comp':>6} {'NeedIE':>6} {'Single':>6} {'Hops':>6}")
    print("-" * 72)
    for level in ["Easy", "Medium", "Hard"]:
        items_l = [s for s in scored if s["difficulty"] == level]
        if not items_l:
            continue
        nl = len(items_l)
        hit = sum(s["target_event_hit"] for s in items_l) / nl
        cor = sum(s["judge_solver_correct"] for s in items_l) / nl
        ans = sum(s["judge_answerable"] for s in items_l) / nl
        com = sum(s["composite"] for s in items_l) / nl
        need = sum(s["faith_need_intermediate"] for s in items_l) / nl
        single = sum(s["faith_can_answer_single"] for s in items_l) / nl
        hops = sum(s["faith_evidence_hops"] for s in items_l) / nl
        print(f"{level:<10} {nl:>5} {hit:>6.3f} {cor:>7.3f} {ans:>6.3f} {com:>6.3f} {need:>6.3f} {single:>6.3f} {hops:>6.3f}")

    # Monotonicity
    print(f"\n{'='*70}")
    print("MONOTONICITY CHECK")
    print(f"{'='*70}")
    e = sum(s["judge_solver_correct"] for s in scored if s["difficulty"] == "Easy")
    en = max(1, sum(1 for s in scored if s["difficulty"] == "Easy"))
    m = sum(s["judge_solver_correct"] for s in scored if s["difficulty"] == "Medium")
    mn = max(1, sum(1 for s in scored if s["difficulty"] == "Medium"))
    h = sum(s["judge_solver_correct"] for s in scored if s["difficulty"] == "Hard")
    hn = max(1, sum(1 for s in scored if s["difficulty"] == "Hard"))
    print(f"  solver_correct: Easy={e/en:.3f} Medium={m/mn:.3f} Hard={h/hn:.3f}")
    if e/en >= m/mn >= h/hn:
        print("  MONOTONICITY: PASS (Easy >= Medium >= Hard)")
    elif e/en >= h/hn:
        print("  MONOTONICITY: PARTIAL (Easy >= Hard)")
    else:
        print("  MONOTONICITY: FAIL")

    # === Sample 30 for manual review ===
    print(f"\n{'='*70}")
    print("MANUAL REVIEW SAMPLE (30 items: 10 per level)")
    print(f"{'='*70}")

    random.seed(123)
    review_items = []
    for level in ["Easy", "Medium", "Hard"]:
        level_items = [s for s in scored if s["difficulty"] == level]
        sample = random.sample(level_items, min(10, len(level_items)))
        review_items.extend(sample)

    review_path = Path("event_qg/outputs/hardaware_32b_manual_review_sample.jsonl")
    with open(review_path, "w", encoding="utf-8") as f:
        for r in review_items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Also print a readable version
    readable_path = Path("event_qg/outputs/hardaware_32b_manual_review.txt")
    with open(readable_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(review_items):
            events = [e["trigger"] for e in r.get("events", [])]
            path_str = " -> ".join(events)
            sents = r.get("supporting_sentences", [])
            # Find which sentences contain the gold trigger
            gold = r["gold_answer_trigger"]
            gold_sents = []
            for s in sents:
                if isinstance(s, (list, tuple)):
                    if gold.lower() in s[1].lower():
                        gold_sents.append(f"[S{s[0]}] {s[1][:200]}")

            f.write(f"--- Item {i+1} | {r['difficulty']} | id={r['item_id']} ---\n")
            f.write(f"Q: {r['generated_question']}\n")
            f.write(f"Gold: \"{gold}\"\n")
            f.write(f"Path: {path_str}\n")
            f.write(f"Solver(32B): \"{r['solver_answer_32b']}\"\n")
            f.write(f"  target_hit={r['target_event_hit']} solver_correct={r['judge_solver_correct']}\n")
            f.write(f"  answerable={r['judge_answerable']} support={r['judge_support_covered']}\n")
            f.write(f"  faith_need_ie={r['faith_need_intermediate']} single_sent={r['faith_can_answer_single']} hops={r['faith_evidence_hops']}\n")
            f.write(f"  composite={r['composite']}\n")
            f.write(f"Sentences with gold trigger \"{gold}\":\n")
            for gs in gold_sents:
                f.write(f"  {gs}\n")
            f.write(f"All sentences ({len(sents)}):\n")
            for s in sents[:8]:
                if isinstance(s, (list, tuple)):
                    f.write(f"  [S{s[0]}] {s[1][:250]}\n")
                else:
                    f.write(f"  {str(s)[:250]}\n")
            f.write("\n")

            # Annotation template
            f.write("MANUAL ANNOTATION:\n")
            f.write("  1. 是否可答? (Is the question answerable from context?)\n")
            f.write("     [ ] yes  [ ] partial  [ ] no\n")
            f.write("  2. 是否真的需要多跳? (Does it genuinely need multi-hop reasoning?)\n")
            f.write("     [ ] yes, needs 2+ sentences  [ ] borderline  [ ] no, single sentence enough\n")
            f.write("  3. 难度排序是否合理? (Is the difficulty label reasonable?)\n")
            f.write(f"     [ ] Easy is correct  [ ] should be Medium  [ ] should be Hard\n" if r['difficulty'] == 'Easy' else "")
            f.write(f"     [ ] Medium is correct  [ ] should be Easy  [ ] should be Hard\n" if r['difficulty'] == 'Medium' else "")
            f.write(f"     [ ] Hard is correct  [ ] should be Easy  [ ] should be Medium\n" if r['difficulty'] == 'Hard' else "")
            f.write(f"\n{'='*60}\n\n")

    print(f"\nManual review sample saved to:")
    print(f"  {review_path} ({len(review_items)} items)")
    print(f"  {readable_path} (readable format with annotation template)")


if __name__ == "__main__":
    main()
