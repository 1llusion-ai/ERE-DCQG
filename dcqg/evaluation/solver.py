"""Solver + Judge for generated questions (from evaluator.py).

- Solver answers question from context (short answer, no JSON)
- Judge scores 4 dimensions: answerability, fluency, path relevance, difficulty alignment
- evaluate_all runs the full pipeline on a JSONL file
"""
import json
import re
import time
from pathlib import Path
from collections import Counter

from dcqg.utils.api_client import call_api
from dcqg.utils.text import normalize, fuzzy_match, text_similarity, detect_loop
from dcqg.question_filter.grammar import grammar_filter


# ── Solver ──────────────────────────────────────────────────
class Solver:
    """Answer a question given context."""

    def answer(self, question, context):
        """Return a short answer (1-5 words), with anti-loop protection."""
        # Truncate context to reduce confusion
        ctx_lines = context.split("\n")
        if len(ctx_lines) > 8:
            context = "\n".join(ctx_lines[:8])

        prompt = f"""Context:
{context}

Question: {question}

Answer the question in 1-5 words. Only output the answer. Do NOT explain."""

        for attempt in range(2):
            resp = call_api(prompt,
                            system="You answer questions briefly. Output ONLY the answer phrase, nothing else.",
                            temperature=0.0,
                            max_tokens=40,
                            timeout=60)
            if resp:
                # Check for looping
                cleaned = detect_loop(resp)
                if cleaned and len(cleaned) < 100 and len(cleaned.split()) <= 10:
                    # Check the answer isn't just repeating the question
                    q_words = set(normalize(question).split())
                    a_words = set(normalize(cleaned).split())
                    # Remove common stopwords from comparison
                    stopwords = {'the', 'a', 'an', 'is', 'was', 'were', 'did', 'do', 'does',
                                 'what', 'who', 'when', 'where', 'why', 'how', 'after', 'in', 'to', 'of'}
                    q_words -= stopwords
                    a_words -= stopwords
                    if len(a_words) >= 1 and not (a_words and a_words == (a_words & q_words)):
                        return cleaned
            time.sleep(0.2)

        return resp if resp else ""


# ── Judge ───────────────────────────────────────────────────
class Judge:
    """Score generated questions on 4 dimensions."""

    def __init__(self):
        from dcqg.utils.config import get_api_config
        self.judge_model = get_api_config()["JUDGE_MODEL"]

    def score_answerability(self, solver_answer, gold_trigger):
        """
        Compare solver answer with gold trigger using text similarity.
        No LLM needed -- objective, fast, consistent.
        """
        # Boost exact/fuzzy match for reliable detection
        match = fuzzy_match(solver_answer, gold_trigger)
        if match == 'exact':
            return 1.0, match
        if match == 'fuzzy':
            return 0.8, match
        if match == 'stem':
            return 0.6, match

        # Text similarity fallback
        sim = text_similarity(solver_answer, gold_trigger)
        if sim > 0.5:
            return sim, f'sim-{sim:.2f}'
        elif sim > 0.3:
            return sim, f'sim-{sim:.2f}'
        return sim, 'no-match'

    def score_all(self, question, solver_answer, gold_trigger, path_events, difficulty):
        """
        Judge fluency, path relevance, difficulty alignment.
        Uses 32B model for reliable judging with 1-3 scale.
        Returns (fluency, path_relevance, difficulty_alignment) all 0-1.
        """
        path_str = " → ".join(e["trigger"] for e in path_events)

        prompt = f"""Question: "{question}"
Event chain: {path_str}
Labeled difficulty: {difficulty}

Score each from 1 to 3:
1. Fluency: 3=natural, 2=minor errors, 1=broken
2. Path usage: 3=uses multiple events, 2=uses two events, 1=single event
3. Difficulty fit: 3=perfect match for {difficulty}, 2=close, 1=wrong level

Reply ONLY: F= P= D= (e.g., "F=3 P=2 D=3")"""

        resp = call_api(prompt, temperature=0.0, max_tokens=40,
                        model=self.judge_model, timeout=120)

        fluency, relevance, diff_align = 1/3, 1/3, 1/3  # defaults
        if resp:
            # Parse "F=3 P=2 D=3" format
            for part in resp.replace(',', ' ').split():
                part = part.strip().upper()
                if part.startswith('F=') or part.startswith('F '):
                    try:
                        fluency = int(re.findall(r'(\d)', part)[0]) / 3.0
                    except (ValueError, IndexError):
                        pass
                elif part.startswith('P=') or part.startswith('P '):
                    try:
                        relevance = int(re.findall(r'(\d)', part)[0]) / 3.0
                    except (ValueError, IndexError):
                        pass
                elif part.startswith('D=') or part.startswith('D '):
                    try:
                        diff_align = int(re.findall(r'(\d)', part)[0]) / 3.0
                    except (ValueError, IndexError):
                        pass

        return fluency, relevance, diff_align


# ── Full evaluation pipeline ────────────────────────────────
def evaluate_all(results_file, output_file, max_items=None, start_idx=0):
    """
    Run solver + judge on all passed questions.
    Saves scored results incrementally.
    """
    # Load passed questions
    with open(results_file, encoding="utf-8") as f:
        all_results = [json.loads(line) for line in f]

    passed = [r for r in all_results if r["filter_pass"]]

    # Apply grammar filter
    grammar_passed = []
    grammar_failed = []
    for r in passed:
        ok, reason = grammar_filter(r["generated_question"])
        if ok:
            grammar_passed.append(r)
        else:
            r["grammar_fail_reason"] = reason
            grammar_failed.append(r)

    print(f"Grammar filter: {len(grammar_passed)} passed, {len(grammar_failed)} removed")
    for reason, count in Counter(r.get("grammar_fail_reason", "?") for r in grammar_failed).most_common(10):
        print(f"  {reason}: {count}")

    to_eval = grammar_passed
    if max_items:
        to_eval = to_eval[start_idx:start_idx + max_items]

    print(f"Evaluating {len(to_eval)} questions...")

    solver = Solver()
    judge = Judge()

    scored = []
    stats = {
        "answerability": [],
        "fluency": [],
        "path_relevance": [],
        "difficulty_alignment": [],
    }

    # Load existing progress
    existing = set()
    if Path(output_file).exists():
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                try:
                    existing.add(json.loads(line).get("doc_id", ""))
                except json.JSONDecodeError:
                    pass

    with open(output_file, "a", encoding="utf-8") as out_f:
        for i, r in enumerate(to_eval):
            q = r["generated_question"]
            ctx = "\n".join(
                s if isinstance(s, str) else s[1]
                for s in r.get("supporting_sentences", [])
            )
            gold_trigger = r["gold_answer_trigger"]
            path_events = r.get("events", [])
            difficulty = r["difficulty"]

            # Solver
            solver_answer = solver.answer(q, ctx)

            # Judge: answerability (text similarity, no LLM)
            ans_score, ans_method = judge.score_answerability(solver_answer, gold_trigger)

            # Judge: fluency, relevance, difficulty
            fluency, relevance, diff_align = judge.score_all(
                q, solver_answer, gold_trigger, path_events, difficulty
            )

            # Composite score (weighted)
            composite = (
                0.35 * ans_score +
                0.25 * fluency +
                0.20 * relevance +
                0.20 * diff_align
            )

            r["solver_answer"] = solver_answer
            r["eval_answerability"] = round(ans_score, 2)
            r["eval_answer_method"] = ans_method
            r["eval_fluency"] = round(fluency, 2)
            r["eval_path_relevance"] = round(relevance, 2)
            r["eval_difficulty_alignment"] = round(diff_align, 2)
            r["eval_composite"] = round(composite, 3)

            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out_f.flush()
            scored.append(r)

            stats["answerability"].append(ans_score)
            stats["fluency"].append(fluency)
            stats["path_relevance"].append(relevance)
            stats["difficulty_alignment"].append(diff_align)

            if (i + 1) % 20 == 0:
                avg_ans = sum(stats["answerability"]) / len(stats["answerability"])
                avg_flu = sum(stats["fluency"]) / len(stats["fluency"])
                avg_rel = sum(stats["path_relevance"]) / len(stats["path_relevance"])
                avg_dif = sum(stats["difficulty_alignment"]) / len(stats["difficulty_alignment"])
                avg_com = sum(r["eval_composite"] for r in scored) / len(scored)
                print(f"  [{i+1}/{len(to_eval)}] ans={avg_ans:.2f} flu={avg_flu:.2f} rel={avg_rel:.2f} dif={avg_dif:.2f} | composite={avg_com:.2f}")

            time.sleep(0.15)

    # Summary
    print(f"\n=== Evaluation Complete: {len(scored)} questions ===")
    avg_ans = sum(stats["answerability"]) / len(stats["answerability"]) if stats["answerability"] else 0
    avg_flu = sum(stats["fluency"]) / len(stats["fluency"]) if stats["fluency"] else 0
    avg_rel = sum(stats["path_relevance"]) / len(stats["path_relevance"]) if stats["path_relevance"] else 0
    avg_dif = sum(stats["difficulty_alignment"]) / len(stats["difficulty_alignment"]) if stats["difficulty_alignment"] else 0
    avg_com = sum(r["eval_composite"] for r in scored) / len(scored) if scored else 0

    print(f"Answerability:  {avg_ans:.3f}")
    print(f"Fluency:        {avg_flu:.3f}")
    print(f"Path Relevance: {avg_rel:.3f}")
    print(f"Difficulty Align: {avg_dif:.3f}")
    print(f"Composite:      {avg_com:.3f}")

    # By difficulty level
    by_level = {"Easy": [], "Medium": [], "Hard": []}
    for r in scored:
        by_level[r["difficulty"]].append(r["eval_composite"])
    for level, scores in by_level.items():
        if scores:
            print(f"  {level}: avg composite = {sum(scores)/len(scores):.3f} (n={len(scores)})")

    return scored
