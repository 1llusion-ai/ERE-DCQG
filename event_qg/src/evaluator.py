"""
Evaluator v1: Solver + Judge for generated questions.
- Solver answers question from context (short answer, no JSON)
- Judge scores 4 dimensions: answerability, fluency, path relevance, difficulty alignment
- Uses exact/fuzzy match for answerability (fast), LLM for complex dimensions
- Grammar filter removes garbled questions before evaluation
"""
import json
import re
import time
import os
from pathlib import Path
from collections import Counter

# ── API config ──────────────────────────────────────────────
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
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-32B-Instruct")  # 32B for reliable judging


# ── API helper ──────────────────────────────────────────────
def _call_api(prompt, system="", temperature=0.1, max_tokens=150, model=None, timeout=90):
    """Simple API call, returns text or None on failure."""
    import urllib.request
    if not SILICONFLOW_API_KEY:
        return None

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {SILICONFLOW_API_KEY}"}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model or MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": ["\n\n"],
    }

    try:
        req = urllib.request.Request(
            SILICONFLOW_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


# ── Grammar filter ──────────────────────────────────────────
def grammar_filter(question):
    """
    Return (passed, reason) tuple.
    Filters out garbled, malformed, or incomprehensible questions.
    """
    q = question.strip()
    if not q:
        return False, "empty"

    # Must end with ?
    if not q.endswith("?"):
        return False, "no question mark"

    words = q.lower().split()
    if len(words) < 4:
        return False, "too short"

    # Repeated adjacent words (e.g., "the the", "not not")
    for i in range(len(words) - 1):
        if words[i] == words[i+1] and len(words[i]) > 1:
            return False, f"word repetition: {words[i]}"

    # Repeated 3-gram (model looping)
    for i in range(len(words) - 5):
        if words[i:i+3] == words[i+1:i+4] == words[i+2:i+5]:
            return False, "looping trigram"

    # Token repetition (model glitch: "on on on on")
    for word in set(words):
        if len(word) <= 2:
            continue
        count = words.count(word)
        if count >= 5 and count / len(words) > 0.4:
            return False, f"excessive repetition: {word}"

    # Garbled unicode (not in common languages)
    # Allow common Latin extensions for European languages
    weird = 0
    for ch in q:
        cp = ord(ch)
        if cp > 127:
            # Allow: Latin Extended, Greek, Cyrillic, common punctuation
            if not (0x0080 <= cp <= 0x024F or  # Latin Extended
                    0x0370 <= cp <= 0x03FF or  # Greek
                    0x0400 <= cp <= 0x04FF or  # Cyrillic
                    0x1E00 <= cp <= 0x1EFF or  # Latin Extended Additional
                    0x2000 <= cp <= 0x206F or  # General Punctuation
                    0x2010 <= cp <= 0x2027):   # Dashes, quotes
                weird += 1
    if weird > 5:
        return False, f"garbled chars: {weird}"

    # Must have at least one actual English word (basic check)
    common_words = {"the", "a", "an", "is", "was", "were", "did", "what", "who", "when",
                    "where", "why", "how", "after", "before", "during", "happened", "of", "in", "to"}
    has_english = any(w in common_words for w in words)
    if not has_english and len(words) < 8:
        return False, "no common English words"

    # Question must start with a question word or auxiliary verb
    first_word = words[0]
    q_starters = {"what", "who", "when", "where", "why", "how", "which", "whose",
                  "did", "was", "were", "is", "are", "do", "does", "had", "has", "have",
                  "can", "could", "would", "should", "will"}
    if first_word not in q_starters:
        return False, f"bad question start: {first_word}"

    return True, "pass"


# ── Simple text normalization ───────────────────────────────
def _normalize(text):
    """Lowercase, remove punctuation, strip."""
    return re.sub(r'[^\w\s]', '', text.lower()).strip()


def _fuzzy_match(answer, trigger):
    """
    Check if answer contains trigger or semantic equivalent.
    Returns: 'exact' | 'fuzzy' | 'stem' | 'none'
    """
    a = _normalize(answer)
    t = _normalize(trigger)

    if not a or not t:
        return 'none'

    # Exact: trigger is fully in answer, or answer is in trigger
    if t in a or a in t:
        return 'exact'

    # Word-level overlap
    a_words = set(a.split())
    t_words = set(t.split())
    a_words = {w for w in a_words if len(w) > 2}
    t_words = {w for w in t_words if len(w) > 2}

    if not a_words or not t_words:
        return 'none'

    # Direct word overlap
    if a_words & t_words:
        return 'fuzzy'

    # Stem matching: first 4 chars match
    for aw in a_words:
        aw_stem = aw[:4]
        for tw in t_words:
            tw_stem = tw[:4]
            if len(aw_stem) >= 3 and len(tw_stem) >= 3 and aw_stem == tw_stem:
                return 'stem'

    # Sub-word: one contains the other (3+ chars)
    for aw in a_words:
        for tw in t_words:
            if len(aw) >= 4 and len(tw) >= 4 and (aw in tw or tw in aw):
                return 'stem'

    return 'none'


# ── Anti-loop helper ────────────────────────────────────────
def _detect_loop(text):
    """Return cleaned text if looping detected, else original."""
    # Pattern 1: repeated word 4+ times consecutively
    if re.search(r'\b(\w+)\b(\s+\1\b){3,}', text):
        # Cut at first sign of repetition
        cleaned = re.sub(r'(\b\w+\b)(\s+\1\b){3,}.*', r'\1', text, flags=re.IGNORECASE)
        return cleaned.strip()

    # Pattern 2: same 3-gram repeated
    words = text.split()
    for i in range(len(words) - 5):
        if words[i:i+3] == words[i+3:i+6]:
            return ' '.join(words[:i+3])

    # Pattern 3: trailing garbage syllables ("isis", "onon", etc.)
    if re.search(r'(isis|onon|availableavailable|meansmeans|usedused)', text.lower()):
        cleaned = re.sub(r'\s*(isis|onon|availableavailable|meansmeans|usedused)\S*.*$', '', text, flags=re.IGNORECASE)
        return cleaned.strip()

    return text


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
            resp = _call_api(prompt,
                           system="You answer questions briefly. Output ONLY the answer phrase, nothing else.",
                           temperature=0.0,
                           max_tokens=40,
                           timeout=60)
            if resp:
                # Check for looping
                cleaned = _detect_loop(resp)
                if cleaned and len(cleaned) < 100 and len(cleaned.split()) <= 10:
                    # Check the answer isn't just repeating the question
                    q_words = set(_normalize(question).split())
                    a_words = set(_normalize(cleaned).split())
                    # Remove common stopwords from comparison
                    stopwords = {'the', 'a', 'an', 'is', 'was', 'were', 'did', 'do', 'does',
                                 'what', 'who', 'when', 'where', 'why', 'how', 'after', 'in', 'to', 'of'}
                    q_words -= stopwords
                    a_words -= stopwords
                    if len(a_words) >= 1 and not (a_words and a_words == (a_words & q_words)):
                        return cleaned
            time.sleep(0.2)

        return resp if resp else ""


# ── Text similarity for answerability ────────────────────────
def _text_similarity(a, b):
    """
    Compute text similarity using token overlap + longest common substring.
    Returns score 0-1. No API call needed.
    """
    a_tok = _normalize(a).split()
    b_tok = _normalize(b).split()
    if not a_tok or not b_tok:
        return 0.0

    # Token overlap (Jaccard)
    a_set = set(a_tok)
    b_set = set(b_tok)
    intersection = a_set & b_set
    union = a_set | b_set
    jaccard = len(intersection) / len(union) if union else 0

    # Longest common token subsequence
    m, n = len(a_tok), len(b_tok)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a_tok[i-1] == b_tok[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                max_len = max(max_len, dp[i][j])

    lcs_score = max_len / max(m, n) if max(m, n) > 0 else 0

    # Containment bonus
    a_str = _normalize(a)
    b_str = _normalize(b)
    containment = 1.0 if (a_str and b_str and (a_str in b_str or b_str in a_str)) else 0.0

    return 0.3 * jaccard + 0.4 * lcs_score + 0.3 * containment


# ── Judge ───────────────────────────────────────────────────
class Judge:
    """Score generated questions on 4 dimensions."""
    def __init__(self):
        self.judge_model = JUDGE_MODEL

    def score_answerability(self, solver_answer, gold_trigger):
        """
        Compare solver answer with gold trigger using text similarity.
        No LLM needed — objective, fast, consistent.
        """
        # Boost exact/fuzzy match for reliable detection
        match = _fuzzy_match(solver_answer, gold_trigger)
        if match == 'exact':
            return 1.0, match
        if match == 'fuzzy':
            return 0.8, match
        if match == 'stem':
            return 0.6, match

        # Text similarity fallback
        sim = _text_similarity(solver_answer, gold_trigger)
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

        resp = _call_api(prompt, temperature=0.0, max_tokens=40, model=self.judge_model or None, timeout=120)

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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", default="event_qg/outputs/stage2_generation_results.jsonl")
    parser.add_argument("--output_file", default="event_qg/outputs/stage2_evaluated.jsonl")
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    args = parser.parse_args()

    evaluate_all(args.results_file, args.output_file, args.max_items, args.start_idx)
