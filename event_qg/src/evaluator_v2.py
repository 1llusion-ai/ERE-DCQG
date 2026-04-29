"""
Evaluator v2: improved evaluation with 3-way LLM judge.
Changes from v1:
- eval_answerability -> target_event_hit (text similarity)
- New LLM judge: answerable / solver_correct / support_covered
- Fluency + path_relevance + difficulty_alignment (32B judge)
- Fixed random seeds, no file append issues
"""
import json
import re
import time
import os
import random
from pathlib import Path
from collections import Counter

# ── Env ─────────────────────────────────────────────────────
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
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-32B-Instruct")

# ── API ─────────────────────────────────────────────────────
def _call_api(prompt, system="", temperature=0.0, max_tokens=80, model=None, timeout=90):
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


# ── Text normalization ──────────────────────────────────────
def _normalize(text):
    return re.sub(r'[^\w\s]', '', text.lower()).strip()


# ── Grammar filter ──────────────────────────────────────────
def grammar_filter(question):
    q = question.strip()
    if not q or not q.endswith("?"):
        return False, "no question mark"
    words = q.lower().split()
    if len(words) < 4:
        return False, "too short"
    for i in range(len(words) - 1):
        if words[i] == words[i+1] and len(words[i]) > 1:
            return False, f"word repetition: {words[i]}"
    for i in range(len(words) - 5):
        if words[i:i+3] == words[i+1:i+4] == words[i+2:i+5]:
            return False, "looping trigram"
    for word in set(words):
        if len(word) <= 2:
            continue
        if words.count(word) >= 5 and words.count(word) / len(words) > 0.4:
            return False, f"excessive repetition: {word}"
    q_starters = {"what", "who", "when", "where", "why", "how", "which", "whose",
                  "did", "was", "were", "is", "are", "do", "does", "had", "has", "have",
                  "can", "could", "would", "should", "will",
                  "after", "before", "during", "following"}
    if words[0] not in q_starters:
        return False, f"bad start: {words[0]}"
    return True, "pass"


# ── Solver ──────────────────────────────────────────────────
def _detect_loop(text):
    if re.search(r'\b(\w+)\b(\s+\1\b){3,}', text):
        return re.sub(r'(\b\w+\b)(\s+\1\b){3,}.*', r'\1', text, flags=re.IGNORECASE).strip()
    words = text.split()
    for i in range(len(words) - 5):
        if words[i:i+3] == words[i+3:i+6]:
            return ' '.join(words[:i+3])
    if re.search(r'(isis|onon|availableavailable|meansmeans|usedused)', text.lower()):
        return re.sub(r'\s*(isis|onon|availableavailable|meansmeans|usedused)\S*.*$', '', text, flags=re.IGNORECASE).strip()
    return text


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
        resp = _call_api(prompt,
                       system="Answer questions briefly. Output ONLY the answer.",
                       temperature=0.0, max_tokens=40, timeout=60)
        if resp:
            cleaned = _detect_loop(resp)
            if cleaned and len(cleaned) < 100 and len(cleaned.split()) <= 10:
                q_words = set(_normalize(question).split()) - {'the','a','an','is','was','were','did','what','who','when','where','why','how','after','in','to','of'}
                a_words = set(_normalize(cleaned).split()) - {'the','a','an','is','was','were','did','what','who','when','where','why','how','after','in','to','of'}
                if len(a_words) >= 1 and not (a_words and a_words == (a_words & q_words)):
                    return cleaned
        time.sleep(0.2)
    return resp if resp else ""


# ── Target event hit (renamed from answerability) ────────────
def _fuzzy_match(answer, trigger):
    a = _normalize(answer)
    t = _normalize(trigger)
    if not a or not t:
        return 'none'
    if t in a or a in t:
        return 'exact'
    a_words = {w for w in a.split() if len(w) > 2}
    t_words = {w for w in t.split() if len(w) > 2}
    if a_words & t_words:
        return 'fuzzy'
    for aw in a_words:
        for tw in t_words:
            if len(aw) >= 4 and len(tw) >= 4 and (aw.startswith(tw[:4]) or aw in tw or tw in aw):
                return 'stem'
    # Text similarity fallback
    a_tok = a.split()
    t_tok = t.split()
    m = len(a_tok)
    n = len(t_tok)
    dp = [[0]*(n+1) for _ in range(m+1)]
    mx = 0
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a_tok[i-1] == t_tok[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                mx = max(mx, dp[i][j])
    lcs = mx / max(m, n) if max(m, n) > 0 else 0
    jac = len(a_words & t_words) / len(a_words | t_words) if (a_words | t_words) else 0
    sim = 0.4*lcs + 0.6*jac
    return 'fuzzy' if sim > 0.5 else ('stem' if sim > 0.3 else 'none')


def target_event_hit(solver_answer, gold_trigger):
    """Score 0-1: how well solver answer matches gold trigger."""
    match = _fuzzy_match(solver_answer, gold_trigger)
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
    resp = _call_api(prompt, temperature=0.0, max_tokens=30, model=JUDGE_MODEL, timeout=120)

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
    path_str = " → ".join(e["trigger"] for e in path_events)

    prompt = f"""Question: "{question}"
Path: {path_str}
Difficulty: {difficulty}

Score 1-3:
1. Fluency: 3=natural 2=minor errors 1=broken
2. Path: 3=uses multiple events 2=uses two 1=single event
3. Difficulty: 3=perfect {difficulty} 2=close 1=wrong

Reply: F= P= D= (e.g., "F=3 P=2 D=3")"""
    resp = _call_api(prompt, temperature=0.0, max_tokens=30, model=JUDGE_MODEL, timeout=120)

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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="event_qg/outputs/stage2_generation_results.jsonl")
    parser.add_argument("--output", default="event_qg/outputs/stage2_evaluated_v2.jsonl")
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--skip_quality", action="store_true")
    args = parser.parse_args()
    evaluate_file(args.input, args.output, args.max_items, args.skip_quality)
