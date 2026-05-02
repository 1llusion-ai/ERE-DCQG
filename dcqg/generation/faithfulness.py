"""Path-faithfulness judge for multi-hop reasoning verification.

- path_faithfulness_judge: LLM judge for intermediate-event dependency
- evaluate_item_with_faithfulness: evaluate one item with v2 + faithfulness
- evaluate_file_with_faithfulness: batch evaluate with faithfulness
"""
import json
import time
import random
from pathlib import Path

from dcqg.utils.api_client import call_api
from dcqg.utils.config import get_api_config
from dcqg.evaluation.judge import evaluate_item


def path_faithfulness_judge(question, path_events, supporting_sentences, difficulty):
    """Judge whether a question genuinely requires multi-hop reasoning across path events.
    Returns dict with:
        - need_intermediate_events: 0.0/0.5/1.0 (yes=1.0)
        - evidence_hops_used: 0.33/0.67/1.0 (1/2/3+)
        - can_answer_single_sentence: 0.0/0.5/1.0 (yes=1.0 = BAD for Hard)
        - hard_pass: bool (all 3 conditions met for Hard)
        - raw_judgment: string
    """
    judge_model = get_api_config()["JUDGE_MODEL"]
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

    resp = call_api(prompt, temperature=0.0, max_tokens=30, model=judge_model, timeout=120)

    need_ie = 0.5
    hops = 0.67
    can_single = 0.5
    raw = resp or "NO_RESPONSE"

    if resp:
        resp_upper = resp.upper().replace(",", " ")
        for part in resp_upper.split():
            if part.startswith("NEED="):
                need_ie = 1.0 if "YES" in part else (0.0 if "NO" in part else 0.5)
            elif part.startswith("EVIDENCE="):
                val = part.split("=", 1)[1].strip()
                if "3" in val and "+" in val:
                    hops = 1.0
                elif "2" in val:
                    hops = 0.67
                elif "1" in val:
                    hops = 0.33
            elif part.startswith("SINGLE="):
                can_single = 1.0 if "YES" in part else (0.0 if "NO" in part else 0.5)

    hard_pass = need_ie >= 0.5 and hops >= 0.67 and can_single <= 0.5

    return {
        "need_intermediate_events": round(need_ie, 2),
        "evidence_hops_used": round(hops, 2),
        "can_answer_single_sentence": round(can_single, 2),
        "hard_pass": hard_pass,
        "raw_judgment": raw,
    }


def evaluate_item_with_faithfulness(r, skip_quality=False):
    """Evaluate one item with v2 metrics + path faithfulness."""
    r = evaluate_item(r, skip_judge=skip_quality)

    q = r["generated_question"]
    path_events = r.get("events", [])
    sents = r.get("supporting_sentences", [])
    diff = r["difficulty"]

    faith = path_faithfulness_judge(q, path_events, sents, diff)
    r["faith_need_intermediate"] = faith["need_intermediate_events"]
    r["faith_evidence_hops"] = faith["evidence_hops_used"]
    r["faith_can_answer_single"] = faith["can_answer_single_sentence"]
    r["faith_hard_pass"] = faith["hard_pass"]
    r["faith_raw"] = faith["raw_judgment"]

    return r


def evaluate_file_with_faithfulness(input_path, output_path, max_items=None):
    """Evaluate all grammar-passed items with v2 + faith judge."""
    with open(input_path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    passed = [r for r in items if r.get("grammar_pass", False)]
    print(f"  {len(passed)}/{len(items)} grammar-passed")

    if max_items:
        random.seed(42)
        passed = random.sample(passed, min(max_items, len(passed)))

    if Path(output_path).exists():
        Path(output_path).unlink()

    scored = []
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, r in enumerate(passed):
            r = evaluate_item_with_faithfulness(r)
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out_f.flush()
            scored.append(r)

            if (i + 1) % 20 == 0:
                n = len(scored)
                avg_com = sum(s["composite"] for s in scored) / n
                avg_need = sum(s["faith_need_intermediate"] for s in scored) / n
                avg_single = sum(s["faith_can_answer_single"] for s in scored) / n
                hard_pass = sum(1 for s in scored if s.get("faith_hard_pass", False))
                print(f"  [{i+1}/{len(passed)}] comp={avg_com:.3f} need={avg_need:.2f} single={avg_single:.2f} hard_pass={hard_pass}", flush=True)

            time.sleep(0.15)

    return scored
