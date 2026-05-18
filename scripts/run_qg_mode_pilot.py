"""Pilot: compare QG modes (with/without target answer) on validation stories.

Tests 2 modes × 3 difficulties on val split stories using Direct method.
Section-only context.

Usage:
  python -m scripts.run_qg_mode_pilot \
    --max_stories 5 \
    --output_dir outputs/runs/qg_mode_pilot_v1/
"""
import argparse
import json
import random
import time
from pathlib import Path

from dcqg.datasets.fairytaleqa_loader import load_fairytaleqa
from dcqg.generation.fairytale_qg import generate_direct, generate_direct_no_answer


DIFFICULTIES = ["Easy", "Medium", "Hard"]


def _select_val_candidates(max_stories=None, seed=42):
    """Load val split, group by story, select one QA pair per story."""
    records = load_fairytaleqa(split="validation")
    rng = random.Random(seed)

    # Group by story
    by_story = {}
    for rec in records:
        name = rec["story_name"]
        if name not in by_story:
            by_story[name] = []
        by_story[name].append(rec)

    stories = sorted(by_story.keys())
    if max_stories:
        stories = stories[:max_stories]

    candidates = []
    for story in stories:
        items = by_story[story]
        picked = rng.choice(items)
        candidates.append({
            "story_name": story,
            "story_section": picked["story_section"],
            "answer1": picked["answer1"],
            "question": picked["question"],
            "attribute": picked.get("attribute", ""),
            "ex_or_im": picked.get("ex_or_im", ""),
        })
    return candidates


def _run_one(candidate, difficulty, mode, retries=2):
    """Run one generation. mode: 'with_answer' or 'no_answer'."""
    section = candidate["story_section"]
    answer = candidate["answer1"]

    t0 = time.time()
    if mode == "with_answer":
        result, attempts = generate_direct(section, answer, difficulty, max_retries=retries)
    else:
        result, attempts = generate_direct_no_answer(section, difficulty, max_retries=retries)
    elapsed = time.time() - t0

    result["story_name"] = candidate["story_name"]
    result["target_difficulty"] = difficulty
    result["target_answer"] = answer if mode == "with_answer" else ""
    result["mode"] = mode
    result["attempts"] = attempts
    result["elapsed_sec"] = round(elapsed, 1)
    result["original_question"] = candidate["question"]
    result["attribute"] = candidate["attribute"]
    result["ex_or_im"] = candidate["ex_or_im"]
    return result


def main():
    parser = argparse.ArgumentParser(description="QG mode pilot")
    parser.add_argument("--max_stories", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="outputs/runs/qg_mode_pilot_v1/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = _select_val_candidates(max_stories=args.max_stories, seed=args.seed)
    print(f"Selected {len(candidates)} val stories")

    results = []
    total = len(candidates) * len(DIFFICULTIES) * 2  # 2 modes
    done = 0

    for cand in candidates:
        for diff in DIFFICULTIES:
            for mode in ["with_answer", "no_answer"]:
                done += 1
                print(f"[{done}/{total}] {cand['story_name']} | {diff} | {mode}")
                try:
                    r = _run_one(cand, diff, mode, retries=args.retries)
                    results.append(r)
                    status = "OK" if r["parse_ok"] else f"FAIL: {r.get('generation_error', '?')}"
                    print(f"  -> {status} ({r['elapsed_sec']}s)")
                except Exception as e:
                    print(f"  -> ERROR: {e}")
                    results.append({
                        "story_name": cand["story_name"],
                        "target_difficulty": diff,
                        "mode": mode,
                        "error": str(e),
                        "parse_ok": False,
                    })

    # Save results
    results_path = out_dir / "results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Generate readable comparison
    report_path = out_dir / "comparison.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# QG Mode Pilot: with_answer vs no_answer\n\n")
        f.write(f"Stories: {len(candidates)}, Difficulties: {DIFFICULTIES}\n\n")

        for cand in candidates:
            f.write(f"## Story: {cand['story_name']}\n\n")
            f.write(f"**Section:** {cand['story_section'][:200]}...\n\n")
            f.write(f"**Original QA:** Q: {cand['question']} | A: {cand['answer1']}\n\n")

            for diff in DIFFICULTIES:
                f.write(f"### Target: {diff}\n\n")
                for mode in ["with_answer", "no_answer"]:
                    label = "With Answer" if mode == "with_answer" else "No Answer"
                    match = [r for r in results
                             if r["story_name"] == cand["story_name"]
                             and r["target_difficulty"] == diff
                             and r["mode"] == mode]
                    if match:
                        r = match[0]
                        if r.get("parse_ok"):
                            f.write(f"**{label}:**\n")
                            f.write(f"- Q: {r.get('generated_question', '?')}\n")
                            f.write(f"- A: {r.get('generated_answer', r.get('target_answer', '?'))}\n")
                            f.write(f"- Reasoning: {r.get('reasoning_type', '?')}\n\n")
                        else:
                            f.write(f"**{label}:** FAILED ({r.get('generation_error', '?')})\n\n")
                    else:
                        f.write(f"**{label}:** NO RESULT\n\n")

    # Summary stats
    parse_ok = sum(1 for r in results if r.get("parse_ok"))
    print(f"\n=== Summary ===")
    print(f"Total: {total}, Parse OK: {parse_ok} ({100*parse_ok/total:.0f}%)")
    for mode in ["with_answer", "no_answer"]:
        mode_rs = [r for r in results if r.get("mode") == mode]
        ok = sum(1 for r in mode_rs if r.get("parse_ok"))
        print(f"  {mode}: {ok}/{len(mode_rs)}")
    print(f"\nResults: {results_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
