"""Compare baseline QG modes with and without target answers.

Runs Direct, ICL, and SelfRefine on the same story sections and target
difficulties. Each method is tested in two modes:

  - with_answer: the prompt receives the dataset target answer.
  - no_answer: the prompt receives only context and target difficulty, and the
    model chooses its own answer.

The script optionally judges each generated QA with the same quality and
difficulty judges used elsewhere in the FairytaleQA pipeline.

Usage:
  python -m scripts.run_qg_mode_pilot \
    --max_stories 5 \
    --output_dir outputs/runs/qg_mode_pilot_baselines_5_seed42/
"""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from dcqg.datasets.fairytaleqa_loader import load_fairytaleqa
from dcqg.generation.fairytale_qg import (
    difficulty_evidence_judge,
    generate_direct,
    generate_direct_no_answer,
    generate_icl,
    generate_icl_no_answer,
    generate_self_refine,
    generate_self_refine_no_answer,
    quality_judge,
)


DIFFICULTIES = ["Easy", "Medium", "Hard"]
METHODS = ["Direct", "ICL", "SelfRefine"]
MODES = ["with_answer", "no_answer"]


def _select_candidates(split, max_stories=None, seed=42):
    """Load a split, group by story, and select one QA pair per sampled story."""
    records = load_fairytaleqa(split=split)
    rng = random.Random(seed)

    by_story = defaultdict(list)
    for rec in records:
        by_story[rec["story_name"]].append(rec)

    stories = sorted(by_story.keys())
    rng.shuffle(stories)
    if max_stories:
        stories = stories[:max_stories]

    candidates = []
    for story in stories:
        picked = rng.choice(by_story[story])
        candidates.append({
            "story_name": story,
            "story_section": picked["story_section"],
            "answer1": picked["answer1"],
            "question": picked["question"],
            "attribute": picked.get("attribute", ""),
            "ex_or_im": picked.get("ex_or_im", ""),
        })
    return candidates


def _result_key(record):
    return "|".join([
        record.get("story_name", ""),
        record.get("target_difficulty", ""),
        record.get("method_family", ""),
        record.get("mode", ""),
    ])


def _load_existing(path):
    existing = {}
    if not path.exists():
        return existing
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            existing[_result_key(row)] = row
    return existing


def _write_results(path, results):
    rows = sorted(
        results.values(),
        key=lambda r: (
            r.get("story_name", ""),
            DIFFICULTIES.index(r.get("target_difficulty", "Easy"))
            if r.get("target_difficulty") in DIFFICULTIES else 99,
            METHODS.index(r.get("method_family", "Direct"))
            if r.get("method_family") in METHODS else 99,
            MODES.index(r.get("mode", "with_answer"))
            if r.get("mode") in MODES else 99,
        ),
    )
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_generation(candidate, difficulty, method, mode, retries=1):
    """Run one generation call."""
    section = candidate["story_section"]
    target_answer = candidate["answer1"]

    if method == "Direct" and mode == "with_answer":
        return generate_direct(section, target_answer, difficulty, max_retries=retries)
    if method == "Direct" and mode == "no_answer":
        return generate_direct_no_answer(section, difficulty, max_retries=retries)
    if method == "ICL" and mode == "with_answer":
        return generate_icl(section, target_answer, difficulty, max_retries=retries)
    if method == "ICL" and mode == "no_answer":
        return generate_icl_no_answer(section, difficulty, max_retries=retries)
    if method == "SelfRefine" and mode == "with_answer":
        return generate_self_refine(section, target_answer, difficulty, max_retries=retries)
    if method == "SelfRefine" and mode == "no_answer":
        return generate_self_refine_no_answer(section, difficulty, max_retries=retries)
    raise ValueError(f"Unknown method/mode: {method}/{mode}")


def _judge(record, candidate, skip_judges=False, skip_quality_judge=False):
    if skip_judges or not record.get("parse_ok"):
        return record

    question = record.get("generated_question", "")
    section = candidate["story_section"]
    difficulty = record["target_difficulty"]
    eval_answer = record.get("generated_answer") or record.get("target_answer") or ""

    if not skip_quality_judge:
        qj = quality_judge(question, section, eval_answer, difficulty)
        record["quality_judge"] = qj
        record["quality_pass"] = qj.get("quality_pass", False)
        record["strict_quality_pass"] = qj.get("strict_quality_pass", False)

    dj = difficulty_evidence_judge(question, section, eval_answer, difficulty)

    record["eval_answer"] = eval_answer
    record["difficulty_judge"] = dj
    record["predicted_difficulty"] = dj.get("predicted_difficulty", "judge_error")
    record["difficulty_match"] = record["predicted_difficulty"] == difficulty
    return record


def _run_one(candidate, difficulty, method, mode, retries=1,
             skip_judges=False, skip_quality_judge=False):
    section = candidate["story_section"]
    target_answer = candidate["answer1"]

    t0 = time.time()
    result, attempts = _run_generation(candidate, difficulty, method, mode, retries=retries)
    elapsed = time.time() - t0

    generated_answer = result.get("generated_answer", "")
    if mode == "with_answer":
        generated_answer = target_answer

    result.update({
        "story_name": candidate["story_name"],
        "target_difficulty": difficulty,
        "method_family": method,
        "mode": mode,
        "target_answer": target_answer if mode == "with_answer" else "",
        "dataset_target_answer": target_answer,
        "generated_answer": generated_answer,
        "attempts": attempts,
        "elapsed_sec": round(elapsed, 1),
        "original_question": candidate["question"],
        "attribute": candidate["attribute"],
        "ex_or_im": candidate["ex_or_im"],
    })
    return _judge(
        result,
        candidate,
        skip_judges=skip_judges,
        skip_quality_judge=skip_quality_judge,
    )


def _pct(num, den):
    return f"{100 * num / den:.1f}%" if den else "NA"


def _summarize(results):
    rows = list(results.values())
    summary = {}
    for method in METHODS:
        for mode in MODES:
            subset = [
                r for r in rows
                if r.get("method_family") == method and r.get("mode") == mode
            ]
            key = f"{method}/{mode}"
            summary[key] = {
                "total": len(subset),
                "parse_ok": sum(1 for r in subset if r.get("parse_ok")),
                "quality_judged": sum(1 for r in subset if "quality_pass" in r),
                "quality_pass": sum(1 for r in subset if r.get("quality_pass")),
                "strict_quality_pass": sum(1 for r in subset if r.get("strict_quality_pass")),
                "difficulty_judged": sum(1 for r in subset if "difficulty_match" in r),
                "difficulty_match": sum(1 for r in subset if r.get("difficulty_match")),
            }
    return summary


def _rate_cell(num, den):
    return f"{num}/{den} ({_pct(num, den)})"


def _optional_rate_cell(num, judged, total):
    if judged == 0:
        return "not run"
    return f"{num}/{judged} ({_pct(num, judged)}); judged {judged}/{total}"


def _write_report(path, candidates, results, args):
    rows = list(results.values())
    summary = _summarize(results)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Baseline QG Mode Pilot\n\n")
        f.write("Compares baseline methods with and without target answers.\n\n")
        f.write("## Settings\n\n")
        f.write(f"- Split: `{args.split}`\n")
        f.write(f"- Stories: {len(candidates)}\n")
        f.write(f"- Difficulties: {', '.join(DIFFICULTIES)}\n")
        f.write(f"- Methods: {', '.join(METHODS)}\n")
        f.write(f"- Modes: {', '.join(MODES)}\n")
        f.write(f"- Retries: {args.retries}\n")
        f.write(f"- Judges skipped: {args.skip_judges}\n\n")
        f.write(f"- Quality judge skipped: {args.skip_quality_judge}\n\n")

        f.write("## Aggregate By Method And Mode\n\n")
        f.write("| Method | Mode | Parse OK | Quality Pass | Strict Quality Pass | Difficulty Match |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for method in METHODS:
            for mode in MODES:
                s = summary[f"{method}/{mode}"]
                total = s["total"]
                f.write(
                    f"| {method} | {mode} | "
                    f"{_rate_cell(s['parse_ok'], total)} | "
                    f"{_optional_rate_cell(s['quality_pass'], s['quality_judged'], total)} | "
                    f"{_optional_rate_cell(s['strict_quality_pass'], s['quality_judged'], total)} | "
                    f"{_optional_rate_cell(s['difficulty_match'], s['difficulty_judged'], total)} |\n"
                )
        f.write("\n")

        if not args.skip_judges:
            f.write("## Difficulty Match By Target Difficulty\n\n")
            f.write("| Method | Mode | Target | Match | Pred Easy | Pred Medium | Pred Hard | Judge Error |\n")
            f.write("|---|---|---|---:|---:|---:|---:|---:|\n")
            for method in METHODS:
                for mode in MODES:
                    for diff in DIFFICULTIES:
                        subset = [
                            r for r in rows
                            if r.get("method_family") == method
                            and r.get("mode") == mode
                            and r.get("target_difficulty") == diff
                        ]
                        total = len(subset)
                        match = sum(1 for r in subset if r.get("difficulty_match"))
                        pred_easy = sum(1 for r in subset if r.get("predicted_difficulty") == "Easy")
                        pred_med = sum(1 for r in subset if r.get("predicted_difficulty") == "Medium")
                        pred_hard = sum(1 for r in subset if r.get("predicted_difficulty") == "Hard")
                        pred_err = total - pred_easy - pred_med - pred_hard
                        f.write(
                            f"| {method} | {mode} | {diff} | "
                            f"{match}/{total} ({_pct(match, total)}) | "
                            f"{pred_easy} | {pred_med} | {pred_hard} | {pred_err} |\n"
                        )
            f.write("\n")

        f.write("## Examples\n\n")
        for cand in candidates:
            f.write(f"### {cand['story_name']}\n\n")
            f.write(f"Original QA: Q: {cand['question']} | A: {cand['answer1']}\n\n")
            for diff in DIFFICULTIES:
                f.write(f"#### Target {diff}\n\n")
                for method in METHODS:
                    for mode in MODES:
                        match = [
                            r for r in rows
                            if r.get("story_name") == cand["story_name"]
                            and r.get("target_difficulty") == diff
                            and r.get("method_family") == method
                            and r.get("mode") == mode
                        ]
                        if not match:
                            continue
                        r = match[0]
                        label = f"{method}/{mode}"
                        if not r.get("parse_ok"):
                            f.write(f"- **{label}:** FAILED ({r.get('generation_error', '?')})\n")
                            continue
                        f.write(
                            f"- **{label}:** Q: {r.get('generated_question', '?')} "
                            f"| A: {r.get('generated_answer', '?')} "
                            f"| pred: {r.get('predicted_difficulty', 'not run')} "
                            f"| quality: {r.get('quality_pass', 'not run')}\n"
                        )
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Baseline QG mode pilot")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_stories", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="outputs/runs/qg_mode_pilot_v1/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--skip_judges", action="store_true")
    parser.add_argument("--skip_quality_judge", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    report_path = out_dir / "comparison.md"
    summary_path = out_dir / "summary.json"

    candidates = _select_candidates(args.split, max_stories=args.max_stories, seed=args.seed)
    print(f"Selected {len(candidates)} stories from {args.split}")

    results = {} if args.no_resume else _load_existing(results_path)
    total = len(candidates) * len(DIFFICULTIES) * len(METHODS) * len(MODES)
    done = len(results)

    for cand in candidates:
        for diff in DIFFICULTIES:
            for method in METHODS:
                for mode in MODES:
                    stub = {
                        "story_name": cand["story_name"],
                        "target_difficulty": diff,
                        "method_family": method,
                        "mode": mode,
                    }
                    key = _result_key(stub)
                    if key in results:
                        print(f"[skip] {cand['story_name']} | {diff} | {method} | {mode}")
                        continue

                    done += 1
                    print(f"[{done}/{total}] {cand['story_name']} | {diff} | {method} | {mode}")
                    try:
                        row = _run_one(
                            cand, diff, method, mode,
                            retries=args.retries,
                            skip_judges=args.skip_judges,
                            skip_quality_judge=args.skip_quality_judge,
                        )
                        results[key] = row
                        status = "OK" if row.get("parse_ok") else f"FAIL: {row.get('generation_error', '?')}"
                        pred = row.get("predicted_difficulty", "not_judged")
                        print(f"  -> {status}; pred={pred}; match={row.get('difficulty_match')}")
                    except Exception as exc:
                        row = {
                            **stub,
                            "parse_ok": False,
                            "error": str(exc),
                        }
                        results[key] = row
                        print(f"  -> ERROR: {exc}")
                    _write_results(results_path, results)

    _write_results(results_path, results)
    _write_report(report_path, candidates, results, args)
    summary = _summarize(results)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    for key, s in summary.items():
        total = s["total"]
        quality_cell = (
            "not run"
            if s["quality_judged"] == 0
            else f"{s['quality_pass']}/{s['quality_judged']}"
        )
        difficulty_cell = (
            "not run"
            if s["difficulty_judged"] == 0
            else f"{s['difficulty_match']}/{s['difficulty_judged']}"
        )
        print(
            f"{key}: parse={s['parse_ok']}/{total}, "
            f"quality={quality_cell}, "
            f"difficulty_match={difficulty_cell}"
        )
    print(f"\nResults: {results_path}")
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
