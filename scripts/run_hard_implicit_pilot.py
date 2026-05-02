"""Hard implicit chain pilot: test new implicit Hard prompt on 9 strict Hard paths.

Usage:
    python -m scripts.run_hard_implicit_pilot
    python -m scripts.run_hard_implicit_pilot --skip_judge
"""
import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from dcqg.utils.jsonl import read_jsonl, write_jsonl
from dcqg.utils.config import get_api_config
from dcqg.utils.api_client import call_openai_compatible
from dcqg.generation.generator import generate_with_retry_hardaware
from dcqg.question_filter.pipeline import quality_filter_pipeline
from dcqg.question_filter.hard_implicitness import count_explicit_prior_triggers

# Reuse judge functions from independent difficulty eval
from scripts.run_independent_difficulty_eval import (
    difficulty_judge, path_dependency_judge, parse_judge_json,
    build_difficulty_prompt, build_path_dependency_prompt,
)


INPUT_FILE = "outputs/runs/path_filter_strict_pilot/paths.filtered.strict.jsonl"
OLD_FILTERED = "outputs/runs/baseline_alignment_pilot/PathQG-HardAware_questions.filtered.jsonl"
OLD_JUDGED = "outputs/runs/independent_difficulty_eval_pilot/judged_all.jsonl"
OUTPUT_DIR = "outputs/runs/hard_implicit_qg_pilot"


def load_hard_paths():
    """Load Hard strict paths (9 items)."""
    items = read_jsonl(INPUT_FILE)
    hard = [x for x in items if x.get("difficulty") == "Hard"]
    for i, item in enumerate(hard):
        item["_item_id"] = i
    return hard


def load_old_hard_data():
    """Load old Hard questions and judge results for same-path comparison."""
    old_questions = {}  # doc_id -> list of items
    try:
        items = read_jsonl(OLD_FILTERED)
        for item in items:
            if item.get("difficulty") == "Hard" and item.get("final_filter_pass"):
                doc_id = item.get("doc_id", "")
                if doc_id not in old_questions:
                    old_questions[doc_id] = []
                old_questions[doc_id].append(item)
    except FileNotFoundError:
        pass

    old_judged = {}  # (doc_id, item_idx) -> judged item
    try:
        items = read_jsonl(OLD_JUDGED)
        for item in items:
            if item.get("method") == "PathQG-HardAware" and item.get("difficulty") == "Hard":
                doc_id = item.get("doc_id", "")
                key = (doc_id, item.get("_item_id", 0))
                old_judged[key] = item
    except FileNotFoundError:
        pass

    return old_questions, old_judged


def merge_context(result, source):
    """Copy metadata from source path to result if missing."""
    preserve_keys = [
        "title", "answer_event_id", "answer_trigger", "gold_answer_phrase",
        "gold_answer_sentence", "gold_event_type", "answer_phrase_status",
        "answer_phrase_pass", "answer_phrase_reason", "weak_trigger_type",
        "weak_trigger_pass", "weak_trigger_reason", "non_temporal_count",
        "relation_group", "support_span", "rule_single_sentence_risk",
        "prefilter_pass", "prefilter_reason", "llm_path_judge",
        "llm_path_judge_status", "llm_path_keep", "llm_path_keep_reason",
    ]
    for key in preserve_keys:
        if key in source and key not in result:
            result[key] = source[key]
    return result


def generate_implicit_hard(items):
    """Generate Hard questions with implicit chain prompt."""
    results = []
    for i, item in enumerate(items):
        doc_id = item.get("doc_id", "")[:12]
        triggers = [e.get("trigger", "?") for e in item.get("events", [])]
        path_str = " -> ".join(triggers)
        print(f"[{i+1}/{len(items)}] {doc_id} path={path_str}", end=" ")
        try:
            result, attempts = generate_with_retry_hardaware(item, max_attempts=5)
            result = merge_context(result, item)
            result["_item_id"] = i
            result["method"] = "PathQG-HardAware-Implicit"
            gen_ok = bool(result.get("generated_question")) and not result.get("generation_error")
            explicit = count_explicit_prior_triggers(
                result.get("generated_question", ""), item.get("events", [])
            )
            print(f"-> {'OK' if gen_ok else 'FAIL'} (attempts={attempts}, explicit_prior={explicit})")
        except Exception as exc:
            result = {
                "_item_id": i,
                "doc_id": item.get("doc_id", ""),
                "difficulty": "Hard",
                "method": "PathQG-HardAware-Implicit",
                "generated_question": "",
                "gold_answer_trigger": item.get("answer_trigger", ""),
                "generation_error": True,
                "generation_status": "exception",
                "generation_reason": f"{type(exc).__name__}: {exc}",
                "grammar_pass": False,
                "grammar_reason": "generation_exception",
                "events": item.get("events", []),
                "supporting_sentences": item.get("supporting_sentences", []),
                "relation_subtypes": item.get("relation_subtypes", []),
            }
            result = merge_context(result, item)
            print(f"-> EXCEPTION: {exc}")
        results.append(result)
        time.sleep(0.1)
    return results


def run_filter(results, skip_llm=False):
    """Run v3 quality filter + implicitness check on results."""
    filtered = []
    for r in results:
        if r.get("generation_error"):
            r.setdefault("final_filter_pass", False)
            r.setdefault("final_filter_reason", "generation_error")
            filtered.append(r)
            continue
        try:
            r = quality_filter_pipeline(r, skip_llm=skip_llm)
        except Exception as exc:
            r["final_filter_pass"] = False
            r["final_filter_reason"] = f"filter_exception: {type(exc).__name__}: {exc}"
        filtered.append(r)
    return filtered


def run_judge_on_items(items, model_config):
    """Run independent difficulty + path dependency judge on filter-passing items."""
    for i, item in enumerate(items):
        if not item.get("final_filter_pass"):
            item["difficulty_judge_status"] = "skipped"
            item["path_dependency_judge_status"] = "skipped"
            continue

        doc_id = item.get("doc_id", "")[:12]
        print(f"  [Judge {i+1}/{len(items)}] {doc_id}", end=" ")

        diff_result = difficulty_judge(item, model_config)
        item.update(diff_result)

        path_result = path_dependency_judge(item, model_config)
        item.update(path_result)

        pred = item["difficulty_judge"].get("predicted_difficulty", "?")
        dep = item["path_dependency_judge"].get("path_dependency", "?")
        d_status = item["difficulty_judge_status"]
        p_status = item["path_dependency_judge_status"]
        print(f"-> pred={pred}({d_status}) dep={dep}({p_status})")

        time.sleep(0.15)
    return items


def generate_report(results, old_questions, output_dir):
    """Generate HARD_IMPLICIT_REPORT.md with comparison tables."""
    lines = []
    lines.append("# Hard Implicit Chain Pilot Report\n")
    lines.append(f"**Input:** `{INPUT_FILE}` (9 Hard strict paths)")
    lines.append(f"**Method:** PathQG-HardAware with implicit_chain prompt")
    lines.append(f"**Filter:** v3 + hard_implicitness_check\n")

    total = len(results)
    gen_ok = sum(1 for r in results if r.get("generated_question") and not r.get("generation_error"))
    filter_pass = sum(1 for r in results if r.get("final_filter_pass"))
    implicit_pass = sum(1 for r in results if r.get("hard_implicit_chain_pass", True))
    implicit_fail = sum(1 for r in results if not r.get("hard_implicit_chain_pass", True))

    # Independent judge stats
    judged = [r for r in results if r.get("difficulty_judge_status") == "ok"
              and r.get("path_dependency_judge_status") == "ok"]
    pred_hard = sum(1 for r in judged if r["difficulty_judge"].get("predicted_difficulty") == "Hard")
    pred_medium = sum(1 for r in judged if r["difficulty_judge"].get("predicted_difficulty") == "Medium")
    pred_easy = sum(1 for r in judged if r["difficulty_judge"].get("predicted_difficulty") == "Easy")
    dep_strong = sum(1 for r in judged if r["path_dependency_judge"].get("path_dependency") == "strong")
    dep_partial = sum(1 for r in judged if r["path_dependency_judge"].get("path_dependency") == "partial")
    answerable = sum(1 for r in judged if r["difficulty_judge"].get("answerable") in ("yes", "partial"))
    fec = sum(1 for r in judged if r["difficulty_judge"].get("final_event_consistent") in ("yes", "partial"))

    lines.append("## Summary\n")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|------:|")
    lines.append(f"| Generated | {gen_ok}/{total} |")
    lines.append(f"| Filter Pass | {filter_pass}/{total} |")
    lines.append(f"| Implicit Chain Pass | {implicit_pass}/{total} |")
    lines.append(f"| Implicit Chain Fail | {implicit_fail}/{total} |")
    lines.append(f"| Independent Judged | {len(judged)}/{filter_pass} |")
    lines.append(f"| Pred Hard | {pred_hard}/{len(judged)} |")
    lines.append(f"| Pred Medium | {pred_medium}/{len(judged)} |")
    lines.append(f"| Pred Easy | {pred_easy}/{len(judged)} |")
    lines.append(f"| PathDep Strong | {dep_strong}/{len(judged)} |")
    lines.append(f"| PathDep Strong+Partial | {dep_strong + dep_partial}/{len(judged)} |")
    lines.append(f"| Answerable | {answerable}/{len(judged)} |")
    lines.append(f"| FinalConsistent | {fec}/{len(judged)} |")

    # Old vs New comparison
    lines.append("\n## Old vs New Comparison (Same Hard Paths)\n")
    lines.append("| Metric | Old Hard Prompt | New Implicit Hard |")
    lines.append("|--------|---:|---:|")

    # Old stats (from baseline alignment pilot)
    old_valid = []
    for doc_items in old_questions.values():
        old_valid.extend(doc_items)
    old_gen = len(old_valid) + (total - gen_ok)  # approximate
    old_filter = len(old_valid)

    # Old judge results
    old_judged_items = []
    for doc_items in old_questions.values():
        for item in doc_items:
            dj = item.get("difficulty_judge", {}) if "difficulty_judge" in item else {}
            if dj:
                old_judged_items.append(item)

    # Load old judge data properly
    old_pred_hard = 0
    old_pred_medium = 0
    old_pred_easy = 0
    old_dep_strong = 0
    old_answerable = 0
    old_fec = 0
    try:
        old_judged_all = read_jsonl(OLD_JUDGED)
        old_hard_judged = [x for x in old_judged_all
                          if x.get("method") == "PathQG-HardAware"
                          and x.get("difficulty") == "Hard"
                          and x.get("final_filter_pass")
                          and x.get("difficulty_judge_status") == "ok"]
        for x in old_hard_judged:
            pred = x.get("difficulty_judge", {}).get("predicted_difficulty", "?")
            if pred == "Hard": old_pred_hard += 1
            elif pred == "Medium": old_pred_medium += 1
            elif pred == "Easy": old_pred_easy += 1
            dep = x.get("path_dependency_judge", {}).get("path_dependency", "?")
            if dep == "strong": old_dep_strong += 1
            if x.get("difficulty_judge", {}).get("answerable") in ("yes", "partial"):
                old_answerable += 1
            if x.get("difficulty_judge", {}).get("final_event_consistent") in ("yes", "partial"):
                old_fec += 1
        old_judge_n = len(old_hard_judged)
    except FileNotFoundError:
        old_judge_n = 0

    lines.append(f"| Generated | {total} | {gen_ok} |")
    lines.append(f"| Filter Pass | {old_filter} | {filter_pass} |")
    lines.append(f"| Judged | {old_judge_n} | {len(judged)} |")
    lines.append(f"| Independent Pred Hard | {old_pred_hard} | {pred_hard} |")
    lines.append(f"| Independent Pred Medium | {old_pred_medium} | {pred_medium} |")
    lines.append(f"| Independent Pred Easy | {old_pred_easy} | {pred_easy} |")
    lines.append(f"| PathDep Strong | {old_dep_strong} | {dep_strong} |")
    lines.append(f"| Answerable | {old_answerable} | {answerable} |")
    lines.append(f"| FinalConsistent | {old_fec} | {fec} |")

    # Same-path comparison table
    lines.append("\n## Same-Path Comparison\n")
    lines.append("| # | Doc ID | Old Question | Old Pred | New Question | New Pred | New PathDep |")
    lines.append("|--:|--------|-------------|----------|-------------|----------|------------|")

    # Build lookup for new results by doc_id
    new_by_doc = {}
    for r in results:
        doc_id = r.get("doc_id", "")
        if doc_id not in new_by_doc:
            new_by_doc[doc_id] = []
        new_by_doc[doc_id].append(r)

    # Build lookup for old results
    old_by_doc = {}
    for doc_id, doc_items in old_questions.items():
        old_by_doc[doc_id] = doc_items

    # Load old judge results
    old_judge_lookup = {}
    try:
        old_judged_all = read_jsonl(OLD_JUDGED)
        for x in old_judged_all:
            if x.get("method") == "PathQG-HardAware" and x.get("difficulty") == "Hard":
                key = (x.get("doc_id", ""), x.get("_item_id", 0))
                old_judge_lookup[key] = x
    except FileNotFoundError:
        pass

    # Get all unique doc_ids from both old and new
    all_doc_ids = sorted(set(list(old_by_doc.keys()) + list(new_by_doc.keys())))

    row_num = 0
    for doc_id in all_doc_ids:
        old_items = old_by_doc.get(doc_id, [])
        new_items = new_by_doc.get(doc_id, [])

        # Show up to 1 old + 1 new per doc_id
        max_rows = max(len(old_items), len(new_items), 1)
        for idx in range(min(max_rows, 2)):
            row_num += 1
            old_q = old_items[idx]["generated_question"][:60] + "..." if idx < len(old_items) else "-"
            old_key = (doc_id, old_items[idx].get("_item_id", 0)) if idx < len(old_items) else None
            old_j = old_judge_lookup.get(old_key, {}) if old_key else {}
            old_pred = old_j.get("difficulty_judge", {}).get("predicted_difficulty", "?") if old_j else "-"

            new_q = new_items[idx]["generated_question"][:60] + "..." if idx < len(new_items) else "-"
            new_j = new_items[idx] if idx < len(new_items) else {}
            new_pred = new_j.get("difficulty_judge", {}).get("predicted_difficulty", "?") if new_j.get("difficulty_judge_status") == "ok" else "-"
            new_dep = new_j.get("path_dependency_judge", {}).get("path_dependency", "?") if new_j.get("path_dependency_judge_status") == "ok" else "-"

            lines.append(f"| {row_num} | {doc_id[:12]} | {old_q} | {old_pred} | {new_q} | {new_pred} | {new_dep} |")

    # Filter failure analysis
    lines.append("\n## Filter Failure Analysis\n")
    fail_items = [r for r in results if not r.get("final_filter_pass")]
    if fail_items:
        fail_reasons = defaultdict(int)
        for r in fail_items:
            reason = r.get("final_filter_reason", "?")
            # Extract first reason
            first = reason.split(";")[0].strip()
            fail_reasons[first] += 1
        lines.append("| Reason | Count |")
        lines.append("|--------|------:|")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("No filter failures.")

    # Examples
    lines.append("\n## Examples\n")

    # Good examples (filter pass + pred Hard or Medium + dep strong)
    good = [r for r in judged
            if r.get("difficulty_judge", {}).get("predicted_difficulty") in ("Hard", "Medium")
            and r.get("path_dependency_judge", {}).get("path_dependency") == "strong"]
    if good:
        lines.append("### Good Examples (Pred Hard/Medium + PathDep Strong)\n")
        for r in good[:5]:
            dj = r["difficulty_judge"]
            pj = r["path_dependency_judge"]
            events = r.get("events", [])
            path_str = " -> ".join(f"{e.get('trigger','?')}/{e.get('type','?')}" for e in events)
            lines.append(f"**[{r.get('difficulty')} -> {dj.get('predicted_difficulty')}] doc_id={r.get('doc_id','?')[:12]}**\n")
            lines.append(f"- Question: {r.get('generated_question', '')}")
            lines.append(f"- Answer: {r.get('gold_answer_phrase', '')[:100]}")
            lines.append(f"- Path: {path_str}")
            lines.append(f"- Steps: {dj.get('required_steps', '?')} | PathDep: {pj.get('path_dependency', '?')} | Answerable: {dj.get('answerable', '?')}")
            lines.append(f"- Reason: {dj.get('reason', '')}")
            lines.append("")

    # Bad examples (filter fail or pred Easy)
    bad = [r for r in results if not r.get("final_filter_pass")
           or (r.get("difficulty_judge_status") == "ok"
               and r.get("difficulty_judge", {}).get("predicted_difficulty") == "Easy")]
    if bad:
        lines.append("### Problem Examples\n")
        for r in bad[:5]:
            dj = r.get("difficulty_judge", {})
            status = "filter_fail" if not r.get("final_filter_pass") else f"pred={dj.get('predicted_difficulty','?')}"
            lines.append(f"**[{status}] doc_id={r.get('doc_id','?')[:12]}**\n")
            lines.append(f"- Question: {r.get('generated_question', '')[:150]}")
            lines.append(f"- Filter reason: {r.get('final_filter_reason', '?')}")
            if dj.get("reason"):
                lines.append(f"- Judge reason: {dj['reason']}")
            lines.append("")

    # Interpretation
    lines.append("\n## Interpretation\n")
    if pred_hard > 0:
        lines.append(f"- **Success**: {pred_hard} question(s) judged as Hard by independent evaluator.")
    else:
        lines.append("- **Failure**: No questions judged as Hard. See failure analysis above.")
    lines.append(f"- Path dependency strong: {dep_strong}/{len(judged)}")
    lines.append(f"- Implicit chain pass rate: {implicit_pass}/{total}")
    lines.append("- Compare with old Hard prompt: 0/6 judged Hard, 6/6 judged Easy/Medium")

    report_path = Path(output_dir) / "HARD_IMPLICIT_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Hard implicit chain pilot.")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--skip_judge", action="store_true", help="Skip independent difficulty judge")
    parser.add_argument("--skip_filter", action="store_true", help="Skip filter (use existing)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model config for judge
    cfg = get_api_config()
    model_config = {
        "api_url": cfg["AIHUBMIX_API_URL"],
        "api_key": cfg["AIHUBMIX_API_KEY"],
        "model": cfg["AIHUBMIX_MODEL"],
    }

    # 1. Load Hard paths
    print("=" * 60)
    print("Step 1: Load Hard strict paths")
    print("=" * 60)
    hard_paths = load_hard_paths()
    print(f"Loaded {len(hard_paths)} Hard paths")

    # 2. Generate
    print("\n" + "=" * 60)
    print("Step 2: Generate Hard questions (implicit chain)")
    print("=" * 60)
    raw_path = output_dir / "questions.raw.jsonl"
    if raw_path.exists() and not args.skip_filter:
        print(f"Loading existing results from {raw_path}")
        results = read_jsonl(raw_path)
    else:
        results = generate_implicit_hard(hard_paths)
        write_jsonl(raw_path, results)

    # 3. Filter
    print("\n" + "=" * 60)
    print("Step 3: Quality filter + implicitness check")
    print("=" * 60)
    filtered = run_filter(results)
    write_jsonl(output_dir / "questions.filtered.jsonl", filtered)
    pass_count = sum(1 for x in filtered if x.get("final_filter_pass"))
    implicit_count = sum(1 for x in filtered if x.get("hard_implicit_chain_pass", True))
    print(f"Filter pass: {pass_count}/{len(filtered)}")
    print(f"Implicit chain pass: {implicit_count}/{len(filtered)}")

    # 4. Independent difficulty judge
    if not args.skip_judge:
        print("\n" + "=" * 60)
        print("Step 4: Independent difficulty judge")
        print("=" * 60)
        judge_items = [x for x in filtered if x.get("final_filter_pass")]
        print(f"Judging {len(judge_items)} filter-passing items...")
        filtered = run_judge_on_items(filtered, model_config)

    write_jsonl(output_dir / "questions.judged.jsonl", filtered)

    # 5. Report
    print("\n" + "=" * 60)
    print("Step 5: Generate report")
    print("=" * 60)
    old_questions, old_judged = load_old_hard_data()
    generate_report(filtered, old_questions, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = len(results)
    gen_ok = sum(1 for r in results if r.get("generated_question") and not r.get("generation_error"))
    filter_pass = sum(1 for r in filtered if r.get("final_filter_pass"))
    judged = [r for r in filtered if r.get("difficulty_judge_status") == "ok"]
    pred_hard = sum(1 for r in judged if r.get("difficulty_judge", {}).get("predicted_difficulty") == "Hard")
    pred_medium = sum(1 for r in judged if r.get("difficulty_judge", {}).get("predicted_difficulty") == "Medium")
    pred_easy = sum(1 for r in judged if r.get("difficulty_judge", {}).get("predicted_difficulty") == "Easy")
    dep_strong = sum(1 for r in judged if r.get("path_dependency_judge", {}).get("path_dependency") == "strong")

    print(f"  Generated: {gen_ok}/{total}")
    print(f"  Filter pass: {filter_pass}/{total}")
    print(f"  Pred Hard: {pred_hard}/{len(judged)}")
    print(f"  Pred Medium: {pred_medium}/{len(judged)}")
    print(f"  Pred Easy: {pred_easy}/{len(judged)}")
    print(f"  PathDep Strong: {dep_strong}/{len(judged)}")
    print(f"\n  Old comparison: Pred Hard=0/6, Pred Medium=2/6, Pred Easy=4/6")
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
