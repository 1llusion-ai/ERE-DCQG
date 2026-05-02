"""QG Pilot on 29 strict-filtered paths.

Runs PathQG-HardAware generation + quality filter + solver + judge on a small
sample of strict-filtered paths (Easy 10, Medium 10, Hard 9).

Usage:
    python -m scripts.run_qg_pilot_strict
    python -m scripts.run_qg_pilot_strict --skip_solver
    python -m scripts.run_qg_pilot_strict --reuse_paths outputs/runs/qg_pilot_strict_29/selected_paths.jsonl --output_dir outputs/runs/qg_pilot_strict_29_v2
"""
import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from dcqg.utils.jsonl import read_jsonl, write_jsonl
from dcqg.generation.generator import generate_with_retry_hardaware
from dcqg.question_filter.pipeline import quality_filter_pipeline
from dcqg.evaluation.judge import llm_judge_v2, quality_judge, solve, target_event_hit
from dcqg.tracing import build_trace_from_pipeline_result, write_full_trace, write_readable_trace


INPUT_FILE = "outputs/runs/path_filter_strict_pilot/paths.filtered.strict.jsonl"
OUTPUT_DIR = "outputs/runs/qg_pilot_strict_29"
SEED = 42


def select_paths(input_file, n_easy=10, n_medium=10, n_hard=9, seed=42):
    """Sample paths by difficulty level."""
    items = read_jsonl(input_file)
    rng = random.Random(seed)
    by_level = defaultdict(list)
    for item in items:
        by_level[item.get("difficulty", "?")].append(item)
    for pool in by_level.values():
        rng.shuffle(pool)

    selected = []
    for level, n in [("Easy", n_easy), ("Medium", n_medium), ("Hard", n_hard)]:
        pool = by_level.get(level, [])
        take = pool[:n]
        selected.extend(take)
        print(f"  {level}: {len(take)}/{len(pool)} selected")

    rng.shuffle(selected)
    for i, item in enumerate(selected):
        item["_item_id"] = i
    return selected


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
        "llm_path_judge_model", "llm_path_judge_prompt", "llm_path_judge_raw_response",
        "policy_strict_keep", "policy_relaxed_keep", "policy_strict_reason",
        "policy_relaxed_reason", "risk_note", "dedup_key",
    ]
    for key in preserve_keys:
        if key in source and key not in result:
            result[key] = source[key]
    return result


def run_solver_eval(item):
    """Run solver + judge on one item. Modifies in-place."""
    if not item.get("final_filter_pass"):
        item["solver_eval_status"] = "not_run"
        item["solver_eval_reason"] = "final_filter_pass=false"
        return item

    question = item.get("generated_question", "")
    context = "\n".join(
        s if isinstance(s, str) else s[1]
        for s in item.get("supporting_sentences", [])
    )
    gold = item.get("gold_answer_trigger", "") or item.get("answer_trigger", "")
    path_events = item.get("events", [])
    difficulty = item.get("difficulty", "")

    try:
        solver_answer = solve(question, context)
        item["solver_answer"] = solver_answer

        hit_score, hit_method = target_event_hit(solver_answer, gold)
        item["target_event_hit"] = round(hit_score, 2)
        item["hit_method"] = hit_method

        answerable, solver_correct, support_covered = llm_judge_v2(question, context, gold, solver_answer)
        item["judge_answerable"] = round(answerable, 2)
        item["judge_solver_correct"] = round(solver_correct, 2)
        item["judge_support_covered"] = round(support_covered, 2)

        fluency, relevance, diff_align = quality_judge(question, path_events, difficulty)
        item["quality_fluency"] = round(fluency, 2)
        item["quality_path_relevance"] = round(relevance, 2)
        item["quality_difficulty_alignment"] = round(diff_align, 2)

        item["composite"] = round(
            0.25 * solver_correct + 0.20 * answerable + 0.15 * support_covered +
            0.15 * fluency + 0.10 * relevance + 0.15 * diff_align, 3)
        item["solver_eval_status"] = "ok"
        item["solver_eval_reason"] = ""
    except Exception as exc:
        item["solver_eval_status"] = "error"
        item["solver_eval_reason"] = f"{type(exc).__name__}: {exc}"
        item.setdefault("solver_answer", "")
        item.setdefault("judge_answerable", 0)
        item.setdefault("judge_solver_correct", 0)
        item.setdefault("judge_support_covered", 0)
        item.setdefault("quality_fluency", 0)
        item.setdefault("quality_path_relevance", 0)
        item.setdefault("quality_difficulty_alignment", 0)
        item.setdefault("composite", 0)

    return item


def load_v1_stats(v1_dir):
    """Load previous run statistics for comparison."""
    v1_path = Path(v1_dir)
    if not (v1_path / "questions.filtered.jsonl").exists():
        return None
    filtered = read_jsonl(v1_path / "questions.filtered.jsonl")
    by_level = defaultdict(list)
    for r in filtered:
        by_level[r.get("difficulty", "?")].append(r)

    hard_items = by_level.get("Hard", [])
    solver_xs = [x for x in filtered if x.get("solver_eval_status") == "ok"]

    stats = {
        "generated": sum(1 for x in filtered if x.get("generated_question") and not x.get("generation_error")),
        "filter_pass": sum(1 for x in filtered if x.get("final_filter_pass")),
        "path_coverage_fail": sum(1 for x in filtered if "path_coverage=" in x.get("final_filter_reason", "")),
        "hard_pc_fail": sum(1 for x in hard_items if "path_coverage=" in x.get("final_filter_reason", "")),
        "hard_degraded_fail": sum(1 for x in filtered if "hard_degraded=" in x.get("final_filter_reason", "") or "hard_shortcut=" in x.get("final_filter_reason", "")),
        "answer_consistency_fail": sum(1 for x in filtered if "answer_consistency=" in x.get("final_filter_reason", "")),
        "judge_error": sum(1 for x in filtered if x.get("answer_consistency_label") == "judge_error"),
        "filter_exceptions": sum(1 for x in filtered if "filter_exception" in x.get("final_filter_reason", "")),
        "solver_correct": sum(1 for x in solver_xs if x.get("judge_solver_correct", 0) >= 0.5),
        "solver_total": len(solver_xs),
        "total": len(filtered),
    }
    for level in ["Easy", "Medium", "Hard"]:
        xs = by_level[level]
        stats[f"{level}_filter_pass"] = sum(1 for x in xs if x.get("final_filter_pass"))
        stats[f"{level}_total"] = len(xs)
        level_solver = [x for x in xs if x.get("solver_eval_status") == "ok"]
        stats[f"{level}_solver_correct"] = sum(1 for x in level_solver if x.get("judge_solver_correct", 0) >= 0.5)
        stats[f"{level}_solver_total"] = len(level_solver)
    return stats


def generate_report(results, output_dir, v1_stats=None, version="v3", available_counts=None):
    """Generate QG_PILOT_REPORT.md with optional comparison to previous version."""
    lines = []
    lines.append(f"# QG Pilot Report (Strict 29 {version})\n")
    lines.append(f"**Input:** `{INPUT_FILE}`")
    lines.append(f"**Method:** PathQG-HardAware")
    lines.append(f"**Selected paths:** `outputs/runs/qg_pilot_strict_29/selected_paths.jsonl` (same as v1)")
    lines.append(f"**Total selected:** {len(results)}\n")
    lines.append(f"**Traces:**")
    lines.append(f"- full trace: `{output_dir}/debug_traces/full_trace.jsonl`")
    lines.append(f"- readable trace: `{output_dir}/debug_traces/readable_trace.md`\n")

    by_level = defaultdict(list)
    for r in results:
        by_level[r.get("difficulty", "?")].append(r)

    # Current stats
    total_gen = sum(1 for x in results if x.get("generated_question") and not x.get("generation_error"))
    total_pass = sum(1 for x in results if x.get("final_filter_pass"))
    total_pc_fail = sum(1 for x in results if "path_coverage=" in x.get("final_filter_reason", ""))
    total_hd_fail = sum(1 for x in results if "hard_degraded=" in x.get("final_filter_reason", "") or "hard_shortcut=" in x.get("final_filter_reason", ""))
    total_ac_fail = sum(1 for x in results if "answer_consistency=" in x.get("final_filter_reason", ""))
    total_je = sum(1 for x in results if x.get("answer_consistency_label") == "judge_error")
    total_fe = sum(1 for x in results if "filter_exception" in x.get("final_filter_reason", ""))
    solver_xs = [x for x in results if x.get("solver_eval_status") == "ok"]
    solver_correct = sum(1 for x in solver_xs if x.get("judge_solver_correct", 0) >= 0.5)

    # Hard path coverage fails
    hard_items = by_level.get("Hard", [])
    hard_pc_fail = sum(1 for x in hard_items if "path_coverage=" in x.get("final_filter_reason", ""))

    if v1_stats:
        prev_label = version.replace("v3", "v2").replace("v4", "v3")
        lines.append(f"## {prev_label} vs {version} Comparison\n")
        lines.append(f"| Metric | {prev_label} | {version} | Change |")
        lines.append("|--------|---:|---:|-------:|")

        def cmp(curr, prev):
            diff = curr - prev
            sign = "+" if diff > 0 else ""
            return f"{sign}{diff}"

        lines.append(f"| Generated | {v1_stats['generated']} | {total_gen} | {cmp(total_gen, v1_stats['generated'])} |")
        lines.append(f"| Filter pass | {v1_stats['filter_pass']} | {total_pass} | {cmp(total_pass, v1_stats['filter_pass'])} |")
        lines.append(f"| Easy filter pass | {v1_stats.get('Easy_filter_pass',0)}/{v1_stats.get('Easy_total',0)} | {sum(1 for x in by_level['Easy'] if x.get('final_filter_pass'))}/{len(by_level['Easy'])} | |")
        lines.append(f"| Medium filter pass | {v1_stats.get('Medium_filter_pass',0)}/{v1_stats.get('Medium_total',0)} | {sum(1 for x in by_level['Medium'] if x.get('final_filter_pass'))}/{len(by_level['Medium'])} | |")
        lines.append(f"| Hard filter pass | {v1_stats.get('Hard_filter_pass',0)}/{v1_stats.get('Hard_total',0)} | {sum(1 for x in by_level['Hard'] if x.get('final_filter_pass'))}/{len(by_level['Hard'])} | |")
        lines.append(f"| Path coverage fails | {v1_stats['path_coverage_fail']} | {total_pc_fail} | {cmp(total_pc_fail, v1_stats['path_coverage_fail'])} |")
        lines.append(f"| Hard path coverage fails | {v1_stats.get('hard_pc_fail', 0)} | {hard_pc_fail} | {cmp(hard_pc_fail, v1_stats.get('hard_pc_fail', 0))} |")
        lines.append(f"| Hard degraded fails | {v1_stats['hard_degraded_fail']} | {total_hd_fail} | {cmp(total_hd_fail, v1_stats['hard_degraded_fail'])} |")
        lines.append(f"| Answer consistency fails | {v1_stats['answer_consistency_fail']} | {total_ac_fail} | {cmp(total_ac_fail, v1_stats['answer_consistency_fail'])} |")
        lines.append(f"| Judge JSON errors | {v1_stats['judge_error']} | {total_je} | {cmp(total_je, v1_stats['judge_error'])} |")
        lines.append(f"| Filter exceptions | {v1_stats.get('filter_exceptions', 0)} | {total_fe} | {cmp(total_fe, v1_stats.get('filter_exceptions', 0))} |")
        lines.append(f"| Solver correct | {v1_stats.get('solver_correct', '-')}/{v1_stats.get('solver_total', 0)} | {solver_correct}/{len(solver_xs)} | |")

        lines.append("\n### Per-Difficulty Comparison\n")
        lines.append(f"| Metric | {prev_label} | {version} |")
        lines.append("|--------|---:|---:|")
        for level in ["Easy", "Medium", "Hard"]:
            v1p = v1_stats.get(f"{level}_filter_pass", 0)
            v1t = v1_stats.get(f"{level}_total", 0)
            v2p = sum(1 for x in by_level[level] if x.get("final_filter_pass"))
            v2t = len(by_level[level])
            lines.append(f"| {level} filter pass | {v1p}/{v1t} | {v2p}/{v2t} |")
        for level in ["Easy", "Medium", "Hard"]:
            v1c = v1_stats.get(f"{level}_solver_correct", 0)
            v1st = v1_stats.get(f"{level}_solver_total", 0)
            v2_xs = [x for x in by_level[level] if x.get("solver_eval_status") == "ok"]
            v2c = sum(1 for x in v2_xs if x.get("judge_solver_correct", 0) >= 0.5)
            lines.append(f"| {level} solver correct | {v1c}/{v1st} | {v2c}/{len(v2_xs)} |")

    # Selection summary (show available if provided)
    lines.append("\n## Selection\n")
    if available_counts:
        lines.append("| Difficulty | Available | Selected |")
        lines.append("|------------|----------:|---------:|")
        total_avail = 0
        for level in ["Easy", "Medium", "Hard"]:
            avail = available_counts.get(level, 0)
            total_avail += avail
            lines.append(f"| {level} | {avail} | {len(by_level[level])} |")
        lines.append(f"| **Total** | **{total_avail}** | **{len(results)}** |")
    else:
        lines.append("| Difficulty | Selected |")
        lines.append("|------------|---------:|")
        for level in ["Easy", "Medium", "Hard"]:
            lines.append(f"| {level} | {len(by_level[level])} |")
        lines.append(f"| **Total** | **{len(results)}** |")

    # Generation success
    lines.append("\n## Generation Success\n")
    lines.append("| Difficulty | Generated | Error | Gen% |")
    lines.append("|------------|----------:|------:|-----:|")
    for level in ["Easy", "Medium", "Hard"]:
        xs = by_level[level]
        gen = sum(1 for x in xs if x.get("generated_question") and not x.get("generation_error"))
        err = sum(1 for x in xs if x.get("generation_error"))
        pct = gen / len(xs) * 100 if xs else 0
        lines.append(f"| {level} | {gen} | {err} | {pct:.0f}% |")
    total_err = sum(1 for x in results if x.get("generation_error"))
    lines.append(f"| **Total** | **{total_gen}** | **{total_err}** | **{total_gen/len(results)*100:.0f}%** |")

    # Question filter pass
    lines.append("\n## Question Filter\n")
    lines.append("| Difficulty | Filter Pass | Filter Fail | Pass% |")
    lines.append("|------------|------------:|------------:|------:|")
    for level in ["Easy", "Medium", "Hard"]:
        xs = by_level[level]
        passed = sum(1 for x in xs if x.get("final_filter_pass"))
        failed = len(xs) - passed
        pct = passed / len(xs) * 100 if xs else 0
        lines.append(f"| {level} | {passed} | {failed} | {pct:.0f}% |")
    total_fail = len(results) - total_pass
    lines.append(f"| **Total** | **{total_pass}** | **{total_fail}** | **{total_pass/len(results)*100:.0f}%** |")

    # Solver results
    has_solver = any(x.get("solver_eval_status") == "ok" for x in results)
    if has_solver:
        lines.append("\n## Solver + Judge Results\n")
        lines.append("| Difficulty | Solver OK | Solver Correct | Answerable | Support Covered | Composite |")
        lines.append("|------------|----------:|---------------:|-----------:|----------------:|----------:|")
        for level in ["Easy", "Medium", "Hard"]:
            xs = [x for x in by_level[level] if x.get("solver_eval_status") == "ok"]
            if not xs:
                lines.append(f"| {level} | 0 | - | - | - | - |")
                continue
            n = len(xs)
            correct = sum(1 for x in xs if x.get("judge_solver_correct", 0) >= 0.5)
            avg_ans = sum(x.get("judge_answerable", 0) for x in xs) / n
            avg_cor = sum(x.get("judge_solver_correct", 0) for x in xs) / n
            avg_sup = sum(x.get("judge_support_covered", 0) for x in xs) / n
            avg_comp = sum(x.get("composite", 0) for x in xs) / n
            lines.append(f"| {level} | {n} | {correct} ({avg_cor:.0%}) | {avg_ans:.0%} | {avg_sup:.0%} | {avg_comp:.3f} |")
        if solver_xs:
            n = len(solver_xs)
            correct = sum(1 for x in solver_xs if x.get("judge_solver_correct", 0) >= 0.5)
            avg_ans = sum(x.get("judge_answerable", 0) for x in solver_xs) / n
            avg_cor = sum(x.get("judge_solver_correct", 0) for x in solver_xs) / n
            avg_sup = sum(x.get("judge_support_covered", 0) for x in solver_xs) / n
            avg_comp = sum(x.get("composite", 0) for x in solver_xs) / n
            lines.append(f"| **Total** | **{n}** | **{correct} ({avg_cor:.0%})** | **{avg_ans:.0%}** | **{avg_sup:.0%}** | **{avg_comp:.3f}** |")

    # Filter failure reasons
    lines.append("\n## Top Filter Fail Reasons\n")
    fail_reasons = Counter()
    for x in results:
        reason = x.get("final_filter_reason", "")
        if reason and not x.get("final_filter_pass"):
            for part in reason.split("; "):
                fail_reasons[part.strip()] += 1
    lines.append("| Reason | Count |")
    lines.append("|--------|------:|")
    for reason, count in fail_reasons.most_common(15):
        lines.append(f"| {reason} | {count} |")

    # Path coverage
    lines.append("\n## Path Coverage\n")
    lines.append("| Difficulty | Avg Prior Coverage | Coverage Pass% |")
    lines.append("|------------|-------------------:|---------------:|")
    for level in ["Easy", "Medium", "Hard"]:
        xs = by_level[level]
        cov = [x.get("path_coverage_count", 0) for x in xs if "path_coverage_count" in x]
        passed = sum(1 for x in xs if x.get("path_coverage_pass"))
        avg = sum(cov) / len(cov) if cov else 0
        pct = passed / len(xs) * 100 if xs else 0
        lines.append(f"| {level} | {avg:.1f} | {pct:.0f}% |")

    # Hard shortcut analysis
    hard_items = by_level.get("Hard", [])
    lines.append(f"\n## Hard Shortcut Analysis\n")
    lines.append(f"- Hard items: {len(hard_items)}")
    degraded = [x for x in hard_items if x.get("hard_degraded")]
    lines.append(f"- Degraded (shortcut=yes AND needs_prior=no): {len(degraded)}")
    shortcut_yes = [x for x in hard_items if x.get("shortcut_without_path") == "yes"]
    needs_no = [x for x in hard_items if x.get("needs_prior_events_to_identify_answer") == "no"]
    lines.append(f"- shortcut_without_path=yes: {len(shortcut_yes)}")
    lines.append(f"- needs_prior=no: {len(needs_no)}")
    if degraded:
        for x in degraded:
            lines.append(f"  - `{x.get('doc_id','')}`: {x.get('hard_degraded_reason','')}")

    # Hard per-item coverage table
    if hard_items:
        lines.append("\n## Hard Per-Item Path Coverage\n")
        lines.append("| item_id | question | prior_count | all_count | pass? | reason |")
        lines.append("|---------|----------|------------:|----------:|-------|--------|")
        for x in hard_items:
            iid = x.get("_item_id", "?")
            q = x.get("generated_question", "")[:60]
            prior = x.get("path_coverage_prior_count", "?")
            allc = x.get("path_coverage_all_count", "?")
            passed = "PASS" if x.get("path_coverage_pass") else "FAIL"
            reason = x.get("path_coverage_reason", "")[:50]
            lines.append(f"| {iid} | {q} | {prior} | {allc} | {passed} | {reason} |")

    # Good examples (5)
    lines.append("\n## Good Examples (5)\n")
    good = [x for x in results if x.get("final_filter_pass")]
    if has_solver:
        good.sort(key=lambda x: x.get("composite", 0), reverse=True)
    for x in good[:10]:
        lines.append(f"### [{x.get('difficulty','?')}] {x.get('title','')[:60]}\n")
        lines.append(f"- **doc_id:** `{x.get('doc_id','')}`")
        events = x.get("events", [])
        path_str = " -> ".join(e.get("trigger", "") for e in events)
        lines.append(f"- **path:** {path_str}")
        lines.append(f"- **question:** {x.get('generated_question','')}")
        lines.append(f"- **gold_answer:** {x.get('gold_answer_phrase','')}")
        if has_solver:
            lines.append(f"- **solver_answer:** {x.get('solver_answer','')}")
            lines.append(f"- **judge_correct:** {x.get('judge_solver_correct', '?')}")
            lines.append(f"- **composite:** {x.get('composite', '?')}")
        lines.append("")

    # Bad examples (10)
    lines.append("\n## Bad Examples (10)\n")
    bad = [x for x in results if not x.get("final_filter_pass")]
    if has_solver:
        failed_solved = [x for x in results if x.get("final_filter_pass") and x.get("judge_solver_correct", 0) < 0.5]
        bad = bad + failed_solved
    for x in bad[:10]:
        lines.append(f"### [{x.get('difficulty','?')}] {x.get('title','')[:60]}\n")
        lines.append(f"- **doc_id:** `{x.get('doc_id','')}`")
        events = x.get("events", [])
        path_str = " -> ".join(e.get("trigger", "") for e in events)
        lines.append(f"- **path:** {path_str}")
        lines.append(f"- **question:** {x.get('generated_question','')}")
        lines.append(f"- **filter_pass:** {x.get('final_filter_pass')}")
        lines.append(f"- **filter_reason:** {x.get('final_filter_reason','')}")
        if has_solver:
            lines.append(f"- **solver_answer:** {x.get('solver_answer','')}")
            lines.append(f"- **judge_correct:** {x.get('judge_solver_correct', '?')}")
        lines.append("")

    # Conclusion
    lines.append("\n## Conclusion\n")
    lines.append(f"- Selected: {len(results)} (Easy {len(by_level['Easy'])}, Medium {len(by_level['Medium'])}, Hard {len(by_level['Hard'])})")
    lines.append(f"- Generated: {total_gen}/{len(results)} ({total_gen/len(results)*100:.0f}%)")
    lines.append(f"- Filter pass: {total_pass}/{len(results)} ({total_pass/len(results)*100:.0f}%)")
    if has_solver and solver_xs:
        lines.append(f"- Solver correct: {solver_correct}/{len(solver_xs)} ({solver_correct/len(solver_xs)*100:.0f}%)")
    lines.append(f"- Path coverage fails: {total_pc_fail}")
    lines.append(f"- Hard shortcut fails: {total_hd_fail}")
    lines.append(f"- Answer consistency fails: {total_ac_fail}")

    report_path = Path(output_dir) / "QG_PILOT_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report: {report_path}")


def generate_audit(results, output_dir, version="v3"):
    """Generate MANUAL_QG_AUDIT.md with all items."""
    lines = []
    lines.append(f"# Manual QG Audit — Strict 29 {version}\n")
    lines.append(f"**Source:** `{INPUT_FILE}`")
    lines.append(f"**Selected paths:** `outputs/runs/qg_pilot_strict_29/selected_paths.jsonl` (same as v1)")
    lines.append(f"**Total:** {len(results)}\n")

    by_level = defaultdict(list)
    for r in results:
        by_level[r.get("difficulty", "?")].append(r)

    for level in ["Easy", "Medium", "Hard"]:
        xs = by_level[level]
        lines.append(f"\n## {level} ({len(xs)} items)\n")
        for i, x in enumerate(xs, 1):
            lines.append(f"### {level} #{i}\n")
            lines.append(f"- **doc_id:** `{x.get('doc_id', '')}`")
            lines.append(f"- **title:** {x.get('title', '')}")
            lines.append(f"- **difficulty:** {x.get('difficulty', '?')}")

            events = x.get("events", [])
            path_str = " -> ".join(f"{e.get('trigger','')}/{e.get('type','')}" for e in events)
            lines.append(f"- **event_path:** {path_str}")
            lines.append(f"- **relations:** {', '.join(x.get('relation_subtypes', []))}")
            lines.append(f"- **gold_answer_phrase:** `{x.get('gold_answer_phrase', '')}`")
            lines.append(f"- **gold_answer_sentence:** {x.get('gold_answer_sentence', '')[:150]}")

            # Generation
            lines.append(f"- **generated_question:** {x.get('generated_question', '')}")
            lines.append(f"- **generation_error:** {x.get('generation_error', False)}")
            lines.append(f"- **retry_attempts:** {x.get('retry_attempts', 0)}")

            # Filter
            lines.append(f"- **grammar_pass:** {x.get('grammar_pass', '?')} ({x.get('grammar_reason', '')})")
            lines.append(f"- **answer_phrase_pass:** {x.get('answer_phrase_pass', '?')} ({x.get('answer_phrase_reason', '')})")
            lines.append(f"- **weak_trigger_pass:** {x.get('weak_trigger_pass', '?')} ({x.get('weak_trigger_reason', '')})")
            lines.append(f"- **answer_consistency:** {x.get('answer_consistency_label', '?')} ({x.get('answer_consistency_reason', '')})")
            lines.append(f"- **path_coverage:** count={x.get('path_coverage_count', '?')} pass={x.get('path_coverage_pass', '?')} ({x.get('path_coverage_reason', '')})")
            if x.get("difficulty") == "Hard":
                lines.append(f"- **shortcut_without_path:** {x.get('shortcut_without_path', '?')}")
                lines.append(f"- **needs_prior_events:** {x.get('needs_prior_events_to_identify_answer', '?')}")
                lines.append(f"- **hard_degraded:** {x.get('hard_degraded', '?')} ({x.get('hard_degraded_reason', '')})")
            lines.append(f"- **final_filter_pass:** {x.get('final_filter_pass', '?')}")
            lines.append(f"- **final_filter_reason:** {x.get('final_filter_reason', '')}")

            # Solver
            lines.append(f"- **solver_answer:** {x.get('solver_answer', 'not_run')}")
            lines.append(f"- **solver_eval_status:** {x.get('solver_eval_status', 'not_run')}")
            lines.append(f"- **judge_answerable:** {x.get('judge_answerable', '?')}")
            lines.append(f"- **judge_solver_correct:** {x.get('judge_solver_correct', '?')}")
            lines.append(f"- **judge_support_covered:** {x.get('judge_support_covered', '?')}")
            lines.append(f"- **quality_fluency:** {x.get('quality_fluency', '?')}")
            lines.append(f"- **quality_path_relevance:** {x.get('quality_path_relevance', '?')}")
            lines.append(f"- **quality_difficulty_alignment:** {x.get('quality_difficulty_alignment', '?')}")
            lines.append(f"- **composite:** {x.get('composite', '?')}")

            # Trace link
            lines.append(f"- **item_id:** {x.get('_item_id', '?')}")
            lines.append("")

    audit_path = Path(output_dir) / "MANUAL_QG_AUDIT.md"
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Audit: {audit_path}")


def main():
    parser = argparse.ArgumentParser(description="QG pilot on 29 strict-filtered paths.")
    parser.add_argument("--input", default=INPUT_FILE, help="Input strict paths JSONL")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--n_easy", type=int, default=10)
    parser.add_argument("--n_medium", type=int, default=10)
    parser.add_argument("--n_hard", type=int, default=9)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max_attempts", type=int, default=3, help="Max generation attempts per item")
    parser.add_argument("--skip_solver", action="store_true", help="Skip solver + judge")
    parser.add_argument("--reuse_paths", default=None, help="Load selected paths from this file instead of re-sampling")
    parser.add_argument("--v1_dir", default="outputs/runs/qg_pilot_strict_29", help="Previous output dir for comparison")
    parser.add_argument("--version", default="v3", help="Version label for report")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    trace_dir = output_dir / "debug_traces"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Select paths (or reuse)
    print("=" * 60)
    print("Step 1: Select paths")
    print("=" * 60)
    # Count available paths per difficulty
    all_items = read_jsonl(args.input)
    available_counts = Counter(x.get("difficulty", "?") for x in all_items)
    print(f"  Available: Easy={available_counts['Easy']}, Medium={available_counts['Medium']}, Hard={available_counts['Hard']}")

    if args.reuse_paths:
        selected = read_jsonl(args.reuse_paths)
        for i, item in enumerate(selected):
            item["_item_id"] = i
        print(f"  Reused {len(selected)} paths from {args.reuse_paths}")
    else:
        # Cap n per level at available
        n_easy = min(args.n_easy, available_counts["Easy"])
        n_medium = min(args.n_medium, available_counts["Medium"])
        n_hard = min(args.n_hard, available_counts["Hard"])
        selected = select_paths(args.input, n_easy, n_medium, n_hard, args.seed)
    print(f"Total selected: {len(selected)}")
    write_jsonl(output_dir / "selected_paths.jsonl", selected)

    # Load v1 stats for comparison
    v1_stats = load_v1_stats(args.v1_dir) if args.v1_dir else None
    if v1_stats:
        print(f"  Loaded v1 stats from {args.v1_dir}")

    # 2. Generate questions
    print("\n" + "=" * 60)
    print("Step 2: Generate questions (PathQG-HardAware)")
    print("=" * 60)
    generated = []
    for i, item in enumerate(selected):
        diff = item.get("difficulty", "?")
        doc_id = item.get("doc_id", "")[:12]
        print(f"[{i+1}/{len(selected)}] {diff} {doc_id}", end=" ")
        try:
            result, attempts = generate_with_retry_hardaware(item, max_attempts=args.max_attempts)
            result = merge_context(result, item)
            result["_item_id"] = i
            gen_ok = bool(result.get("generated_question")) and not result.get("generation_error")
            print(f"-> {'OK' if gen_ok else 'FAIL'} (attempts={attempts})")
        except Exception as exc:
            result = {
                "_item_id": i,
                "doc_id": item.get("doc_id", ""),
                "difficulty": diff,
                "method": "PathQG-HardAware",
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
                "generation_prompts": [],
                "generation_raw_responses": [],
            }
            result = merge_context(result, item)
            print(f"-> EXCEPTION: {exc}")
        generated.append(result)
        time.sleep(0.1)

    write_jsonl(output_dir / "questions.raw.jsonl", generated)
    gen_ok_count = sum(1 for x in generated if x.get("generated_question") and not x.get("generation_error"))
    print(f"\nGenerated: {gen_ok_count}/{len(generated)}")

    # 3. Quality filter
    print("\n" + "=" * 60)
    print("Step 3: Quality filter")
    print("=" * 60)
    filtered = []
    for i, r in enumerate(generated):
        if r.get("generation_error"):
            r.setdefault("final_filter_pass", False)
            r.setdefault("final_filter_reason", "generation_error")
            filtered.append(r)
            continue
        try:
            r = quality_filter_pipeline(r, skip_llm=False)
        except Exception as exc:
            r["final_filter_pass"] = False
            r["final_filter_reason"] = f"filter_exception: {type(exc).__name__}: {exc}"
        filtered.append(r)

    write_jsonl(output_dir / "questions.filtered.jsonl", filtered)
    pass_count = sum(1 for x in filtered if x.get("final_filter_pass"))
    print(f"Filter pass: {pass_count}/{len(filtered)}")

    # 4. Solver + Judge
    if not args.skip_solver:
        print("\n" + "=" * 60)
        print("Step 4: Solver + Judge")
        print("=" * 60)
        for i, r in enumerate(filtered):
            diff = r.get("difficulty", "?")
            if not r.get("final_filter_pass"):
                r["solver_eval_status"] = "not_run"
                r["solver_eval_reason"] = "final_filter_pass=false"
                print(f"[{i+1}/{len(filtered)}] {diff} SKIP (filter fail)")
                continue
            print(f"[{i+1}/{len(filtered)}] {diff}", end=" ")
            r = run_solver_eval(r)
            status = r.get("solver_eval_status", "?")
            correct = r.get("judge_solver_correct", "?")
            print(f"-> solver={status} correct={correct}")
            time.sleep(0.15)

        write_jsonl(output_dir / "solver_judge.jsonl", filtered)
        solver_ok = sum(1 for x in filtered if x.get("solver_eval_status") == "ok")
        solver_correct = sum(1 for x in filtered if x.get("judge_solver_correct", 0) >= 0.5)
        print(f"\nSolver OK: {solver_ok}, Correct: {solver_correct}")
    else:
        for r in filtered:
            r["solver_eval_status"] = "not_run"
            r["solver_eval_reason"] = "skip_solver"
        write_jsonl(output_dir / "solver_judge.jsonl", filtered)
        print("\nSolver skipped.")

    # 5. Traces
    print("\n" + "=" * 60)
    print("Step 5: Generate traces")
    print("=" * 60)
    traces = [build_trace_from_pipeline_result(r, item_id=i) for i, r in enumerate(filtered)]
    write_full_trace(traces, trace_dir)
    write_readable_trace(traces, trace_dir)
    print(f"Traces written to {trace_dir}")

    # 6. Reports
    print("\n" + "=" * 60)
    print("Step 6: Generate reports")
    print("=" * 60)
    generate_report(filtered, output_dir, v1_stats, version=args.version, available_counts=available_counts)
    generate_audit(filtered, output_dir, version=args.version)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    by_level = defaultdict(list)
    for r in filtered:
        by_level[r.get("difficulty", "?")].append(r)
    for level in ["Easy", "Medium", "Hard"]:
        xs = by_level[level]
        gen = sum(1 for x in xs if x.get("generated_question") and not x.get("generation_error"))
        passed = sum(1 for x in xs if x.get("final_filter_pass"))
        solver_xs = [x for x in xs if x.get("solver_eval_status") == "ok"]
        correct = sum(1 for x in solver_xs if x.get("judge_solver_correct", 0) >= 0.5)
        print(f"  {level}: selected={len(xs)} gen={gen} filter_pass={passed} solver_correct={correct}/{len(solver_xs)}")

    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
