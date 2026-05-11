"""FairytaleQA evidence-role-aware QG pilot.

Runs 4 methods (Direct, ICL, SelfRefine, Ours) on Hard candidates
with narrative evidence graphs. Judges quality and difficulty.

Usage:
  python -m scripts.run_fairytale_qg_pilot \
    --graphs outputs/runs/narrative_graph_audit_fixed_20260510/graphs.jsonl \
    --candidates outputs/runs/fairytale_evidence_audit_train_implicit_500_20260510/candidates.jsonl \
    --limit 15 \
    --output_dir outputs/runs/fairytale_qg_hard_pilot_20260510
"""
import argparse
import json
import math
import re
import time
from pathlib import Path
from collections import Counter


def _ascii_safe(text):
    """Replace Unicode dashes/smart punctuation with ASCII equivalents."""
    if not text:
        return ""
    return (text
            .replace('—', ' - ')  # em dash
            .replace('–', '-')     # en dash
            .replace('‘', "'")     # left single quote
            .replace('’', "'")     # right single quote
            .replace('“', '"')     # left double quote
            .replace('”', '"'))    # right double quote


def _wilson_ci(k, n, z=1.96):
    """Wilson score interval for binomial proportion. Returns (lower, upper)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    lo = max(0.0, center - spread)
    hi = min(1.0, center + spread)
    return (lo, hi)


def _wilson_str(k, n):
    """Format Wilson CI as string (ASCII only)."""
    lo, hi = _wilson_ci(k, n)
    return f"{k}/{n} ({100*k/n:.1f}%, 95%CI [{100*lo:.1f}, {100*hi:.1f}%])" if n else "N/A"

from dcqg.generation.fairytale_qg import (
    generate_direct,
    generate_icl,
    generate_self_refine,
    generate_ours,
    quality_judge,
    difficulty_evidence_judge,
    compute_evidence_coverage,
    semantic_evidence_match_judge,
)


def _split_sentences(text):
    """Split text into sentences."""
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if s.strip()]


def _load_graphs(path):
    """Load graph records from JSONL."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_candidates(path):
    """Load candidate records and build lookup by (story_name, question)."""
    lookup = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                key = (rec.get("story_name", ""), rec.get("question", ""))
                lookup[key] = rec
    return lookup


def _join_graphs_with_candidates(graphs, candidates_lookup):
    """Join graph records with candidate records to get story_section."""
    joined = []
    for g in graphs:
        key = (g.get("story_name", ""), g.get("question", ""))
        cand = candidates_lookup.get(key)
        if cand and cand.get("story_section"):
            merged = dict(g)
            merged["story_section"] = cand["story_section"]
            merged["original_question"] = g.get("question", "")
            joined.append(merged)
    return joined


def _select_candidates(graphs, limit):
    """Filter for valid Hard graphs and sample."""
    valid = [g for g in graphs if g.get("graph_valid") is True]
    # Already Hard-only from graph extraction
    print(f"  Valid Hard graphs: {len(valid)}")
    if limit and limit < len(valid):
        # Stratified by necessity_type
        by_type = {}
        for g in valid:
            nt = g.get("necessity_type", "unknown")
            by_type.setdefault(nt, []).append(g)
        selected = []
        round_robin = list(by_type.values())
        idx = 0
        while len(selected) < limit:
            pool = round_robin[idx % len(round_robin)]
            if pool:
                selected.append(pool.pop(0))
            idx += 1
            # Safety: break if all pools empty
            if all(len(p) == 0 for p in round_robin):
                break
        return selected
    return valid


def _run_method(method_name, candidate):
    """Run a single generation method on a candidate. Returns result dict."""
    story_section = candidate["story_section"]
    target_answer = candidate.get("answer", "") or candidate.get("answer1", "")
    difficulty = "Hard"

    if method_name == "Direct":
        result, attempts = generate_direct(story_section, target_answer, difficulty)
    elif method_name == "ICL":
        result, attempts = generate_icl(story_section, target_answer, difficulty)
    elif method_name == "SelfRefine":
        result, attempts = generate_self_refine(story_section, target_answer, difficulty)
    elif method_name == "Ours":
        result, attempts = generate_ours(
            story_section, target_answer, difficulty,
            nodes=candidate.get("nodes", []),
            edges=candidate.get("edges", []),
            required_evidence_sentences=candidate.get("required_evidence_sentences", []),
            bridge_sentence_ids=candidate.get("bridge_sentence_ids", []),
            reasoning_operation=candidate.get("reasoning_operation", ""),
            necessity_type=candidate.get("necessity_type", ""),
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    result["attempts"] = attempts
    return result


def _judge_generation(result, candidate):
    """Run both judges on a generation result."""
    question = result.get("generated_question", "")
    story_section = candidate["story_section"]
    target_answer = candidate.get("answer", "") or candidate.get("answer1", "")
    difficulty = "Hard"

    # Quality judge
    qj = quality_judge(question, story_section, target_answer, difficulty)
    result["quality_judge"] = qj
    result["quality_pass"] = qj.get("quality_pass", False)
    result["strict_quality_pass"] = qj.get("strict_quality_pass", False)

    # Difficulty/evidence judge
    dj = difficulty_evidence_judge(
        question, story_section, target_answer, difficulty,
        required_evidence_sentences=candidate.get("required_evidence_sentences", []),
        bridge_sentence_ids=candidate.get("bridge_sentence_ids", []),
    )
    result["difficulty_judge"] = dj
    result["predicted_difficulty"] = dj.get("predicted_difficulty", "judge_error")
    result["difficulty_judge_status"] = dj.get("difficulty_judge_status", "ok")
    result["difficulty_judge_parse_ok"] = dj.get("difficulty_judge_parse_ok", True)

    # Evidence coverage diagnostic
    ec = compute_evidence_coverage(
        dj,
        candidate.get("required_evidence_sentences", []),
        candidate.get("bridge_sentence_ids", []),
    )
    result["evidence_coverage"] = ec
    result["target_evidence_coverage"] = ec.get("target_evidence_coverage", 0.0)
    result["hard_realization_pass"] = ec.get("hard_realization_pass", "no")

    # Semantic evidence match judge
    sem = semantic_evidence_match_judge(
        question, story_section, target_answer,
        candidate.get("required_evidence_sentences", []),
        dj.get("required_evidence_sentences_used", []),
    )
    result["semantic_evidence_match"] = sem.get("semantic_evidence_match", "judge_error")
    result["semantic_match_reason"] = sem.get("semantic_match_reason", "")

    # Hard realization pass v2
    hrp_v2 = (
        result.get("predicted_difficulty") == "Hard"
        and dj.get("difficulty_judge_status") == "ok"
        and len(dj.get("required_evidence_sentences_used", [])) >= 3
        and dj.get("bridge_required") == "yes"
        and dj.get("answer_sentence_alone_sufficient") == "no"
        and result.get("semantic_evidence_match") in ("yes", "partial")
    )
    result["hard_realization_pass_v2"] = "yes" if hrp_v2 else "no"

    # Strict hrp_v2: hrp_v2 + strict quality + focus match
    strict_hrp_v2 = (
        hrp_v2
        and result.get("strict_quality_pass") is True
        and result.get("focus_match") == "yes"
    )
    result["strict_hrp_v2"] = "yes" if strict_hrp_v2 else "no"

    return result


def _build_report(all_results, output_dir, meta=None):
    """Write the pilot report."""
    report_path = output_dir / "FAIRYTALE_QG_HARD_PILOT_REPORT.md"
    meta = meta or {}

    # Aggregate stats
    methods = ["Direct", "ICL", "SelfRefine", "Ours"]
    method_results = {m: [] for m in methods}
    for r in all_results:
        m = r.get("method", "Unknown")
        if m in method_results:
            method_results[m].append(r)

    lines = [
        "# FairytaleQA Hard QG Pilot Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Run Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Methods | {', '.join(methods)} |",
        f"| Requested limit | {(meta or {}).get('requested_limit', 'N/A')} |",
        f"| Graph total | {(meta or {}).get('graph_total', 'N/A')} |",
        f"| Graph valid | {(meta or {}).get('graph_valid', 'N/A')} |",
        f"| Selected candidates | {(meta or {}).get('selected_candidates', len(all_results) // len(methods) if methods else 0)} |",
        f"| Total generations | {len(all_results)} |",
        f"| Target difficulty | Hard |",
        "",
    ]

    # Parse OK stats
    lines.append("### Parse success by method")
    lines.append("")
    lines.append("| Method | parse_ok | Total | Pct |")
    lines.append("|---|---:|---:|---:|")
    for m in methods:
        rs = method_results[m]
        ok = sum(1 for r in rs if r.get("parse_ok"))
        total = len(rs)
        pct = f"{100 * ok / total:.1f}%" if total else "N/A"
        lines.append(f"| {m} | {ok} | {total} | {pct} |")
    lines.append("")

    # 1b. Generation robustness by method
    lines.append("### 1b. Generation Robustness by Method")
    lines.append("")
    lines.append("| Method | degenerate | repair_attempted | repair_success | quality_pass |")
    lines.append("|---|---:|---:|---:|---:|")
    for m in methods:
        rs = method_results[m]
        degenerate = sum(1 for r in rs if r.get("generation_error") == "degenerate output")
        repair_att = sum(1 for r in rs if r.get("repair_attempted"))
        repair_ok = sum(1 for r in rs if r.get("repair_success"))
        qp = sum(1 for r in rs if r.get("quality_pass"))
        lines.append(f"| {m} | {degenerate} | {repair_att} | {repair_ok} | {qp} |")
    lines.append("")

    # 2. Quality pass by method
    lines.append("## 2. Quality Pass by Method")
    lines.append("")
    lines.append("| Method | quality_pass | strict_quality_pass | Total | Pct (loose) | Pct (strict) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m in methods:
        rs = method_results[m]
        qp = sum(1 for r in rs if r.get("quality_pass"))
        sqp = sum(1 for r in rs if r.get("strict_quality_pass"))
        total = len(rs)
        pct = f"{100 * qp / total:.1f}%" if total else "N/A"
        spct = f"{100 * sqp / total:.1f}%" if total else "N/A"
        lines.append(f"| {m} | {qp} | {sqp} | {total} | {pct} | {spct} |")
    lines.append("")

    # 3. Blind difficulty distribution among quality-pass
    lines.append("## 3. Blind Difficulty Distribution (quality-pass only)")
    lines.append("")
    lines.append("| Method | Easy | Medium | Hard | JudgeError | Total |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m in methods:
        qp_rs = [r for r in method_results[m] if r.get("quality_pass")]
        diff_counts = Counter(r.get("predicted_difficulty", "judge_error") for r in qp_rs)
        easy = diff_counts.get("Easy", 0)
        med = diff_counts.get("Medium", 0)
        hard = diff_counts.get("Hard", 0)
        jerr = diff_counts.get("judge_error", 0)
        lines.append(f"| {m} | {easy} | {med} | {hard} | {jerr} | {len(qp_rs)} |")
    lines.append("")

    # 3b. Difficulty judge status by method
    lines.append("### 3b. Difficulty Judge Status by Method")
    lines.append("")
    lines.append("| Method | judge_ok | judge_error | Total | Error rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for m in methods:
        qp_rs = [r for r in method_results[m] if r.get("quality_pass")]
        ok_count = sum(1 for r in qp_rs if r.get("difficulty_judge_status") == "ok")
        err_count = sum(1 for r in qp_rs if r.get("difficulty_judge_status") != "ok")
        total = len(qp_rs)
        rate = f"{100 * err_count / total:.1f}%" if total else "N/A"
        lines.append(f"| {m} | {ok_count} | {err_count} | {total} | {rate} |")
    lines.append("")

    # 4. Hard hit rate (denominator: quality_pass AND judge_status=ok)
    lines.append("## 4. Hard Hit Rate by Method")
    lines.append("")
    lines.append("Denominator: quality-pass AND difficulty_judge_status=ok")
    lines.append("")
    lines.append("| Method | Hard hit | Wilson 95% CI |")
    lines.append("|---|---|---|")
    for m in methods:
        qp_ok_rs = [r for r in method_results[m]
                    if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
        hard_count = sum(1 for r in qp_ok_rs if r.get("predicted_difficulty") == "Hard")
        total = len(qp_ok_rs)
        lines.append(f"| {m} | {_wilson_str(hard_count, total)} | |")
    lines.append("")

    # 5. Evidence dependency (judge-ok only)
    lines.append("## 5. Evidence Dependency by Method (quality-pass, judge-ok only)")
    lines.append("")
    lines.append("| Method | alone_sufficient=no | bridge_required=yes | removal in {ambiguous,unanswerable} | Total |")
    lines.append("|---|---:|---:|---:|---:|")
    for m in methods:
        qp_ok_rs = [r for r in method_results[m]
                    if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
        dj_key = "difficulty_judge"
        alone_no = sum(1 for r in qp_ok_rs if r.get(dj_key, {}).get("answer_sentence_alone_sufficient") == "no")
        bridge_yes = sum(1 for r in qp_ok_rs if r.get(dj_key, {}).get("bridge_required") == "yes")
        removal_good = sum(1 for r in qp_ok_rs if r.get(dj_key, {}).get("bridge_removal_effect") in ("ambiguous", "unanswerable"))
        total = len(qp_ok_rs)
        lines.append(f"| {m} | {alone_no} | {bridge_yes} | {removal_good} | {total} |")
    lines.append("")

    # 5b. Evidence coverage by method (quality-pass, judge-ok only)
    lines.append("### 5b. Target Evidence Coverage by Method (quality-pass, judge-ok only)")
    lines.append("")
    lines.append("| Method | mean coverage | coverage>=0.67 | uses_all_target | Total |")
    lines.append("|---|---:|---:|---:|---:|")
    for m in methods:
        qp_ok_rs = [r for r in method_results[m]
                    if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
        if qp_ok_rs:
            coverages = [r.get("target_evidence_coverage", 0.0) for r in qp_ok_rs]
            mean_cov = sum(coverages) / len(coverages)
            high_cov = sum(1 for c in coverages if c >= 0.67)
            uses_all = sum(1 for r in qp_ok_rs
                          if r.get("evidence_coverage", {}).get("uses_all_target_required_sentences") == "yes")
            total = len(qp_ok_rs)
            lines.append(f"| {m} | {mean_cov:.3f} | {high_cov} | {uses_all} | {total} |")
        else:
            lines.append(f"| {m} | N/A | 0 | 0 | 0 |")
    lines.append("")

    # 5c. Hard realization pass by method (exact-id diagnostic)
    lines.append("### 5c. Hard Realization Pass by Method (exact-id diagnostic)")
    lines.append("")
    lines.append("Hard realization (legacy) = judge_ok AND num_judge_used>=3 AND uses_bridge in {yes,partial} AND coverage>=0.67 AND predicted=Hard")
    lines.append("")
    lines.append("| Method | hard_realization_pass | quality-pass judge-ok | Rate |")
    lines.append("|---|---:|---:|---:|")
    for m in methods:
        qp_ok_rs = [r for r in method_results[m]
                    if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
        hrp = sum(1 for r in qp_ok_rs if r.get("hard_realization_pass") == "yes")
        total = len(qp_ok_rs)
        rate = f"{100 * hrp / total:.1f}%" if total else "N/A"
        lines.append(f"| {m} | {hrp} | {total} | {rate} |")
    lines.append("")

    # 5e. Hard realization pass v2 by method (with Wilson CI)
    lines.append("### 5e. Hard Realization Pass v2 by Method")
    lines.append("")
    lines.append("Denominator: quality-pass AND difficulty_judge_status=ok")
    lines.append("")
    lines.append("hrp_v2 = predicted=Hard AND num_judge_used>=3 AND bridge_required=yes AND alone_sufficient=no AND semantic_evidence_match in {yes,partial}")
    lines.append("")
    lines.append("| Method | hrp_v2 | Wilson 95% CI |")
    lines.append("|---|---|---|")
    for m in methods:
        qp_ok_rs = [r for r in method_results[m]
                    if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
        hrp2 = sum(1 for r in qp_ok_rs if r.get("hard_realization_pass_v2") == "yes")
        total = len(qp_ok_rs)
        lines.append(f"| {m} | {_wilson_str(hrp2, total)} | |")
    lines.append("")

    # 5e2. Strict hrp_v2 by method (with Wilson CI)
    lines.append("### 5e2. Strict HRP-v2 by Method")
    lines.append("")
    lines.append("strict_hrp_v2 = hard_realization_pass_v2=yes AND strict_quality_pass=true AND focus_match=yes")
    lines.append("")
    lines.append("| Method | strict_hrp_v2 | Wilson 95% CI |")
    lines.append("|---|---|---|")
    for m in methods:
        qp_ok_rs = [r for r in method_results[m]
                    if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
        shrp2 = sum(1 for r in qp_ok_rs if r.get("strict_hrp_v2") == "yes")
        total = len(qp_ok_rs)
        lines.append(f"| {m} | {_wilson_str(shrp2, total)} | |")
    lines.append("")

    # 5f. Semantic evidence match by method
    lines.append("### 5f. Semantic Evidence Match by Method (quality-pass, judge-ok)")
    lines.append("")
    lines.append("| Method | yes | partial | no | judge_error | Total |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m in methods:
        qp_ok_rs = [r for r in method_results[m]
                    if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
        sem_counts = Counter(r.get("semantic_evidence_match", "judge_error") for r in qp_ok_rs)
        total = len(qp_ok_rs)
        lines.append(f"| {m} | {sem_counts.get('yes', 0)} | {sem_counts.get('partial', 0)} | {sem_counts.get('no', 0)} | {sem_counts.get('judge_error', 0)} | {total} |")
    lines.append("")

    # 5d. Answer focus diagnostics (Ours only)
    lines.append("### 5d. Answer Focus Diagnostics (Ours)")
    lines.append("")
    ours_qp_ok_all = [r for r in method_results["Ours"]
                      if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
    if ours_qp_ok_all:
        # Focus distribution
        focus_counts = Counter(r.get("question_focus", "unknown") for r in ours_qp_ok_all)
        lines.append("#### Question focus distribution")
        lines.append("")
        lines.append("| Focus | Count | Pct |")
        lines.append("|---|---:|---:|")
        for focus, count in focus_counts.most_common():
            pct = f"{100 * count / len(ours_qp_ok_all):.1f}%"
            lines.append(f"| {focus} | {count} | {pct} |")
        lines.append("")

        # Focus match rate
        focus_match_yes = sum(1 for r in ours_qp_ok_all if r.get("focus_match") == "yes")
        focus_match_no = sum(1 for r in ours_qp_ok_all if r.get("focus_match") == "no")
        focus_match_unk = sum(1 for r in ours_qp_ok_all if r.get("focus_match") not in ("yes", "no"))
        lines.append("#### Focus match rate")
        lines.append("")
        lines.append(f"- focus_match=yes: {focus_match_yes} / {len(ours_qp_ok_all)}")
        lines.append(f"- focus_match=no: {focus_match_no} / {len(ours_qp_ok_all)}")
        if focus_match_unk:
            lines.append(f"- focus_match=unknown: {focus_match_unk}")
        lines.append("")

        # Focus by answer role
        role_focus = {}
        for r in ours_qp_ok_all:
            role = r.get("answer_role", "unknown")
            focus = r.get("question_focus", "unknown")
            role_focus.setdefault(role, Counter())[focus] += 1
        lines.append("#### Answer role -> question focus mapping")
        lines.append("")
        lines.append("| answer_role | question_focus | count |")
        lines.append("|---|---|---:|")
        for role, fcounts in sorted(role_focus.items()):
            for focus, count in fcounts.most_common():
                lines.append(f"| {role} | {focus} | {count} |")
        lines.append("")

        # Focus mismatch examples
        focus_mismatch = [r for r in ours_qp_ok_all if r.get("focus_match") == "no"]
        if focus_mismatch:
            lines.append("#### Focus mismatch examples")
            lines.append("")
            for i, r in enumerate(focus_mismatch[:3]):
                lines.append(f"**Mismatch {i+1}:**")
                lines.append(f"- Story: {r.get('story_name', '?')}")
                lines.append(f"- Question: {r.get('generated_question', '?')}")
                lines.append(f"- Target answer: {r.get('target_answer', '?')}")
                lines.append(f"- answer_role={r.get('answer_role', '?')}, question_focus={r.get('question_focus', '?')}")
                lines.append("")

        # Per-focus metrics (all Ours, not just quality-pass)
        ours_all = method_results["Ours"]
        if ours_all:
            lines.append("#### Per question_focus metrics (all Ours)")
            lines.append("")
            lines.append("| Focus | N | quality_pass | Hard | hrp_v2 | strict_hrp_v2 | focus_match=yes |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            all_focuses = sorted(set(r.get("question_focus", "unknown") for r in ours_all))
            for focus in all_focuses:
                fr = [r for r in ours_all if r.get("question_focus") == focus]
                n = len(fr)
                qp = sum(1 for r in fr if r.get("quality_pass"))
                hard = sum(1 for r in fr if r.get("quality_pass") and r.get("predicted_difficulty") == "Hard")
                hrp2 = sum(1 for r in fr if r.get("quality_pass") and r.get("hard_realization_pass_v2") == "yes")
                shrp2 = sum(1 for r in fr if r.get("quality_pass") and r.get("strict_hrp_v2") == "yes")
                fm = sum(1 for r in fr if r.get("focus_match") == "yes")
                lines.append(f"| {focus} | {n} | {qp} | {hard} | {hrp2} | {shrp2} | {fm} |")
            lines.append("")

        # Per answer_role metrics (all Ours)
        if ours_all:
            lines.append("#### Per answer_role metrics (all Ours)")
            lines.append("")
            lines.append("| answer_role | N | quality_pass | Hard | hrp_v2 | strict_hrp_v2 |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            all_roles = sorted(set(r.get("answer_role", "unknown") for r in ours_all))
            for role in all_roles:
                rr = [r for r in ours_all if r.get("answer_role") == role]
                n = len(rr)
                qp = sum(1 for r in rr if r.get("quality_pass"))
                hard = sum(1 for r in rr if r.get("quality_pass") and r.get("predicted_difficulty") == "Hard")
                hrp2 = sum(1 for r in rr if r.get("quality_pass") and r.get("hard_realization_pass_v2") == "yes")
                shrp2 = sum(1 for r in rr if r.get("quality_pass") and r.get("strict_hrp_v2") == "yes")
                lines.append(f"| {role} | {n} | {qp} | {hard} | {hrp2} | {shrp2} |")
            lines.append("")

        # Per answer_node_type metrics (all Ours)
        if ours_all:
            lines.append("#### Per answer_node_type metrics (all Ours)")
            lines.append("")
            lines.append("| node_type | N | quality_pass | Hard | hrp_v2 | strict_hrp_v2 |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            all_ntypes = sorted(set(r.get("answer_node_type", "unknown") for r in ours_all))
            for ntype in all_ntypes:
                nr = [r for r in ours_all if r.get("answer_node_type") == ntype]
                n = len(nr)
                qp = sum(1 for r in nr if r.get("quality_pass"))
                hard = sum(1 for r in nr if r.get("quality_pass") and r.get("predicted_difficulty") == "Hard")
                hrp2 = sum(1 for r in nr if r.get("quality_pass") and r.get("hard_realization_pass_v2") == "yes")
                shrp2 = sum(1 for r in nr if r.get("quality_pass") and r.get("strict_hrp_v2") == "yes")
                lines.append(f"| {ntype} | {n} | {qp} | {hard} | {hrp2} | {shrp2} |")
            lines.append("")
    else:
        lines.append("No Ours quality-pass judge-ok results.")
        lines.append("")

    # 5g. Unique story counts and cluster diagnostic
    lines.append("### 5g. Unique Story Diversity and Cluster Diagnostic")
    lines.append("")
    lines.append("#### Unique stories among predicted Hard (quality-pass, judge-ok)")
    lines.append("")
    lines.append("| Method | unique stories | Hard count | stories |")
    lines.append("|---|---:|---:|---|")
    for m in methods:
        qp_ok_rs = [r for r in method_results[m]
                    if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
        hard_rs = [r for r in qp_ok_rs if r.get("predicted_difficulty") == "Hard"]
        stories = set(r.get("story_name", "") for r in hard_rs)
        story_list = ", ".join(sorted(stories))
        lines.append(f"| {m} | {len(stories)} | {len(hard_rs)} | {story_list} |")
    lines.append("")

    lines.append("#### Unique stories among hrp_v2 (quality-pass, judge-ok)")
    lines.append("")
    lines.append("| Method | unique stories | hrp_v2 count | stories |")
    lines.append("|---|---:|---:|---|")
    for m in methods:
        qp_ok_rs = [r for r in method_results[m]
                    if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
        hrp2_rs = [r for r in qp_ok_rs if r.get("hard_realization_pass_v2") == "yes"]
        stories = set(r.get("story_name", "") for r in hrp2_rs)
        story_list = ", ".join(sorted(stories))
        lines.append(f"| {m} | {len(stories)} | {len(hrp2_rs)} | {story_list} |")
    lines.append("")

    lines.append("#### Unique stories among strict_hrp_v2 (quality-pass, judge-ok)")
    lines.append("")
    lines.append("| Method | unique stories | strict_hrp_v2 count | stories |")
    lines.append("|---|---:|---:|---|")
    for m in methods:
        qp_ok_rs = [r for r in method_results[m]
                    if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
        shrp2_rs = [r for r in qp_ok_rs if r.get("strict_hrp_v2") == "yes"]
        stories = set(r.get("story_name", "") for r in shrp2_rs)
        story_list = ", ".join(sorted(stories))
        lines.append(f"| {m} | {len(stories)} | {len(shrp2_rs)} | {story_list} |")
    lines.append("")

    # Cluster diagnostic: three-dogs concentration
    lines.append("#### Cluster diagnostic: three-dogs concentration")
    lines.append("")
    lines.append("| Method | three-dogs in Hard | total Hard | three-dogs in hrp_v2 | total hrp_v2 | three-dogs in strict_hrp_v2 | total strict_hrp_v2 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for m in methods:
        qp_ok_rs = [r for r in method_results[m]
                    if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
        hard_rs = [r for r in qp_ok_rs if r.get("predicted_difficulty") == "Hard"]
        hrp2_rs = [r for r in qp_ok_rs if r.get("hard_realization_pass_v2") == "yes"]
        shrp2_rs = [r for r in qp_ok_rs if r.get("strict_hrp_v2") == "yes"]
        td_hard = sum(1 for r in hard_rs if r.get("story_name") == "three-dogs")
        td_hrp2 = sum(1 for r in hrp2_rs if r.get("story_name") == "three-dogs")
        td_shrp2 = sum(1 for r in shrp2_rs if r.get("story_name") == "three-dogs")
        lines.append(f"| {m} | {td_hard} | {len(hard_rs)} | {td_hrp2} | {len(hrp2_rs)} | {td_shrp2} | {len(shrp2_rs)} |")
    lines.append("")

    # 6. Failure reasons
    lines.append("## 6. Failure Reasons by Method")
    lines.append("")
    lines.append("| Method | Failure reason | Count |")
    lines.append("|---|---|---:|")
    for m in methods:
        fails = [r for r in method_results[m] if not r.get("quality_pass")]
        reasons = Counter()
        for r in fails:
            qj = r.get("quality_judge", {})
            if qj.get("answerable") == "no":
                reasons["not answerable"] += 1
            elif qj.get("asks_expected_answer") == "no":
                reasons["wrong answer"] += 1
            elif qj.get("answer_leakage") == "yes":
                reasons["answer leakage"] += 1
            elif qj.get("fluency") == "no":
                reasons["not fluent"] += 1
            elif r.get("generation_error"):
                reasons[f"gen error: {r['generation_error'][:30]}"] += 1
            else:
                reasons["other"] += 1
        for reason, count in reasons.most_common():
            lines.append(f"| {m} | {reason} | {count} |")
    lines.append("")

    # 6b. Difficulty judge parse failures
    parse_failures = [r for r in all_results if r.get("difficulty_judge_status") != "ok"]
    if parse_failures:
        lines.append("### 6b. Difficulty Judge Parse Failures")
        lines.append("")
        lines.append(f"Total parse failures: {len(parse_failures)} / {len(all_results)}")
        lines.append("")
        for pf in parse_failures[:10]:
            m = pf.get("method", "?")
            story = pf.get("story_name", "?")[:30]
            q = pf.get("generated_question", "(empty)")[:60]
            dj = pf.get("difficulty_judge", {})
            raw = dj.get("difficulty_judge_raw", "")[:300]
            lines.append(f"- **{m}** / {story}")
            lines.append(f"  - Question: {q}")
            lines.append(f"  - Raw: `{raw}`")
            lines.append("")
    else:
        lines.append("### 6b. Difficulty Judge Parse Failures")
        lines.append("")
        lines.append("None.")
        lines.append("")

    # 7. Copy/reference diagnostics
    lines.append("## 7. Copy/Reference Diagnostics")
    lines.append("")
    lines.append("| Method | Total | Copies source | Copy rate |")
    lines.append("|---|---:|---:|---:|")
    for m in methods:
        rs = method_results[m]
        copies = 0
        for r in rs:
            gen_q = r.get("generated_question", "").strip().lower()
            src_q = r.get("source_question", "").strip().lower()
            if gen_q and src_q and gen_q == src_q:
                copies += 1
        total = len(rs)
        rate = f"{100 * copies / total:.1f}%" if total else "N/A"
        lines.append(f"| {m} | {total} | {copies} | {rate} |")
    lines.append("")

    # 8. Examples
    lines.append("## 8. Examples")
    lines.append("")

    # 3 best Ours (quality_pass, predicted Hard)
    ours_qp_hard = [
        r for r in method_results["Ours"]
        if r.get("quality_pass") and r.get("predicted_difficulty") == "Hard"
    ]
    lines.append("### Best Ours examples (quality-pass, predicted Hard)")
    lines.append("")
    for i, r in enumerate(ours_qp_hard[:3]):
        lines.append(f"**Example {i+1}:**")
        lines.append(f"- Story: {r.get('story_name', '?')}")
        lines.append(f"- Question: {r.get('generated_question', '?')}")
        lines.append(f"- Target answer: {r.get('target_answer', '?')}")
        qj = r.get("quality_judge", {})
        dj = r.get("difficulty_judge", {})
        ec = r.get("evidence_coverage", {})
        lines.append(f"- Quality: answerable={qj.get('answerable')}, asks_expected={qj.get('asks_expected_answer')}, leakage={qj.get('answer_leakage')}")
        lines.append(f"- Difficulty: predicted={dj.get('predicted_difficulty')}, alone_sufficient={dj.get('answer_sentence_alone_sufficient')}, bridge_required={dj.get('bridge_required')}")
        lines.append(f"- Coverage: {ec.get('target_evidence_coverage', 0):.3f}, hard_realization={ec.get('hard_realization_pass', '?')}, hrp_v2={r.get('hard_realization_pass_v2', '?')}")
        lines.append(f"- Focus: answer_role={r.get('answer_role', '?')}, question_focus={r.get('question_focus', '?')}, focus_match={r.get('focus_match', '?')}")
        lines.append(f"- Semantic match: {r.get('semantic_evidence_match', '?')} - {_ascii_safe(r.get('semantic_match_reason', ''))}")
        lines.append("")

    # Hard realization pass v2 examples (quality-pass only)
    hrp2_examples = [r for r in all_results
                     if r.get("hard_realization_pass_v2") == "yes" and r.get("quality_pass")]
    if hrp2_examples:
        lines.append("### Hard realization pass v2 examples")
        lines.append("")
        for i, r in enumerate(hrp2_examples[:5]):
            lines.append(f"**HRP-v2 Example {i+1} ({r.get('method', '?')}):**")
            lines.append(f"- Story: {r.get('story_name', '?')}")
            lines.append(f"- Question: {r.get('generated_question', '?')}")
            lines.append(f"- Target answer: {r.get('target_answer', '?')}")
            dj = r.get("difficulty_judge", {})
            lines.append(f"- Predicted: {dj.get('predicted_difficulty')}, num_used={len(dj.get('required_evidence_sentences_used', []))}, bridge={dj.get('bridge_required')}, alone={dj.get('answer_sentence_alone_sufficient')}")
            lines.append(f"- Semantic match: {r.get('semantic_evidence_match', '?')} - {_ascii_safe(r.get('semantic_match_reason', ''))}")
            lines.append("")

    # Focus match examples (Ours, quality-pass, focus_match=yes)
    ours_focus_ok = [r for r in method_results["Ours"]
                     if r.get("quality_pass") and r.get("focus_match") == "yes"]
    if ours_focus_ok:
        lines.append("### Ours focus-match examples (quality-pass, focus_match=yes)")
        lines.append("")
        for i, r in enumerate(ours_focus_ok[:3]):
            lines.append(f"**Focus Example {i+1}:**")
            lines.append(f"- Story: {r.get('story_name', '?')}")
            lines.append(f"- Question: {r.get('generated_question', '?')}")
            lines.append(f"- Target answer: {r.get('target_answer', '?')}")
            lines.append(f"- answer_role={r.get('answer_role', '?')}, question_focus={r.get('question_focus', '?')}, node_type={r.get('answer_node_type', '?')}")
            lines.append("")

    # 3 best baselines
    baseline_methods = ["Direct", "ICL", "SelfRefine"]
    lines.append("### Best baseline examples (quality-pass)")
    lines.append("")
    shown = 0
    for bm in baseline_methods:
        bm_qp = [r for r in method_results[bm] if r.get("quality_pass")]
        for r in bm_qp[:1]:
            if shown >= 3:
                break
            lines.append(f"**{bm} Example:**")
            lines.append(f"- Story: {r.get('story_name', '?')}")
            lines.append(f"- Question: {r.get('generated_question', '?')}")
            lines.append(f"- Target answer: {r.get('target_answer', '?')}")
            dj = r.get("difficulty_judge", {})
            lines.append(f"- Predicted difficulty: {dj.get('predicted_difficulty', '?')}")
            lines.append("")
            shown += 1

    # Ours failure cases grouped
    lines.append("### Ours failure cases (grouped)")
    lines.append("")
    ours_fails = [r for r in method_results["Ours"] if not r.get("quality_pass")]
    if ours_fails:
        fail_groups = Counter()
        for r in ours_fails:
            q = (r.get("generated_question") or "").strip()
            err = r.get("generation_error", "")
            qj = r.get("quality_judge", {})
            repaired = r.get("repair_success", False)
            if not q or "empty" in (err or "").lower() or "degenerate" in (err or "").lower():
                if repaired:
                    fail_groups["repaired but still failed"] += 1
                else:
                    fail_groups["degenerate / parse failure"] += 1
            elif qj.get("answerable") == "no":
                fail_groups["not answerable"] += 1
            elif qj.get("asks_expected_answer") == "no":
                fail_groups["answer mismatch"] += 1
            elif qj.get("fluency") == "no":
                fail_groups["not fluent"] += 1
            elif r.get("focus_match") == "no":
                fail_groups["focus mismatch"] += 1
            else:
                fail_groups["other"] += 1
        lines.append("| Failure category | Count |")
        lines.append("|---|---:|")
        for cat, count in fail_groups.most_common():
            lines.append(f"| {cat} | {count} |")
        lines.append("")

        # Show up to 3 example failures
        lines.append("#### Ours failure examples")
        lines.append("")
        for i, r in enumerate(ours_fails[:3]):
            q = (r.get("generated_question") or "").strip()
            err = r.get("generation_error", "")
            qj = r.get("quality_judge", {})
            raw = r.get("generation_raw", "")
            lines.append(f"**Failure {i+1}:**")
            lines.append(f"- Story: {r.get('story_name', '?')}")
            lines.append(f"- Question: {q[:80] if q else '(empty)'}")
            if not q and raw:
                lines.append(f"- Raw prefix: `{raw[:200]}`")
            reason = qj.get("reason", err or "unknown")
            lines.append(f"- Reason: {reason}")
            lines.append("")
    else:
        lines.append("No Ours failures.")
        lines.append("")

    # 3 baseline failure cases
    lines.append("### Baseline failure cases")
    lines.append("")
    baseline_fails = [r for r in all_results if not r.get("quality_pass") and r.get("method") != "Ours"]
    for i, r in enumerate(baseline_fails[:3]):
        lines.append(f"**Failure {i+1} ({r.get('method', '?')}):**")
        lines.append(f"- Story: {r.get('story_name', '?')}")
        lines.append(f"- Question: {r.get('generated_question', '(empty)')[:80]}")
        qj = r.get("quality_judge", {})
        lines.append(f"- Reason: {qj.get('reason', r.get('generation_error', 'unknown'))}")
        lines.append("")

    # Success criteria (variables computed first, table printed after pairwise diff)
    ours_results = method_results["Ours"]
    ours_qp = [r for r in ours_results if r.get("quality_pass")]
    ours_qp_ok = [r for r in ours_qp if r.get("difficulty_judge_status") == "ok"]
    ours_total = len(ours_results)
    ours_qp_count = len(ours_qp)
    ours_qp_rate = 100 * ours_qp_count / ours_total if ours_total else 0
    ours_hard = sum(1 for r in ours_qp_ok if r.get("predicted_difficulty") == "Hard")
    ours_hard_rate = 100 * ours_hard / len(ours_qp_ok) if ours_qp_ok else 0
    ours_bridge_yes = sum(
        1 for r in ours_qp
        if r.get("difficulty_judge", {}).get("bridge_required") == "yes"
    )
    ours_bridge_rate = 100 * ours_bridge_yes / ours_qp_count if ours_qp_count else 0

    # Hard hit comparison (judge-ok denominator)
    def hard_hit_rate(method):
        rs = [r for r in method_results[method]
              if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
        if not rs:
            return 0
        return sum(1 for r in rs if r.get("predicted_difficulty") == "Hard") / len(rs)

    ours_hhr = hard_hit_rate("Ours")
    baseline_hhrs = {m: hard_hit_rate(m) for m in baseline_methods}
    ours_beats_all = all(ours_hhr > v for v in baseline_hhrs.values())

    no_copies = True
    for m in methods:
        for r in method_results[m]:
            gen_q = r.get("generated_question", "").strip().lower()
            src_q = r.get("source_question", "").strip().lower()
            if gen_q and src_q and gen_q == src_q:
                no_copies = False

    # Hard realization pass v2 for Ours
    ours_hrp2 = sum(1 for r in ours_qp_ok if r.get("hard_realization_pass_v2") == "yes")
    ours_hrp2_rate = 100 * ours_hrp2 / len(ours_qp_ok) if ours_qp_ok else 0

    # Strict hrp_v2 for Ours
    ours_shrp2 = sum(1 for r in ours_qp_ok if r.get("strict_hrp_v2") == "yes")
    ours_shrp2_rate = 100 * ours_shrp2 / len(ours_qp_ok) if ours_qp_ok else 0

    # Unique story counts for Ours
    ours_hard_stories = set(r.get("story_name", "") for r in ours_qp_ok
                           if r.get("predicted_difficulty") == "Hard")
    ours_hrp2_stories = set(r.get("story_name", "") for r in ours_qp_ok
                            if r.get("hard_realization_pass_v2") == "yes")
    # Unique baseline hrp_v2 stories
    baseline_hrp2_stories = {}
    for bm in baseline_methods:
        bm_qp_ok = [r for r in method_results[bm]
                     if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
        baseline_hrp2_stories[bm] = set(r.get("story_name", "") for r in bm_qp_ok
                                        if r.get("hard_realization_pass_v2") == "yes")
    max_baseline_hrp2_stories = max(len(v) for v in baseline_hrp2_stories.values()) if baseline_hrp2_stories else 0
    ours_unique_hrp2_beats = len(ours_hrp2_stories) > max_baseline_hrp2_stories

    # Ours Hard hit >= each baseline
    ours_hhr_ge_all = all(ours_hhr >= v for v in baseline_hhrs.values())

    # Pairwise difference table (Ours vs each baseline)
    lines.append("## 8b. Pairwise Difference Table (Ours - Baseline)")
    lines.append("")
    lines.append("| Metric | Ours | " + " | ".join(baseline_methods) + " | " +
                 " | ".join(f"Ours-{m}" for m in baseline_methods) + " |")
    lines.append("|---|---" + "|---:" * len(baseline_methods) + "|---" * len(baseline_methods) + "|")

    # quality_pass rate
    ours_qp_pct = 100 * ours_qp_count / ours_total if ours_total else 0
    row = f"| quality_pass | {ours_qp_pct:.1f}% ({ours_qp_count}/{ours_total})"
    diffs = []
    for bm in baseline_methods:
        bm_total = len(method_results[bm])
        bm_qp = sum(1 for r in method_results[bm] if r.get("quality_pass"))
        bm_pct = 100 * bm_qp / bm_total if bm_total else 0
        row += f" | {bm_pct:.1f}% ({bm_qp}/{bm_total})"
        diffs.append(f"{ours_qp_pct - bm_pct:+.1f}pp")
    row += " | " + " | ".join(diffs) + " |"
    lines.append(row)

    # Hard hit rate (quality-pass, judge-ok)
    row = f"| Hard hit | {100*ours_hhr:.1f}% ({ours_hard}/{len(ours_qp_ok)})"
    diffs = []
    for bm in baseline_methods:
        bm_qp_ok = [r for r in method_results[bm]
                     if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
        bm_hard = sum(1 for r in bm_qp_ok if r.get("predicted_difficulty") == "Hard")
        bm_rate = 100 * bm_hard / len(bm_qp_ok) if bm_qp_ok else 0
        row += f" | {bm_rate:.1f}% ({bm_hard}/{len(bm_qp_ok)})"
        diffs.append(f"{100*ours_hhr - bm_rate:+.1f}pp")
    row += " | " + " | ".join(diffs) + " |"
    lines.append(row)

    # HRP-v2 rate
    row = f"| HRP-v2 | {ours_hrp2_rate:.1f}% ({ours_hrp2}/{len(ours_qp_ok)})"
    diffs = []
    for bm in baseline_methods:
        bm_qp_ok = [r for r in method_results[bm]
                     if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
        bm_hrp2 = sum(1 for r in bm_qp_ok if r.get("hard_realization_pass_v2") == "yes")
        bm_rate = 100 * bm_hrp2 / len(bm_qp_ok) if bm_qp_ok else 0
        row += f" | {bm_rate:.1f}% ({bm_hrp2}/{len(bm_qp_ok)})"
        diffs.append(f"{ours_hrp2_rate - bm_rate:+.1f}pp")
    row += " | " + " | ".join(diffs) + " |"
    lines.append(row)

    # Unique HRP-v2 stories
    row = f"| unique HRP-v2 stories | {len(ours_hrp2_stories)}"
    diffs = []
    for bm in baseline_methods:
        bm_count = len(baseline_hrp2_stories.get(bm, set()))
        row += f" | {bm_count}"
        diffs.append(f"{len(ours_hrp2_stories) - bm_count:+d}")
    row += " | " + " | ".join(diffs) + " |"
    lines.append(row)

    lines.append("")

    # Success criteria header
    lines.append("## 9. Success Criteria")
    lines.append("")

    criteria = [
        ("Ours quality_pass >= 65%", ours_qp_rate >= 65,
         f"{ours_qp_rate:.1f}% ({_wilson_str(ours_qp_count, ours_total)})"),
        ("Ours predicted Hard >= 25%", ours_hard_rate >= 25,
         f"{ours_hard_rate:.1f}% ({_wilson_str(ours_hard, len(ours_qp_ok))})"),
        ("Ours eval_hrp_v2 >= 25% (quality-pass, judge-ok)", ours_hrp2_rate >= 25,
         f"{ours_hrp2_rate:.1f}% ({_wilson_str(ours_hrp2, len(ours_qp_ok))})"),
        ("Ours strict_hrp_v2 >= 10%", ours_shrp2_rate >= 10,
         f"{ours_shrp2_rate:.1f}% ({_wilson_str(ours_shrp2, len(ours_qp_ok))})"),
        ("Ours unique HRP-v2 stories > each baseline", ours_unique_hrp2_beats,
         f"Ours={len(ours_hrp2_stories)}, " + ", ".join(
             f"{m}={len(v)}" for m, v in baseline_hrp2_stories.items())),
        ("Ours Hard hit >= Direct/ICL/SelfRefine", ours_hhr_ge_all,
         f"Ours={ours_hhr:.2f}, " + ", ".join(f"{m}={v:.2f}" for m, v in baseline_hhrs.items())),
    ]

    lines.append("| Criterion | Status | Value |")
    lines.append("|---|---|---|")
    all_pass = True
    for name, passed, value in criteria:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        lines.append(f"| {name} | {status} | {value} |")
    lines.append("")
    lines.append(f"**Overall: {'ALL CRITERIA PASS' if all_pass else 'SOME CRITERIA FAILED'}**")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nReport: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="FairytaleQA Hard QG Pilot")
    parser.add_argument("--graphs", required=True, help="Path to graphs.jsonl")
    parser.add_argument("--candidates", default=None,
                        help="Path to candidates.jsonl (for story_section). Auto-detected if not given.")
    parser.add_argument("--limit", type=int, default=15, help="Max candidates to process")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading graphs...")
    graphs = _load_graphs(args.graphs)
    print(f"  Loaded {len(graphs)} graphs")

    # Auto-detect candidates path if not given
    candidates_path = args.candidates
    if not candidates_path:
        # Try to find it from graph metadata
        meta = graphs[0] if graphs else {}
        # Common paths
        for p in [
            "outputs/runs/fairytale_evidence_audit_train_implicit_500_20260510/candidates.jsonl",
            "outputs/runs/fairytale_evidence_audit_val_implicit_200_20260510/candidates.jsonl",
        ]:
            if Path(p).exists():
                candidates_path = p
                break

    if not candidates_path or not Path(candidates_path).exists():
        print("ERROR: candidates.jsonl not found. Provide --candidates path.")
        return

    print(f"Loading candidates from {candidates_path}...")
    candidates_lookup = _load_candidates(candidates_path)
    print(f"  Loaded {len(candidates_lookup)} candidates")

    # Join
    joined = _join_graphs_with_candidates(graphs, candidates_lookup)
    print(f"  Joined: {len(joined)} graphs with story_section")

    # Select
    selected = _select_candidates(joined, args.limit)
    print(f"  Selected: {len(selected)} candidates for pilot")

    if not selected:
        print("ERROR: no candidates selected.")
        return

    # Run all methods
    methods = ["Direct", "ICL", "SelfRefine", "Ours"]
    all_results = []

    gen_path = output_dir / "generations.raw.jsonl"
    judged_path = output_dir / "generations.judged.jsonl"

    print(f"\n=== Running QG Pilot ({len(selected)} candidates x {len(methods)} methods) ===\n")

    with open(gen_path, "w", encoding="utf-8") as gen_f:
        for i, cand in enumerate(selected):
            story = cand.get("story_name", "?")[:30]
            print(f"[{i+1}/{len(selected)}] {story}...")

            for method in methods:
                t0 = time.time()
                result = _run_method(method, cand)
                elapsed = time.time() - t0

                # Enrich with candidate metadata
                result["story_name"] = cand.get("story_name", "")
                result["story_section"] = cand.get("story_section", "")
                result["target_answer"] = cand.get("answer", "") or cand.get("answer1", "")
                result["target_difficulty"] = "Hard"
                result["source_question"] = cand.get("original_question", "")
                result["attribute"] = cand.get("attribute", "")
                result["reasoning_operation"] = cand.get("reasoning_operation", "")
                result["necessity_type"] = cand.get("necessity_type", "")
                result["required_evidence_sentences"] = cand.get("required_evidence_sentences", [])
                result["bridge_sentence_ids"] = cand.get("bridge_sentence_ids", [])
                result["graph_valid"] = cand.get("graph_valid", False)
                result["elapsed_seconds"] = round(elapsed, 1)

                # Write generation (before judging)
                gen_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                gen_f.flush()

                # Judge
                result = _judge_generation(result, cand)
                all_results.append(result)

                qp = "QP" if result.get("quality_pass") else "FAIL"
                pd = result.get("predicted_difficulty", "?")
                print(f"  {method}: {qp} pred={pd} ({elapsed:.1f}s)")

                time.sleep(0.1)

    # Write judged results
    with open(judged_path, "w", encoding="utf-8") as f:
        for r in all_results:
            # Remove large fields for judged output
            r_copy = dict(r)
            r_copy.pop("generation_prompt", None)
            r_copy.pop("quality_judge", None)
            r_copy.pop("difficulty_judge", None)
            f.write(json.dumps(r_copy, ensure_ascii=False) + "\n")

    # Write full judged results (with judge details)
    judged_full_path = output_dir / "generations.judged.full.jsonl"
    with open(judged_full_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Report
    graph_total = len(graphs)
    graph_valid = sum(1 for g in graphs if g.get("graph_valid") is True)
    meta = {
        "requested_limit": args.limit,
        "graph_total": graph_total,
        "graph_valid": graph_valid,
        "selected_candidates": len(selected),
    }
    _build_report(all_results, output_dir, meta=meta)

    # Summary
    print("\n=== Summary ===")
    for m in methods:
        rs = [r for r in all_results if r.get("method") == m]
        qp = sum(1 for r in rs if r.get("quality_pass"))
        hard = sum(1 for r in rs if r.get("quality_pass") and r.get("predicted_difficulty") == "Hard")
        print(f"  {m}: quality_pass={qp}/{len(rs)}, predicted Hard={hard}")


if __name__ == "__main__":
    main()
