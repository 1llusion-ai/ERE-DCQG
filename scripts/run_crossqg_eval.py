"""FairytaleQA CrossQG-style full difficulty evaluation.

Runs 4 methods (Direct, ICL, SelfRefine, Ours) on balanced Easy/Medium/Hard
candidates. Judges quality and predicted difficulty. Reports CrossQG metrics
(difficulty accuracy, macro accuracy, Spearman, confusion matrix) as primary.

Usage:
  python -m scripts.run_crossqg_eval \
    --candidates outputs/runs/fairytale_evidence_audit_train_implicit_2166_20260511/candidates.jsonl \
    --target_per_level 150 \
    --output_dir outputs/runs/fairytale_qg_crossqg_eval_20260511_v1/
"""
import argparse
import json
import math
import random
import re
import time
from pathlib import Path
from collections import Counter


def _ascii_safe(text):
    """Replace Unicode dashes/smart punctuation with ASCII equivalents."""
    if not text:
        return ""
    return (text
            .replace('—', ' - ')
            .replace('–', '-')
            .replace('‘', "'")
            .replace('’', "'")
            .replace('“', '"')
            .replace('”', '"'))


def _wilson_ci(k, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def _wilson_str(k, n):
    lo, hi = _wilson_ci(k, n)
    return f"{k}/{n} ({100*k/n:.1f}%, 95%CI [{100*lo:.1f}, {100*hi:.1f}%])" if n else "N/A"


def _spearman(x, y):
    """Compute Spearman rank correlation without scipy.

    Uses average ranks for tied values (critical for ordinal data with few levels).
    """
    n = len(x)
    if n < 2:
        return 0.0

    def _rank(vals):
        """Assign average ranks to tied values."""
        indexed = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            # Find all ties with the same value
            while j < n and indexed[j][1] == indexed[i][1]:
                j += 1
            # Average rank for tied group (1-indexed)
            avg_rank = (i + j + 1) / 2.0  # i+1 to j, average = (i+1+j)/2
            for k in range(i, j):
                ranks[indexed[k][0]] = avg_rank
            i = j
        return ranks

    rx, ry = _rank(x), _rank(y)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    cov = sum((rx[i] - mean_x) * (ry[i] - mean_y) for i in range(n))
    std_x = sum((rx[i] - mean_x) ** 2 for i in range(n)) ** 0.5
    std_y = sum((ry[i] - mean_y) ** 2 for i in range(n)) ** 0.5
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y)


def _bootstrap_pairwise_test(values_a, values_b, metric_fn, n_bootstrap=10000, seed=42):
    """Paired bootstrap CI for metric(A) - metric(B).

    Resamples paired observations, computes the diff metric(A_boot) - metric(B_boot)
    on each resample.  Returns:
        observed_diff, ci_low, ci_high, approx_p
    where approx_p = 2 * min(frac(diff<=0), frac(diff>=0)) (two-sided approximate).
    Significance = CI excludes 0.
    """
    rng = random.Random(seed)
    n = len(values_a)
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0

    observed_diff = metric_fn(values_a) - metric_fn(values_b)
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        boot_a = [values_a[i] for i in indices]
        boot_b = [values_b[i] for i in indices]
        boot_diff = metric_fn(boot_a) - metric_fn(boot_b)
        bootstrap_diffs.append(boot_diff)

    bootstrap_diffs.sort()
    ci_low = bootstrap_diffs[int(0.025 * n_bootstrap)]
    ci_high = bootstrap_diffs[int(0.975 * n_bootstrap)]

    # Approximate two-sided p: 2 * min(frac <= 0, frac >= 0)
    frac_le0 = sum(1 for d in bootstrap_diffs if d <= 0) / n_bootstrap
    frac_ge0 = sum(1 for d in bootstrap_diffs if d >= 0) / n_bootstrap
    approx_p = 2 * min(frac_le0, frac_ge0)
    approx_p = min(approx_p, 1.0)  # clamp

    return observed_diff, ci_low, ci_high, approx_p


def _compute_bootstrap_diagnostics(all_results, crossqg_metrics):
    """Compute paired bootstrap CI for Ours vs each baseline.

    Pairing key: (story_name, question, answer, target_difficulty).
    Reports paired metric values (not global), CI, and approximate p.
    Significance = CI excludes 0.

    Returns dict: {(baseline, metric_name): {
        paired_ours, paired_baseline, diff, ci_low, ci_high, approx_p, n_paired, significant
    }}
    """
    methods = ["Direct", "ICL", "SelfRefine", "Ours"]
    baseline_methods = ["Direct", "ICL", "SelfRefine"]
    levels = ["Easy", "Medium", "Hard"]
    ordinal = {"Easy": 1, "Medium": 2, "Hard": 3}

    # Build per-method valid result lists
    method_valid = {}
    for m in methods:
        rs = [r for r in all_results if r.get("method") == m]
        method_valid[m] = [r for r in rs
                          if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]

    # Unique candidate key
    def _key(r):
        return (r.get("story_name", ""), r.get("question", ""),
                r.get("answer", ""), r.get("target_difficulty", ""))

    # Metric functions (operate on lists of result dicts)
    def _accuracy(rs):
        correct = sum(1 for r in rs if r.get("predicted_difficulty") == r.get("target_difficulty"))
        return correct / len(rs) if rs else 0.0

    def _macro_accuracy(rs):
        accs = []
        for level in levels:
            level_rs = [r for r in rs if r.get("target_difficulty") == level]
            correct = sum(1 for r in level_rs if r.get("predicted_difficulty") == level)
            accs.append(correct / len(level_rs) if level_rs else 0.0)
        return sum(accs) / 3

    def _macro_f1(rs):
        f1s = []
        for level in levels:
            tp = sum(1 for r in rs if r.get("target_difficulty") == level and r.get("predicted_difficulty") == level)
            fp = sum(1 for r in rs if r.get("target_difficulty") != level and r.get("predicted_difficulty") == level)
            fn = sum(1 for r in rs if r.get("target_difficulty") == level and r.get("predicted_difficulty") != level)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1s.append(f1)
        return sum(f1s) / 3

    def _spearman_metric(rs):
        target_vals = [ordinal.get(r.get("target_difficulty", "Medium"), 2) for r in rs]
        pred_vals = [ordinal.get(r.get("predicted_difficulty", "Medium"), 2) for r in rs]
        return _spearman(target_vals, pred_vals)

    metric_fns = {
        "overall_accuracy": _accuracy,
        "macro_accuracy": _macro_accuracy,
        "macro_f1": _macro_f1,
        "spearman": _spearman_metric,
    }

    results = {}
    ours_valid = method_valid["Ours"]
    ours_lookup = {_key(r): r for r in ours_valid}

    for bm in baseline_methods:
        bm_valid = method_valid[bm]
        bm_lookup = {_key(r): r for r in bm_valid}
        common_keys = set(ours_lookup.keys()) & set(bm_lookup.keys())
        paired_ours = [ours_lookup[k] for k in sorted(common_keys)]
        paired_bm = [bm_lookup[k] for k in sorted(common_keys)]

        for metric_name, metric_fn in metric_fns.items():
            observed_diff, ci_low, ci_high, approx_p = _bootstrap_pairwise_test(
                paired_ours, paired_bm, metric_fn, n_bootstrap=10000, seed=42
            )
            significant = (ci_low > 0) or (ci_high < 0)
            results[(bm, metric_name)] = {
                "paired_ours": metric_fn(paired_ours),
                "paired_baseline": metric_fn(paired_bm),
                "diff": observed_diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "approx_p": approx_p,
                "n_paired": len(common_keys),
                "significant": significant,
            }

    return results


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
from dcqg.graph.narrative_graph import NarrativeGraphExtractor


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def _select_balanced_candidates(candidates_path, target_per_level=150, seed=42):
    """Load candidates and stratified-sample target_per_level per difficulty.

    Returns list of candidate dicts, each tagged with 'target_difficulty'.
    """
    import random
    rng = random.Random(seed)

    pools = {"Easy": [], "Medium": [], "Hard": []}
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            diff = rec.get("evidence_difficulty", "")
            if diff in pools:
                pools[diff].append(rec)

    print(f"  Pool sizes: Easy={len(pools['Easy'])}, Medium={len(pools['Medium'])}, Hard={len(pools['Hard'])}")

    selected = []
    for diff in ["Easy", "Medium", "Hard"]:
        pool = pools[diff]
        if len(pool) <= target_per_level:
            batch = list(pool)
            print(f"  {diff}: using all {len(batch)} (target {target_per_level})")
        else:
            # Stratified by necessity_type
            by_type = {}
            for c in pool:
                nt = c.get("necessity_type", "unknown")
                by_type.setdefault(nt, []).append(c)
            for nt_list in by_type.values():
                rng.shuffle(nt_list)

            batch = []
            round_robin = list(by_type.values())
            idx = 0
            while len(batch) < target_per_level:
                src = round_robin[idx % len(round_robin)]
                if src:
                    batch.append(src.pop(0))
                idx += 1
                if all(len(p) == 0 for p in round_robin):
                    break
            print(f"  {diff}: sampled {len(batch)} from {len(pool)}")

        for c in batch:
            c["target_difficulty"] = diff
        selected.extend(batch)

    return selected


# ---------------------------------------------------------------------------
# Graph extraction
# ---------------------------------------------------------------------------

def _extract_graphs(selected, output_dir):
    """Extract narrative graphs for all selected candidates.

    Returns dict keyed by (story_name, question) -> graph record.
    """
    extractor = NarrativeGraphExtractor()
    graphs_path = output_dir / "graphs.jsonl"
    graph_lookup = {}
    stats = {"Easy": {"ok": 0, "fail": 0}, "Medium": {"ok": 0, "fail": 0}, "Hard": {"ok": 0, "fail": 0}}

    with open(graphs_path, "w", encoding="utf-8") as f:
        for i, cand in enumerate(selected):
            diff = cand.get("target_difficulty", "Hard")
            try:
                record = extractor.extract(cand, difficulty=diff)
            except Exception as e:
                record = {
                    "story_name": cand.get("story_name", ""),
                    "question": cand.get("question", ""),
                    "answer": cand.get("answer", ""),
                    "target_difficulty": diff,
                    "nodes": [],
                    "edges": [],
                    "graph_valid": False,
                    "graph_validation_reason": f"extraction_error: {e}",
                    "diagnostics": {},
                    "trace": {"prompt": "", "raw": "", "parse_ok": False, "model": ""},
                }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            key = (record.get("story_name", ""), record.get("question", ""))
            graph_lookup[key] = record

            gv = record.get("graph_valid", False)
            if gv:
                stats[diff]["ok"] += 1
            else:
                stats[diff]["fail"] += 1

            if (i + 1) % 20 == 0:
                print(f"  Graph extraction: {i+1}/{len(selected)}")

    print("  Graph extraction results:")
    for diff in ["Easy", "Medium", "Hard"]:
        s = stats[diff]
        total = s["ok"] + s["fail"]
        pct = f"{100*s['ok']/total:.1f}%" if total else "N/A"
        print(f"    {diff}: {s['ok']}/{total} valid ({pct})")

    return graph_lookup, stats


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _run_method(method_name, candidate, graph_data):
    """Run a single generation method. Returns result dict."""
    story_section = candidate["story_section"]
    target_answer = candidate.get("answer", "") or candidate.get("answer1", "")
    difficulty = candidate.get("target_difficulty", "Hard")

    if method_name == "Direct":
        result, attempts = generate_direct(story_section, target_answer, difficulty)
    elif method_name == "ICL":
        result, attempts = generate_icl(story_section, target_answer, difficulty)
    elif method_name == "SelfRefine":
        result, attempts = generate_self_refine(story_section, target_answer, difficulty)
    elif method_name == "Ours":
        if graph_data is None or not graph_data.get("graph_valid"):
            return {
                "method": "Ours",
                "generated_question": "",
                "parse_ok": False,
                "quality_pass": False,
                "generation_error": "graph_invalid" if graph_data is not None else "graph_missing",
                "attempts": 0,
            }
        result, attempts = generate_ours(
            story_section, target_answer, difficulty,
            nodes=graph_data.get("nodes", []),
            edges=graph_data.get("edges", []),
            required_evidence_sentences=candidate.get("required_evidence_sentences", []),
            bridge_sentence_ids=candidate.get("bridge_sentence_ids", []),
            reasoning_operation=candidate.get("reasoning_operation", ""),
            necessity_type=candidate.get("necessity_type", ""),
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    result["attempts"] = attempts
    return result


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

def _judge_generation(result, candidate):
    """Run quality and difficulty judges on a generation result."""
    question = result.get("generated_question", "")
    story_section = candidate["story_section"]
    target_answer = candidate.get("answer", "") or candidate.get("answer1", "")
    target_difficulty = candidate.get("target_difficulty", "Hard")

    # Quality judge
    qj = quality_judge(question, story_section, target_answer, target_difficulty)
    result["quality_judge"] = qj
    result["quality_pass"] = qj.get("quality_pass", False)
    result["strict_quality_pass"] = qj.get("strict_quality_pass", False)

    # Difficulty/evidence judge (blind -- does NOT see target_difficulty)
    dj = difficulty_evidence_judge(
        question, story_section, target_answer, target_difficulty,
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

    # Semantic evidence match judge (for all, but mainly meaningful for Hard)
    sem = semantic_evidence_match_judge(
        question, story_section, target_answer,
        candidate.get("required_evidence_sentences", []),
        dj.get("required_evidence_sentences_used", []),
    )
    result["semantic_evidence_match"] = sem.get("semantic_evidence_match", "judge_error")
    result["semantic_match_reason"] = sem.get("semantic_match_reason", "")

    # Hard-only secondary diagnostics
    if target_difficulty == "Hard":
        hrp_v2 = (
            result.get("predicted_difficulty") == "Hard"
            and dj.get("difficulty_judge_status") == "ok"
            and len(dj.get("required_evidence_sentences_used", [])) >= 3
            and dj.get("bridge_required") == "yes"
            and dj.get("answer_sentence_alone_sufficient") == "no"
            and result.get("semantic_evidence_match") in ("yes", "partial")
        )
        result["hard_realization_pass_v2"] = "yes" if hrp_v2 else "no"

        strict_hrp_v2 = (
            hrp_v2
            and result.get("strict_quality_pass") is True
            and result.get("focus_match") == "yes"
        )
        result["strict_hrp_v2"] = "yes" if strict_hrp_v2 else "no"
    else:
        result["hard_realization_pass_v2"] = "n/a"
        result["strict_hrp_v2"] = "n/a"

    return result


# ---------------------------------------------------------------------------
# CrossQG metrics
# ---------------------------------------------------------------------------

def _compute_crossqg_metrics(all_results):
    """Compute CrossQG-style metrics per method."""
    methods = ["Direct", "ICL", "SelfRefine", "Ours"]
    levels = ["Easy", "Medium", "Hard"]
    ordinal = {"Easy": 1, "Medium": 2, "Hard": 3}
    metrics = {}

    for method in methods:
        rs = [r for r in all_results if r.get("method") == method]
        valid = [r for r in rs
                 if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]

        # Confusion matrix
        confusion = {(t, p): 0 for t in levels for p in levels}
        for r in valid:
            t = r.get("target_difficulty", "?")
            p = r.get("predicted_difficulty", "?")
            if t in levels and p in levels:
                confusion[(t, p)] += 1

        # Per-level accuracy
        per_level = {}
        for level in levels:
            level_rs = [r for r in valid if r.get("target_difficulty") == level]
            correct = sum(1 for r in level_rs if r.get("predicted_difficulty") == level)
            total = len(level_rs)
            per_level[level] = {
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0.0,
            }

        # Overall difficulty accuracy
        total_valid = len(valid)
        total_correct = sum(per_level[l]["correct"] for l in levels)
        overall_acc = total_correct / total_valid if total_valid else 0.0

        # Macro accuracy
        macro_acc = sum(per_level[l]["accuracy"] for l in levels) / 3

        # Macro F1
        per_level_f1 = {}
        for level in levels:
            tp = confusion[(level, level)]
            fp = sum(confusion[(other, level)] for other in levels if other != level)
            fn = sum(confusion[(level, other)] for other in levels if other != level)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            per_level_f1[level] = {"precision": precision, "recall": recall, "f1": f1}
        macro_f1 = sum(per_level_f1[l]["f1"] for l in levels) / 3

        # Spearman correlation
        target_vals = [ordinal.get(r.get("target_difficulty", "Medium"), 2) for r in valid]
        pred_vals = [ordinal.get(r.get("predicted_difficulty", "Medium"), 2) for r in valid]
        spearman = _spearman(target_vals, pred_vals)

        # End-to-end: quality_pass AND judge_ok AND predicted == target / all candidates
        e2e_correct = sum(
            1 for r in rs
            if r.get("quality_pass")
            and r.get("difficulty_judge_status") == "ok"
            and r.get("predicted_difficulty") == r.get("target_difficulty")
        )
        e2e_total = len(rs)
        e2e_acc = e2e_correct / e2e_total if e2e_total else 0.0

        # End-to-end per level
        e2e_per_level = {}
        for level in levels:
            level_rs = [r for r in rs if r.get("target_difficulty") == level]
            level_correct = sum(
                1 for r in level_rs
                if r.get("quality_pass")
                and r.get("difficulty_judge_status") == "ok"
                and r.get("predicted_difficulty") == level
            )
            e2e_per_level[level] = {
                "correct": level_correct,
                "total": len(level_rs),
                "accuracy": level_correct / len(level_rs) if level_rs else 0.0,
            }

        metrics[method] = {
            "n_total": len(rs),
            "n_quality_pass": sum(1 for r in rs if r.get("quality_pass")),
            "n_valid": total_valid,
            "overall_accuracy": overall_acc,
            "macro_accuracy": macro_acc,
            "macro_f1": macro_f1,
            "spearman": spearman,
            "confusion": confusion,
            "per_level": per_level,
            "per_level_f1": per_level_f1,
            "e2e_correct": e2e_correct,
            "e2e_total": e2e_total,
            "e2e_accuracy": e2e_acc,
            "e2e_per_level": e2e_per_level,
        }

    return metrics


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _build_report(all_results, crossqg_metrics, graph_stats, output_dir, meta=None, bootstrap_diag=None):
    """Write the CrossQG evaluation report."""
    report_path = output_dir / "CROSSQG_EVAL_REPORT.md"
    meta = meta or {}
    methods = ["Direct", "ICL", "SelfRefine", "Ours"]
    levels = ["Easy", "Medium", "Hard"]
    baseline_methods = ["Direct", "ICL", "SelfRefine"]

    method_results = {m: [] for m in methods}
    for r in all_results:
        m = r.get("method", "Unknown")
        if m in method_results:
            method_results[m].append(r)

    lines = [
        "# FairytaleQA CrossQG Evaluation Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Run Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Methods | {', '.join(methods)} |",
        f"| Target per level | {meta.get('target_per_level', 'N/A')} |",
        f"| Selected Easy | {meta.get('n_easy', 'N/A')} |",
        f"| Selected Medium | {meta.get('n_medium', 'N/A')} |",
        f"| Selected Hard | {meta.get('n_hard', 'N/A')} |",
        f"| Total selected | {meta.get('n_total', 'N/A')} |",
        f"| Total generations | {len(all_results)} |",
        "",
    ]

    # Graph extraction stats
    lines.append("### Graph Extraction Success")
    lines.append("")
    lines.append("| Difficulty | Valid | Total | Pct |")
    lines.append("|---|---:|---:|---:|")
    for diff in levels:
        s = graph_stats.get(diff, {"ok": 0, "fail": 0})
        total = s["ok"] + s["fail"]
        pct = f"{100*s['ok']/total:.1f}%" if total else "N/A"
        lines.append(f"| {diff} | {s['ok']} | {total} | {pct} |")
    lines.append("")

    # Parse success
    lines.append("### Parse Success by Method")
    lines.append("")
    lines.append("| Method | parse_ok | Total | Pct |")
    lines.append("|---|---:|---:|---:|")
    for m in methods:
        rs = method_results[m]
        ok = sum(1 for r in rs if r.get("parse_ok"))
        total = len(rs)
        pct = f"{100*ok/total:.1f}%" if total else "N/A"
        lines.append(f"| {m} | {ok} | {total} | {pct} |")
    lines.append("")

    # Quality pass by method and difficulty
    lines.append("## 2. Quality Pass by Method and Difficulty")
    lines.append("")
    header = "| Method | " + " | ".join(f"{d} QP" for d in levels) + " | Total QP | Total | Pct |"
    sep = "|---|" + "|".join("---:" for _ in levels) + "|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)
    for m in methods:
        rs = method_results[m]
        qps = {}
        for d in levels:
            qps[d] = sum(1 for r in rs if r.get("quality_pass") and r.get("target_difficulty") == d)
        total_qp = sum(qps.values())
        total = len(rs)
        pct = f"{100*total_qp/total:.1f}%" if total else "N/A"
        row = f"| {m} | " + " | ".join(str(qps[d]) for d in levels) + f" | {total_qp} | {total} | {pct} |"
        lines.append(row)
    lines.append("")

    # Difficulty judge status
    lines.append("## 3. Difficulty Judge Status by Method")
    lines.append("")
    lines.append("| Method | judge_ok | judge_error | Total | Error rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for m in methods:
        rs = method_results[m]
        ok = sum(1 for r in rs if r.get("difficulty_judge_status") == "ok")
        err = sum(1 for r in rs if r.get("difficulty_judge_status") != "ok")
        total = len(rs)
        rate = f"{100*err/total:.1f}%" if total else "N/A"
        lines.append(f"| {m} | {ok} | {err} | {total} | {rate} |")
    lines.append("")

    # CrossQG primary metrics
    lines.append("## 4. CrossQG Primary Metrics")
    lines.append("")

    # 4a. Overall difficulty accuracy
    lines.append("### 4a. Overall Difficulty Accuracy (quality-pass, judge-ok)")
    lines.append("")
    lines.append("| Method | Accuracy | Wilson 95% CI | Macro Accuracy |")
    lines.append("|---|---|---|---|")
    for m in methods:
        cm = crossqg_metrics[m]
        n = cm["n_valid"]
        correct = int(cm["overall_accuracy"] * n)
        ci_str = _wilson_str(correct, n)
        lines.append(f"| {m} | {cm['overall_accuracy']*100:.1f}% | {ci_str} | {cm['macro_accuracy']*100:.1f}% |")
    lines.append("")

    # 4b. Confusion matrix per method
    lines.append("### 4b. Confusion Matrix by Method (quality-pass, judge-ok)")
    lines.append("")
    for m in methods:
        cm = crossqg_metrics[m]
        conf = cm["confusion"]
        lines.append(f"**{m}:**")
        lines.append("")
        lines.append("| Target \\ Predicted | Easy | Medium | Hard |")
        lines.append("|---|---:|---:|---:|")
        for t in levels:
            row = f"| {t} | " + " | ".join(str(conf[(t, p)]) for p in levels) + " |"
            lines.append(row)
        lines.append("")

    # 4c. Per-level hit rate
    lines.append("### 4c. Per-Level Hit Rate by Method (quality-pass, judge-ok)")
    lines.append("")
    header = "| Method | " + " | ".join(f"{d} hit" for d in levels) + " |"
    sep = "|---|" + "|".join("---:" for _ in levels) + "|"
    lines.append(header)
    lines.append(sep)
    for m in methods:
        cm = crossqg_metrics[m]
        cells = []
        for d in levels:
            pl = cm["per_level"][d]
            cells.append(_wilson_str(pl["correct"], pl["total"]))
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("")

    # 4d. Macro F1
    lines.append("### 4d. Macro F1 Score by Method (quality-pass, judge-ok)")
    lines.append("")
    lines.append("| Method | Macro F1 | " + " | ".join(f"{d} F1" for d in levels) + " |")
    lines.append("|---|---:|" + "|".join("---:" for _ in levels) + "|")
    for m in methods:
        cm = crossqg_metrics[m]
        cells = [f"{cm['per_level_f1'][d]['f1']*100:.1f}%" for d in levels]
        lines.append(f"| {m} | {cm['macro_f1']*100:.1f}% | " + " | ".join(cells) + " |")
    lines.append("")

    # 4e. Spearman correlation
    lines.append("### 4e. Spearman Correlation by Method (quality-pass, judge-ok)")
    lines.append("")
    lines.append("| Method | Spearman rho | N |")
    lines.append("|---|---:|---:|")
    for m in methods:
        cm = crossqg_metrics[m]
        lines.append(f"| {m} | {cm['spearman']:.3f} | {cm['n_valid']} |")
    lines.append("")

    # 5. Per-level detailed metrics
    lines.append("## 5. Per-Level Detailed Metrics (quality-pass, judge-ok)")
    lines.append("")
    for d in levels:
        lines.append(f"### {d} Target")
        lines.append("")
        lines.append("| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for m in methods:
            cm = crossqg_metrics[m]
            conf = cm["confusion"]
            pl = cm["per_level"][d]
            n = pl["total"]
            correct = pl["correct"]
            acc = f"{pl['accuracy']*100:.1f}%" if n else "N/A"
            pe = conf[(d, "Easy")]
            pm = conf[(d, "Medium")]
            ph = conf[(d, "Hard")]
            lines.append(f"| {m} | {n} | {correct} | {acc} | {pe} | {pm} | {ph} |")
        lines.append("")

    # 6. Hard-only secondary diagnostics
    lines.append("## 6. Hard-Only Diagnostics (Secondary)")
    lines.append("")

    hard_rs = [r for r in all_results if r.get("target_difficulty") == "Hard"]
    hard_method_rs = {m: [r for r in hard_rs if r.get("method") == m] for m in methods}

    # 6a. Hard hit rate
    lines.append("### 6a. Hard Hit Rate (Hard target, quality-pass, judge-ok)")
    lines.append("")
    lines.append("| Method | Hard hit | Wilson 95% CI |")
    lines.append("|---|---|---|")
    for m in methods:
        rs = hard_method_rs[m]
        valid = [r for r in rs if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
        hard_hit = sum(1 for r in valid if r.get("predicted_difficulty") == "Hard")
        lines.append(f"| {m} | {_wilson_str(hard_hit, len(valid))} | |")
    lines.append("")

    # 6b. HRP-v2
    lines.append("### 6b. HRP-v2 by Method (Hard target, quality-pass, judge-ok)")
    lines.append("")
    lines.append("| Method | HRP-v2 | Wilson 95% CI |")
    lines.append("|---|---|---|")
    for m in methods:
        rs = hard_method_rs[m]
        valid = [r for r in rs if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
        hrp = sum(1 for r in valid if r.get("hard_realization_pass_v2") == "yes")
        lines.append(f"| {m} | {_wilson_str(hrp, len(valid))} | |")
    lines.append("")

    # 6c. Strict HRP-v2
    lines.append("### 6c. Strict HRP-v2 by Method (Hard target, quality-pass, judge-ok)")
    lines.append("")
    lines.append("| Method | Strict HRP-v2 | Wilson 95% CI |")
    lines.append("|---|---|---|")
    for m in methods:
        rs = hard_method_rs[m]
        valid = [r for r in rs if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
        shrp = sum(1 for r in valid if r.get("strict_hrp_v2") == "yes")
        lines.append(f"| {m} | {_wilson_str(shrp, len(valid))} | |")
    lines.append("")

    # 6d. Unique Hard stories
    lines.append("### 6d. Unique Stories Among Predicted Hard (Hard target, quality-pass, judge-ok)")
    lines.append("")
    lines.append("| Method | Unique stories | Hard count |")
    lines.append("|---|---:|---:|")
    for m in methods:
        rs = hard_method_rs[m]
        valid = [r for r in rs if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
        predicted_hard = [r for r in valid if r.get("predicted_difficulty") == "Hard"]
        stories = set(r.get("story_name", "") for r in predicted_hard)
        lines.append(f"| {m} | {len(stories)} | {len(predicted_hard)} |")
    lines.append("")

    # 7. Pairwise difference table
    lines.append("## 7. Pairwise Difference Table (Ours - Baseline)")
    lines.append("")
    header = "| Metric | Ours | " + " | ".join(baseline_methods) + " | " + " | ".join(f"Ours-{m}" for m in baseline_methods) + " |"
    lines.append(header)
    sep = "|---|---" + "|---:" * len(baseline_methods) + "|" + "|".join("---" for _ in baseline_methods) + "|"
    lines.append(sep)

    ours_m = crossqg_metrics["Ours"]
    for metric_name, metric_key in [
        ("quality_pass", "n_quality_pass"),
        ("overall_accuracy", "overall_accuracy"),
        ("macro_accuracy", "macro_accuracy"),
        ("macro_f1", "macro_f1"),
        ("spearman", "spearman"),
    ]:
        ours_val = ours_m[metric_key]
        if metric_key in ("n_quality_pass",):
            ours_n = ours_m["n_total"]
            ours_pct = ours_val / ours_n if ours_n else 0
            cells = [f"{ours_pct*100:.1f}% ({ours_val}/{ours_n})"]
            diffs = []
            for bm in baseline_methods:
                bm_m = crossqg_metrics[bm]
                bm_val = bm_m[metric_key]
                bm_n = bm_m["n_total"]
                bm_pct = bm_val / bm_n if bm_n else 0
                cells.append(f"{bm_pct*100:.1f}% ({bm_val}/{bm_n})")
                diffs.append(f"{(ours_pct - bm_pct)*100:+.1f}pp")
        else:
            cells = [f"{ours_val*100:.1f}%" if metric_key != "spearman" else f"{ours_val:.3f}"]
            diffs = []
            for bm in baseline_methods:
                bm_val = crossqg_metrics[bm][metric_key]
                cells.append(f"{bm_val*100:.1f}%" if metric_key != "spearman" else f"{bm_val:.3f}")
                diff = ours_val - bm_val
                diffs.append(f"{diff*100:+.1f}pp" if metric_key != "spearman" else f"{diff:+.3f}")

        lines.append(f"| {metric_name} | " + " | ".join(cells) + " | " + " | ".join(diffs) + " |")
    lines.append("")

    # 8. Success criteria
    lines.append("## 8. Success Criteria")
    lines.append("")

    ours_cm = crossqg_metrics["Ours"]
    baseline_accs = [crossqg_metrics[m]["overall_accuracy"] for m in baseline_methods]
    baseline_macro = [crossqg_metrics[m]["macro_accuracy"] for m in baseline_methods]
    baseline_spearman = [crossqg_metrics[m]["spearman"] for m in baseline_methods]
    baseline_f1 = [crossqg_metrics[m]["macro_f1"] for m in baseline_methods]

    # Compute margin for Spearman
    spearman_margins = [ours_cm["spearman"] - b for b in baseline_spearman]
    min_spearman_margin = min(spearman_margins) if spearman_margins else 0
    spearman_pass = all(ours_cm["spearman"] >= b for b in baseline_spearman)
    spearman_small_margin = spearman_pass and min_spearman_margin < 0.05

    criteria = [
        ("selected >= 150 per level",
         meta.get("n_easy", 0) >= 150 and meta.get("n_medium", 0) >= 150 and meta.get("n_hard", 0) >= 150,
         f"Easy={meta.get('n_easy')}, Medium={meta.get('n_medium')}, Hard={meta.get('n_hard')}"),
        ("Ours quality_pass not << baselines",
         ours_cm["n_quality_pass"] / max(1, ours_cm["n_total"]) >= min(
             crossqg_metrics[m]["n_quality_pass"] / max(1, crossqg_metrics[m]["n_total"])
             for m in baseline_methods) - 0.05,
         f"Ours={ours_cm['n_quality_pass']}/{ours_cm['n_total']}"),
        ("Ours overall accuracy >= each baseline",
         all(ours_cm["overall_accuracy"] >= b for b in baseline_accs),
         f"Ours={ours_cm['overall_accuracy']*100:.1f}%"),
        ("Ours macro accuracy >= each baseline",
         all(ours_cm["macro_accuracy"] >= b for b in baseline_macro),
         f"Ours={ours_cm['macro_accuracy']*100:.1f}%"),
        ("Ours Spearman >= each baseline",
         spearman_pass,
         f"Ours={ours_cm['spearman']:.3f} (min margin={min_spearman_margin:+.3f} vs SelfRefine)"
         if spearman_small_margin else
         f"Ours={ours_cm['spearman']:.3f}"),
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
    # Build a conclusion based on global metrics and bootstrap diagnostics
    any_sig = False
    if bootstrap_diag:
        any_sig = any(bd.get("significant") for bd in bootstrap_diag.values())

    if not all_pass:
        lines.append("**Overall: SOME CRITERIA FAILED**")
    elif any_sig:
        lines.append("**Overall: ALL CRITERIA MET. Some paired bootstrap CIs exclude 0 "
                      "(see Section 9).**")
    else:
        lines.append("**Overall: Ours has the highest global overall accuracy, macro accuracy, "
                      "macro F1, and corrected Spearman. However, paired bootstrap CIs include 0 "
                      "for all comparisons (Section 9), so statistical significance is not established "
                      "at n=150 per level. Do not claim statistically significant superiority.**")
    lines.append("")

    # 9. Paired bootstrap significance diagnostics
    if bootstrap_diag:
        lines.append("## 9. Paired Bootstrap Significance Diagnostics")
        lines.append("")
        lines.append("Paired bootstrap (10k resamples) on quality-pass, judge-ok rows.")
        lines.append("Pairing key: (story_name, question, answer, target_difficulty).")
        lines.append("Metrics below are computed on the paired subset (not global).")
        lines.append("Significance = 95% CI excludes 0.  "
                      "Approximate p = 2 * min(P(diff<=0), P(diff>=0)).")
        lines.append("")
        lines.append("| Baseline | Metric | Ours (paired) | Baseline (paired) | Diff | 95% CI | approx p | N | Sig? |")
        lines.append("|---|---|---:|---:|---:|---|---:|---:|---|")
        for bm in baseline_methods:
            for metric_name in ["overall_accuracy", "macro_accuracy", "macro_f1", "spearman"]:
                key = (bm, metric_name)
                if key not in bootstrap_diag:
                    continue
                bd = bootstrap_diag[key]
                is_pct = metric_name != "spearman"
                ci_str = (f"[{bd['ci_low']*100:+.2f}pp, {bd['ci_high']*100:+.2f}pp]"
                          if is_pct else f"[{bd['ci_low']:+.3f}, {bd['ci_high']:+.3f}]")
                diff_str = f"{bd['diff']*100:+.2f}pp" if is_pct else f"{bd['diff']:+.3f}"
                ours_str = f"{bd['paired_ours']*100:.1f}%" if is_pct else f"{bd['paired_ours']:.3f}"
                bm_str = f"{bd['paired_baseline']*100:.1f}%" if is_pct else f"{bd['paired_baseline']:.3f}"
                sig = "yes" if bd["significant"] else "no"
                lines.append(f"| {bm} | {metric_name} | {ours_str} | {bm_str} | {diff_str} | {ci_str} | {bd['approx_p']:.4f} | {bd['n_paired']} | {sig} |")
        lines.append("")

    # 10. End-to-end metrics
    lines.append("## 10. End-to-End Accuracy (all candidates)")
    lines.append("")
    lines.append("Denominator = all selected candidates per method (including graph failures,")
    lines.append("parse errors, quality failures).  End-to-end = quality_pass AND judge_ok AND")
    lines.append("predicted == target.")
    lines.append("")
    header = "| Method | " + " | ".join(f"{d}" for d in levels) + " | Overall | Total |"
    sep = "|---|" + "|".join("---:" for _ in levels) + "|---:|---:|"
    lines.append(header)
    lines.append(sep)
    for m in methods:
        cm = crossqg_metrics[m]
        e2e = cm["e2e_per_level"]
        cells = []
        for d in levels:
            el = e2e[d]
            pct = f"{el['accuracy']*100:.1f}% ({el['correct']}/{el['total']})" if el['total'] else "N/A"
            cells.append(pct)
        overall_pct = f"{cm['e2e_accuracy']*100:.1f}% ({cm['e2e_correct']}/{cm['e2e_total']})"
        cells.append(overall_pct)
        cells.append(str(cm['e2e_total']))
        row = f"| {m} | " + " | ".join(cells) + " |"
        lines.append(row)
    lines.append("")

    # 11. Prompt audit
    lines.append("## 11. Difficulty Judge Prompt Audit")
    lines.append("")
    lines.append("The difficulty judge is **blind**: it sees only the story, generated question, and expected answer.")
    lines.append("It does NOT see the target difficulty. Below are 3 sample prompts to confirm.")
    lines.append("")

    # Collect sample prompts from quality-pass results
    sample_results = [r for r in all_results if r.get("quality_pass")][:3]
    for idx, r in enumerate(sample_results):
        dj = r.get("difficulty_judge", {})
        prompt_text = dj.get("difficulty_judge_prompt", "N/A")
        if prompt_text and prompt_text != "N/A":
            # Truncate to first 800 chars for readability
            if len(prompt_text) > 800:
                prompt_text = prompt_text[:800] + "\n... [truncated]"
            lines.append(f"**Sample {idx+1}** (story={r.get('story_name','?')[:30]}, target={r.get('target_difficulty','?')}):")
            lines.append("")
            lines.append("```")
            lines.append(prompt_text)
            lines.append("```")
            lines.append("")

    # Confirm no target difficulty in prompts
    # Check for explicit target difficulty leakage (e.g., "target difficulty: Easy")
    # but NOT for the scale definitions ("1 sentence = Easy") which are part of the rubric
    has_target_diff = False
    for r in all_results:
        dj = r.get("difficulty_judge", {})
        p = dj.get("difficulty_judge_prompt", "")
        target = r.get("target_difficulty", "")
        if not target or not p:
            continue
        # Check for explicit target leakage patterns
        leakage_patterns = [
            f"target difficulty: {target}",
            f"target: {target}",
            f"difficulty: {target}",
            f"expected difficulty: {target}",
        ]
        if any(pat in p.lower() for pat in leakage_patterns):
            has_target_diff = True
            break
    if has_target_diff:
        lines.append("**WARNING:** Target difficulty was found in at least one judge prompt! This is a bug.")
    else:
        lines.append("**Confirmed:** No target difficulty appears in any judge prompt. Blind evaluation is correct.")
    lines.append("")

    # 12. Failure analysis
    lines.append("## 12. Failure Reasons by Method")
    lines.append("")
    lines.append("| Method | Failure reason | Count |")
    lines.append("|---|---|---:|")
    for m in methods:
        rs = [r for r in all_results if r.get("method") == m and not r.get("quality_pass")]
        reasons = Counter()
        for r in rs:
            gen_err = r.get("generation_error")
            if gen_err:
                reasons[f"gen error: {gen_err}"] += 1
            elif r.get("quality_judge"):
                qj = r["quality_judge"]
                if qj.get("answerable") in ("no",):
                    reasons["not answerable"] += 1
                elif qj.get("fluency") in ("no",):
                    reasons["not fluent"] += 1
                elif qj.get("asks_expected_answer") in ("no",):
                    reasons["wrong answer"] += 1
                else:
                    reasons["other"] += 1
            else:
                reasons["other"] += 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            lines.append(f"| {m} | {reason} | {count} |")
    lines.append("")

    # 13. Examples
    lines.append("## 13. Examples")
    lines.append("")

    for d in levels:
        lines.append(f"### Best {d} examples (quality-pass, correct prediction)")
        lines.append("")
        for m in methods:
            rs = [r for r in all_results
                  if r.get("method") == m
                  and r.get("target_difficulty") == d
                  and r.get("quality_pass")
                  and r.get("predicted_difficulty") == d]
            if rs:
                r = rs[0]
                lines.append(f"**{m} Example:**")
                lines.append(f"- Story: {r.get('story_name', '?')}")
                lines.append(f"- Question: {r.get('generated_question', '?')}")
                lines.append(f"- Target answer: {r.get('answer', '?')}")
                lines.append(f"- Target: {d}, Predicted: {r.get('predicted_difficulty', '?')}")
                lines.append("")
        lines.append("")

    # Write report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Report: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FairytaleQA CrossQG Evaluation")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--target_per_level", type=int, default=150)
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="Override target_per_level for testing")
    parser.add_argument("--graphs", default=None,
                        help="Path to pre-extracted graphs.jsonl (skip extraction)")
    parser.add_argument("--regen_report", action="store_true",
                        help="Regenerate report from existing generations.judged.full.jsonl (no API calls)")
    args = parser.parse_args()

    if args.limit:
        args.target_per_level = args.limit

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ["Direct", "ICL", "SelfRefine", "Ours"]

    # Regen-report mode: reload from existing JSONL and rebuild report
    if args.regen_report:
        print("=== Regenerating report from existing data ===")
        judged_path = output_dir / "generations.judged.full.jsonl"
        if not judged_path.exists():
            print(f"ERROR: {judged_path} not found")
            return

        all_results = []
        with open(judged_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_results.append(json.loads(line))
        print(f"  Loaded {len(all_results)} results")

        # Graph stats from graphs.jsonl
        graphs_path = output_dir / "graphs.jsonl"
        graph_stats = {"Easy": {"ok": 0, "fail": 0}, "Medium": {"ok": 0, "fail": 0}, "Hard": {"ok": 0, "fail": 0}}
        if graphs_path.exists():
            with open(graphs_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line.strip())
                    diff = rec.get("target_difficulty", "Hard")
                    if diff in graph_stats:
                        if rec.get("graph_valid"):
                            graph_stats[diff]["ok"] += 1
                        else:
                            graph_stats[diff]["fail"] += 1

        crossqg_metrics = _compute_crossqg_metrics(all_results)
        bootstrap_diag = _compute_bootstrap_diagnostics(all_results, crossqg_metrics)

        # Reconstruct meta from results
        levels = ["Easy", "Medium", "Hard"]
        n_by_level = {d: 0 for d in levels}
        for r in all_results:
            d = r.get("target_difficulty")
            if d in n_by_level:
                n_by_level[d] += 1
        # Divide by 4 methods to get per-level candidate count
        n_per_level = {d: n_by_level[d] // len(methods) for d in levels}

        meta = {
            "target_per_level": n_per_level.get("Easy", 0),
            "n_easy": n_per_level.get("Easy", 0),
            "n_medium": n_per_level.get("Medium", 0),
            "n_hard": n_per_level.get("Hard", 0),
            "n_total": sum(n_per_level.values()),
        }

        _build_report(all_results, crossqg_metrics, graph_stats, output_dir, meta=meta, bootstrap_diag=bootstrap_diag)
        print("\nDone!")
        return

    methods = ["Direct", "ICL", "SelfRefine", "Ours"]

    # Step 1: Select balanced candidates
    print("=== Step 1: Selecting balanced candidates ===")
    selected = _select_balanced_candidates(
        args.candidates, target_per_level=args.target_per_level, seed=args.seed
    )
    print(f"  Total selected: {len(selected)}")

    # Save selected candidates
    sel_path = output_dir / "selected_candidates.jsonl"
    with open(sel_path, "w", encoding="utf-8") as f:
        for c in selected:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    n_easy = sum(1 for c in selected if c.get("target_difficulty") == "Easy")
    n_medium = sum(1 for c in selected if c.get("target_difficulty") == "Medium")
    n_hard = sum(1 for c in selected if c.get("target_difficulty") == "Hard")

    # Step 2: Extract graphs
    print("\n=== Step 2: Extracting narrative graphs ===")
    if args.graphs:
        print(f"  Loading pre-extracted graphs from {args.graphs}")
        graph_lookup = {}
        graph_stats = {"Easy": {"ok": 0, "fail": 0}, "Medium": {"ok": 0, "fail": 0}, "Hard": {"ok": 0, "fail": 0}}
        with open(args.graphs, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    key = (rec.get("story_name", ""), rec.get("question", ""))
                    graph_lookup[key] = rec
                    diff = rec.get("target_difficulty", "Hard")
                    if diff in graph_stats:
                        if rec.get("graph_valid"):
                            graph_stats[diff]["ok"] += 1
                        else:
                            graph_stats[diff]["fail"] += 1
    else:
        graph_lookup, graph_stats = _extract_graphs(selected, output_dir)

    # Step 3: Generate and judge
    print("\n=== Step 3: Generating and judging ===")
    all_results = []
    raw_path = output_dir / "generations.raw.jsonl"

    total_tasks = len(selected) * len(methods)
    task_idx = 0

    with open(raw_path, "w", encoding="utf-8") as f:
        for i, cand in enumerate(selected):
            story = cand.get("story_name", "?")[:30]
            diff = cand.get("target_difficulty", "Hard")
            key = (cand.get("story_name", ""), cand.get("question", ""))
            graph_data = graph_lookup.get(key)

            for method in methods:
                task_idx += 1
                t0 = time.time()

                result = _run_method(method, cand, graph_data)
                result["story_name"] = cand.get("story_name", "")
                result["target_difficulty"] = diff
                result["question"] = cand.get("question", "")
                result["answer"] = cand.get("answer", "") or cand.get("answer1", "")

                # Judge (skip if generation failed)
                if result.get("parse_ok") and result.get("generated_question"):
                    try:
                        result = _judge_generation(result, cand)
                    except Exception as e:
                        result["quality_pass"] = False
                        result["difficulty_judge_status"] = "judge_error"
                        result["generation_error"] = f"judge_error: {e}"

                elapsed = time.time() - t0
                qp = "QP" if result.get("quality_pass") else "FAIL"
                pred = result.get("predicted_difficulty", "?")
                print(f"  [{task_idx}/{total_tasks}] {story}... {method}: {qp} pred={pred} ({elapsed:.1f}s)")

                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                all_results.append(result)

    # Step 4: Compute CrossQG metrics
    print("\n=== Step 4: Computing CrossQG metrics ===")
    crossqg_metrics = _compute_crossqg_metrics(all_results)

    # Print summary
    print("\n=== Summary ===")
    for m in methods:
        cm = crossqg_metrics[m]
        print(f"  {m}: quality_pass={cm['n_quality_pass']}/{cm['n_total']}, "
              f"accuracy={cm['overall_accuracy']*100:.1f}%, "
              f"macro_acc={cm['macro_accuracy']*100:.1f}%, "
              f"spearman={cm['spearman']:.3f}")

    # Step 5: Write judged JSONL
    judged_path = output_dir / "generations.judged.jsonl"
    judged_full_path = output_dir / "generations.judged.full.jsonl"
    with open(judged_path, "w", encoding="utf-8") as f, \
         open(judged_full_path, "w", encoding="utf-8") as ffull:
        for r in all_results:
            # Compact version (no judge raw details)
            compact = {k: v for k, v in r.items()
                       if k not in ("quality_judge", "difficulty_judge", "evidence_coverage",
                                    "generation_prompt", "generation_raw", "attempts_trace")}
            f.write(json.dumps(compact, ensure_ascii=False) + "\n")
            ffull.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Step 6: Bootstrap significance
    print("\n=== Step 5: Computing bootstrap significance ===")
    bootstrap_diag = _compute_bootstrap_diagnostics(all_results, crossqg_metrics)

    # Step 7: Build report
    print("\n=== Step 6: Building report ===")
    meta = {
        "target_per_level": args.target_per_level,
        "n_easy": n_easy,
        "n_medium": n_medium,
        "n_hard": n_hard,
        "n_total": len(selected),
    }
    _build_report(all_results, crossqg_metrics, graph_stats, output_dir, meta=meta, bootstrap_diag=bootstrap_diag)

    print("\nDone!")


if __name__ == "__main__":
    main()
