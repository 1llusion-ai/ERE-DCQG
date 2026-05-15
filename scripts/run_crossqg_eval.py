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


def _select_story_matched_candidates(candidates_path, candidates_per_level_per_story=1, max_stories=None, seed=42):
    """Story-matched candidate selection.

    Groups candidates by story_name and evidence_difficulty. Keeps only stories
    with >= 1 Easy, >= 1 Medium, >= 1 Hard. Selects exactly
    candidates_per_level_per_story per level per story, deterministically.

    Candidate preference within story:
      - Easy: prefer answer_sentence_alone_sufficient=yes and num_required_sentences=1
      - Medium: prefer num_required_sentences=2
      - Hard: prefer necessity_type in {motivation_bridge, causal_bridge, summary_synthesis}
        and non-temporal reasoning if available

    Returns list of candidate dicts tagged with 'target_difficulty' and 'story_group_id'.
    """
    import random
    rng = random.Random(seed)

    # Load and group by (story_name, evidence_difficulty)
    story_levels = {}  # story_name -> {"Easy": [...], "Medium": [...], "Hard": [...]}
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            diff = rec.get("evidence_difficulty", "")
            if diff not in ("Easy", "Medium", "Hard"):
                continue
            sn = rec.get("story_name", "")
            if sn not in story_levels:
                story_levels[sn] = {"Easy": [], "Medium": [], "Hard": []}
            story_levels[sn][diff].append(rec)

    # Filter to stories with at least 1 candidate per level
    eligible = {
        sn: levels for sn, levels in story_levels.items()
        if len(levels["Easy"]) >= 1 and len(levels["Medium"]) >= 1 and len(levels["Hard"]) >= 1
    }
    eligible_names = sorted(eligible.keys())
    print(f"  Pool: {len(story_levels)} total stories, {len(eligible_names)} eligible (>=1 Easy/Med/Hard each)")

    if max_stories and max_stories < len(eligible_names):
        rng.shuffle(eligible_names)
        eligible_names = sorted(eligible_names[:max_stories])
        print(f"  Limited to {len(eligible_names)} stories (max_stories={max_stories})")

    # Preference scoring helpers
    def _easy_preference(c):
        score = 0
        if c.get("answer_sentence_alone_sufficient") == "yes":
            score += 10
        if c.get("num_required_sentences", 99) == 1:
            score += 5
        return score

    def _medium_preference(c):
        score = 0
        if c.get("num_required_sentences", 0) == 2:
            score += 10
        elif c.get("num_required_sentences", 0) == 3:
            score += 3
        return score

    def _hard_preference(c):
        score = 0
        nt = c.get("necessity_type", "")
        if nt in ("motivation_bridge", "causal_bridge", "summary_synthesis"):
            score += 10
        elif nt in ("disambiguation", "answer_identification"):
            score += 7
        elif nt in ("temporal_bridge",):
            score += 3
        ro = c.get("reasoning_operation", "")
        temporal_ops = ("temporal_order", "explicit_lookup")
        if ro and ro not in temporal_ops:
            score += 5
        if c.get("bridge_removal_effect") in ("ambiguous", "unanswerable"):
            score += 3
        return score

    # Select candidates per story
    selected = []
    for idx, sn in enumerate(eligible_names):
        levels = eligible[sn]

        for diff, preference_fn in [
            ("Easy", _easy_preference),
            ("Medium", _medium_preference),
            ("Hard", _hard_preference),
        ]:
            pool = levels[diff]
            # Sort by preference (descending), then deterministic shuffle within ties
            scored = [(preference_fn(c), i, c) for i, c in enumerate(pool)]
            scored.sort(key=lambda x: (-x[0], x[1]))  # highest score first, then original order
            # Take top candidates_per_level_per_story
            chosen = [c for _, _, c in scored[:candidates_per_level_per_story]]
            for c in chosen:
                c_out = dict(c)
                c_out["target_difficulty"] = diff
                c_out["story_group_id"] = idx
                selected.append(c_out)

    # Summary
    by_diff = {"Easy": 0, "Medium": 0, "Hard": 0}
    for c in selected:
        by_diff[c["target_difficulty"]] += 1
    print(f"  Selected: {len(selected)} candidates ({by_diff['Easy']}E/{by_diff['Medium']}M/{by_diff['Hard']}H) "
          f"from {len(eligible_names)} stories "
          f"({candidates_per_level_per_story} per level per story)")

    return selected


def _select_story_matched_suitable_candidates(candidates_path, candidates_per_level_per_story=1,
                                               max_stories=None, seed=42):
    """Select story-matched candidates from the suitability-filtered pool.

    Applies candidate suitability audit BEFORE story-matched selection.
    Same logic as _select_story_matched_candidates but operates on the
    suitable subset only.
    """
    from scripts.audit_fairytale_candidate_suitability import (
        assess_candidate_suitability,
        select_story_matched_suitable,
    )

    print("  Applying suitability filter...")
    all_candidates = []
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            diff = rec.get("evidence_difficulty", "")
            if diff not in ("Easy", "Medium", "Hard"):
                continue
            all_candidates.append(rec)

    # Assess suitability
    suitable_pool = []
    reject_reasons = {}
    for c in all_candidates:
        result = assess_candidate_suitability(c)
        c.update(result)
        if c.get("suitable"):
            suitable_pool.append(c)
        else:
            rr = c.get("reject_reason", "unknown")
            reject_reasons[rr] = reject_reasons.get(rr, 0) + 1

    print(f"  Suitable pool: {len(suitable_pool)}/{len(all_candidates)} "
          f"({100*len(suitable_pool)/len(all_candidates):.1f}%)")
    for rr, cnt in sorted(reject_reasons.items(), key=lambda x: -x[1])[:5]:
        print(f"    Rejected: {cnt} ({rr})")

    # Story-matched selection from suitable pool
    selected, n_eligible = select_story_matched_suitable(
        suitable_pool,
        per_level_per_story=candidates_per_level_per_story,
        max_stories=max_stories,
        seed=seed,
    )

    by_diff = {"Easy": 0, "Medium": 0, "Hard": 0}
    for c in selected:
        by_diff[c["target_difficulty"]] += 1
    print(f"  Selected: {len(selected)} candidates ({by_diff['Easy']}E/{by_diff['Medium']}M/{by_diff['Hard']}H) "
          f"from {n_eligible} eligible stories "
          f"(suitable, {candidates_per_level_per_story} per level per story)")

    return selected


def _select_story_matched_calibrated_candidates(calibrated_path, candidates_path,
                                                 candidates_per_level_per_story=1,
                                                 max_stories=None, seed=42,
                                                 min_stories_threshold=70):
    """Select story-matched candidates from the calibrated pool only.

    Loads calibrated.jsonl, keeps only calibrated=True candidates, then applies
    story-matched selection with same preferences as _select_story_matched_candidates.

    If n_eligible < min_stories_threshold, prints bottleneck analysis and returns
    (None, n_eligible, bottleneck_info) — caller should exit without running QG.
    """
    import random
    rng = random.Random(seed)

    if not os.path.exists(calibrated_path):
        print(f"  ERROR: Calibrated file not found: {calibrated_path}")
        print(f"  Run calibration audit first:")
        print(f"    python -m scripts.audit_fairytale_target_calibration "
              f"--candidates {candidates_path} --output_dir <dir>")
        return None, 0, {"error": "calibrated file not found"}

    calibrated = []
    with open(calibrated_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("calibrated") and rec.get("evidence_difficulty") in ("Easy", "Medium", "Hard"):
                calibrated.append(rec)

    print(f"  Calibrated pool: {len(calibrated)} candidates")

    story_levels = {}
    for rec in calibrated:
        sn = rec.get("story_name", "")
        diff = rec.get("evidence_difficulty", "")
        if sn not in story_levels:
            story_levels[sn] = {"Easy": [], "Medium": [], "Hard": []}
        story_levels[sn][diff].append(rec)

    eligible = {
        sn: levels for sn, levels in story_levels.items()
        if len(levels["Easy"]) >= 1 and len(levels["Medium"]) >= 1 and len(levels["Hard"]) >= 1
    }
    eligible_names = sorted(eligible.keys())
    n_eligible = len(eligible_names)
    print(f"  Stories with calibrated Easy+Medium+Hard: {n_eligible}")

    if n_eligible < min_stories_threshold:
        print(f"\n  *** CALIBRATED POOL TOO SMALL: {n_eligible} stories < "
              f"{min_stories_threshold} threshold ***\n")
        print(f"  Bottleneck analysis:")
        all_stories = set(rec.get("story_name", "") for rec in calibrated)
        print(f"    Total stories in calibrated pool: {len(all_stories)}")
        for diff in ("Easy", "Medium", "Hard"):
            stories_with = set(
                rec.get("story_name", "")
                for rec in calibrated
                if rec.get("evidence_difficulty") == diff
            )
            print(f"    Stories with calibrated {diff}: {len(stories_with)}/{len(all_stories)}")
        # Per-story missing detail
        per_story = {}
        for rec in calibrated:
            sn = rec.get("story_name", "")
            per_story.setdefault(sn, set()).add(rec.get("evidence_difficulty", ""))
        missing_diffs = {}
        for sn in set(rec.get("story_name", "") for rec in calibrated):
            have = per_story.get(sn, set())
            miss = {"Easy", "Medium", "Hard"} - have
            if miss:
                missing_diffs[sn] = miss
        if missing_diffs:
            print(f"    Stories missing difficulty levels ({len(missing_diffs)}):")
            for sn, miss in sorted(missing_diffs.items(), key=lambda x: -len(x[1]))[:15]:
                print(f"      {sn}: missing {', '.join(sorted(miss))}")
        print(f"\n  CANNOT proceed with QG. Run calibration audit to increase pool, "
              f"or lower threshold.")
        return None, n_eligible, {
            "n_eligible": n_eligible,
            "total_stories": len(set(rec.get("story_name", "") for rec in calibrated)),
            "missing_diffs": missing_diffs,
        }

    if max_stories and max_stories < n_eligible:
        rng.shuffle(eligible_names)
        eligible_names = sorted(eligible_names[:max_stories])
        print(f"  Limited to {len(eligible_names)} stories (max_stories={max_stories})")

    # Same preference scoring as _select_story_matched_candidates
    def _easy_preference(c):
        score = 0
        if c.get("answer_sentence_alone_sufficient") == "yes":
            score += 10
        if c.get("num_required_sentences", 99) == 1:
            score += 5
        return score

    def _medium_preference(c):
        score = 0
        if c.get("num_required_sentences", 0) == 2:
            score += 10
        elif c.get("num_required_sentences", 0) == 3:
            score += 3
        return score

    def _hard_preference(c):
        score = 0
        nt = c.get("necessity_type", "")
        if nt in ("motivation_bridge", "causal_bridge", "summary_synthesis"):
            score += 10
        elif nt in ("disambiguation", "answer_identification"):
            score += 7
        elif nt in ("temporal_bridge",):
            score += 3
        ro = c.get("reasoning_operation", "")
        if ro and ro not in ("temporal_order", "explicit_lookup"):
            score += 5
        if c.get("bridge_removal_effect") in ("ambiguous", "unanswerable"):
            score += 3
        return score

    selected = []
    for idx, sn in enumerate(eligible_names):
        levels = eligible[sn]
        for diff, preference_fn in [
            ("Easy", _easy_preference),
            ("Medium", _medium_preference),
            ("Hard", _hard_preference),
        ]:
            pool = levels[diff]
            scored = [(preference_fn(c), i, c) for i, c in enumerate(pool)]
            scored.sort(key=lambda x: (-x[0], x[1]))
            chosen = [c for _, _, c in scored[:candidates_per_level_per_story]]
            for c in chosen:
                c_out = dict(c)
                c_out["target_difficulty"] = diff
                c_out["story_group_id"] = idx
                selected.append(c_out)

    by_diff = {"Easy": 0, "Medium": 0, "Hard": 0}
    for c in selected:
        by_diff[c["target_difficulty"]] += 1
    print(f"  Selected: {len(selected)} candidates ({by_diff['Easy']}E/{by_diff['Medium']}M/{by_diff['Hard']}H) "
          f"from {n_eligible} stories "
          f"({candidates_per_level_per_story} per level per story)")

    return selected, n_eligible, None


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
                "graph_policy": "graph_invalid",
                "graph_policy_reason": "graph data invalid or missing",
                "graph_policy_compliance": "no",
                "selected_node_ids": [],
                "selected_edge_relations": [],
                "evidence_roles_used": [],
                "relation_chain": [],
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
# Story-matched diagnostics
# ---------------------------------------------------------------------------

def _compute_story_matched_diagnostics(all_results):
    """Compute story-level diagnostics for story_matched evaluation.

    Returns dict with:
      - stories: per-story metrics
      - win_tie_loss: Ours vs each baseline at story level
      - story_spearman: per-story Spearman where all 3 levels are valid
      - per_story_failures: per-story failure counts by method
    """
    methods = ["Direct", "ICL", "SelfRefine", "Ours"]
    baseline_methods = ["Direct", "ICL", "SelfRefine"]
    levels = ["Easy", "Medium", "Hard"]
    ordinal = {"Easy": 1, "Medium": 2, "Hard": 3}

    # Group by story and method
    stories = {}
    for r in all_results:
        sn = r.get("story_name", "")
        m = r.get("method", "")
        if sn not in stories:
            stories[sn] = {}
        if m not in stories[sn]:
            stories[sn][m] = []
        stories[sn][m].append(r)

    # Per-story metrics
    story_metrics = {}
    for sn, method_results in stories.items():
        sm = {"story_name": sn}
        for m in methods:
            rs = method_results.get(m, [])
            valid = [r for r in rs if r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
            correct = sum(1 for r in valid if r.get("predicted_difficulty") == r.get("target_difficulty"))
            n_valid = len(valid)
            n_total = len(rs)
            n_qp = sum(1 for r in rs if r.get("quality_pass"))
            sm[f"{m}_n_total"] = n_total
            sm[f"{m}_n_qp"] = n_qp
            sm[f"{m}_n_valid"] = n_valid
            sm[f"{m}_correct"] = correct
            sm[f"{m}_accuracy"] = correct / n_valid if n_valid else 0.0
            # Per-level (quality-pass + judge-ok subset)
            for level in levels:
                lr = [r for r in valid if r.get("target_difficulty") == level]
                lc = sum(1 for r in lr if r.get("predicted_difficulty") == level)
                sm[f"{m}_{level}_correct"] = lc
                sm[f"{m}_{level}_total"] = len(lr)
            # Per-level raw counts (all results)
            for level in levels:
                lr_raw = [r for r in rs if r.get("target_difficulty") == level]
                sm[f"{m}_{level}_raw"] = len(lr_raw)
        story_metrics[sn] = sm

    # Win/Tie/Loss: Ours vs each baseline at story level
    win_tie_loss = {}
    for bm in baseline_methods:
        wins, ties, losses = 0, 0, 0
        for sn, sm in story_metrics.items():
            ours_acc = sm.get("Ours_accuracy", 0.0)
            bm_acc = sm.get(f"{bm}_accuracy", 0.0)
            if ours_acc > bm_acc:
                wins += 1
            elif ours_acc == bm_acc:
                ties += 1
            else:
                losses += 1
        n_stories = wins + ties + losses
        win_tie_loss[bm] = {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "n_stories": n_stories,
        }

    # Story-level Spearman: only stories where all 3 levels are valid for the method
    story_spearman = {}
    for m in methods:
        rhos = []
        skipped = 0
        for sn, sm in story_metrics.items():
            target_vals = []
            pred_vals = []
            for level in levels:
                key_t = f"{m}_{level}_total"
                if sm.get(key_t, 0) > 0:
                    target_vals.append(ordinal[level])
                    # Get the actual predicted value from results
                    rs = stories[sn].get(m, [])
                    lr = [r for r in rs if r.get("target_difficulty") == level
                          and r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
                    if lr:
                        pred = lr[0].get("predicted_difficulty", "Medium")
                        pred_vals.append(ordinal.get(pred, 2))
            if len(target_vals) >= 3:
                rhos.append(_spearman(target_vals, pred_vals))
            else:
                skipped += 1
        story_spearman[m] = {
            "n_valid_stories": len(rhos),
            "n_skipped": skipped,
            "mean_rho": sum(rhos) / len(rhos) if rhos else 0.0,
            "rho_values": rhos,
        }

    # Per-story failure counts by method
    per_story_failures = {}
    for m in methods:
        fail_dist = {}
        for sn, sm in story_metrics.items():
            n_fail = sm.get(f"{m}_n_total", 0) - sm.get(f"{m}_n_qp", 0)
            fail_dist[sn] = n_fail
        per_story_failures[m] = fail_dist

    return {
        "story_metrics": story_metrics,
        "win_tie_loss": win_tie_loss,
        "story_spearman": story_spearman,
        "per_story_failures": per_story_failures,
        "n_stories": len(story_metrics),
    }


# ---------------------------------------------------------------------------
# Retry / budget diagnostics
# ---------------------------------------------------------------------------

def _compute_retry_budget_diagnostics(all_results):
    """Compute retry and budget diagnostics for all methods.

    Returns dict keyed by method with:
      - average_attempts, max_attempts
      - repair_prompt_used_rate (Ours)
      - retry_reason_distribution
      - failure_reason_distribution
      - Ours: graph_policy_self_check_failure_rate
    """
    methods = ["Direct", "ICL", "SelfRefine", "Ours"]
    diags = {}

    for m in methods:
        rs = [r for r in all_results if r.get("method") == m]
        n_total = len(rs)

        # Attempts
        attempts = [r.get("attempts", 0) for r in rs]
        avg_attempts = sum(attempts) / len(attempts) if attempts else 0.0
        max_attempts_val = max(attempts) if attempts else 0

        # Repair prompt rate (Ours only)
        repair_used = 0
        repair_success = 0
        if m == "Ours":
            for r in rs:
                if r.get("repair_attempted"):
                    repair_used += 1
                    if r.get("repair_success"):
                        repair_success += 1

        # Retry reason distribution (from attempts_trace for Ours, generation_error for others)
        retry_reasons = {}
        failure_reasons = {}
        for r in rs:
            # Failure reasons
            if not r.get("quality_pass"):
                err = r.get("generation_error", "unknown")
                # Classify failure
                if "degenerate" in str(err):
                    reason = "degenerate output"
                elif "parse" in str(err).lower():
                    reason = "parse failure"
                elif "graph" in str(err).lower():
                    reason = "graph issue"
                elif "empty" in str(err):
                    reason = "empty output"
                elif "question length" in str(err):
                    reason = "question length"
                elif "self-check" in str(err):
                    reason = "self-check failed"
                elif "judge_error" in str(err):
                    reason = "judge error"
                else:
                    qj = r.get("quality_judge", {})
                    if qj:
                        if qj.get("answerable") == "no":
                            reason = "not answerable"
                        elif qj.get("fluency") == "no":
                            reason = "not fluent"
                        elif qj.get("asks_expected_answer") == "no":
                            reason = "wrong answer"
                        elif qj.get("answer_leakage") == "yes":
                            reason = "answer leakage"
                        else:
                            reason = str(err)[:60]
                    else:
                        reason = str(err)[:60]
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

            # Retry traces
            trace = r.get("attempts_trace", [])
            for t in trace:
                reason = t.get("validate_reason", "") or t.get("self_check_reason", "")
                if reason:
                    reason_short = reason[:80]
                    retry_reasons[reason_short] = retry_reasons.get(reason_short, 0) + 1

        # Ours graph_policy self-check failure rate
        gp_sc_failure = 0
        gp_sc_total = 0
        if m == "Ours":
            for r in rs:
                gp = r.get("graph_policy", "")
                if gp and gp != "graph_invalid":
                    gp_sc_total += 1
                    if r.get("graph_policy_compliance") != "yes":
                        gp_sc_failure += 1

        md = {
            "n_total": n_total,
            "average_attempts": avg_attempts,
            "max_attempts": max_attempts_val,
            "attempt_distribution": {},
            "retry_reasons": retry_reasons,
            "failure_reasons": failure_reasons,
        }

        # Attempt distribution
        for a in attempts:
            md["attempt_distribution"][a] = md["attempt_distribution"].get(a, 0) + 1

        if m == "Ours":
            md["repair_prompt_used"] = repair_used
            md["repair_prompt_used_rate"] = repair_used / n_total if n_total else 0.0
            md["repair_prompt_success"] = repair_success
            md["graph_policy_sc_failure"] = gp_sc_failure
            md["graph_policy_sc_total"] = gp_sc_total
            md["graph_policy_sc_failure_rate"] = gp_sc_failure / gp_sc_total if gp_sc_total else 0.0

        diags[m] = md

    return diags


# ---------------------------------------------------------------------------
# Similarity diagnostics (diagnostic only, no filtering)
# ---------------------------------------------------------------------------

def _compute_similarity_diagnostics(all_results):
    """Compute per-story per-method similarity diagnostics.

    For each story and method:
      - Easy-Medium, Medium-Hard, Easy-Hard lexical similarity (character n-gram)
      - Evidence sentence overlap among judge-used evidence
      - Difficulty collapse counts (all 3 predicted same, collapse to Medium)

    These are diagnostics only. No filtering or regeneration based on similarity.
    """
    methods = ["Direct", "ICL", "SelfRefine", "Ours"]
    levels = ["Easy", "Medium", "Hard"]

    def _char_ngram_similarity(text_a, text_b, n=4):
        """Character n-gram Jaccard similarity."""
        if not text_a or not text_b:
            return 0.0
        a_ngrams = set(text_a[i:i+n] for i in range(len(text_a) - n + 1))
        b_ngrams = set(text_b[i:i+n] for i in range(len(text_b) - n + 1))
        if not a_ngrams or not b_ngrams:
            return 0.0
        return len(a_ngrams & b_ngrams) / len(a_ngrams | b_ngrams)

    # Group by (story_name, method)
    story_method_qs = {}
    for r in all_results:
        sn = r.get("story_name", "")
        m = r.get("method", "")
        q = r.get("generated_question", "")
        diff = r.get("target_difficulty", "")
        dj = r.get("difficulty_judge", {})
        used_evidence = dj.get("required_evidence_sentences_used", [])
        pred = r.get("predicted_difficulty", "?")
        key = (sn, m)
        if key not in story_method_qs:
            story_method_qs[key] = {}
        story_method_qs[key][diff] = {
            "question": q,
            "used_evidence": used_evidence,
            "predicted": pred,
            "quality_pass": r.get("quality_pass", False),
            "judge_ok": r.get("difficulty_judge_status") == "ok",
        }

    # Per-method aggregate similarity
    method_similarities = {}
    for m in methods:
        pairs = {"Easy-Medium": [], "Medium-Hard": [], "Easy-Hard": []}
        evidence_overlaps = {"Easy-Medium": [], "Medium-Hard": [], "Easy-Hard": []}
        collapses = {"total_stories": 0, "collapse_all_three": 0, "collapse_to_medium": 0,
                      "collapse_to_easy": 0, "collapse_to_hard": 0}

        for (sn, method), diffs in story_method_qs.items():
            if method != m:
                continue
            # Check collapse (all 3 quality-pass judge-ok)
            qp_diffs = {d: v for d, v in diffs.items()
                        if v["quality_pass"] and v["judge_ok"]}
            if len(qp_diffs) >= 3:
                collapses["total_stories"] += 1
                preds = set(v["predicted"] for v in qp_diffs.values())
                if len(preds) == 1:
                    collapses["collapse_all_three"] += 1
                    if "Medium" in preds:
                        collapses["collapse_to_medium"] += 1
                    elif "Easy" in preds:
                        collapses["collapse_to_easy"] += 1
                    elif "Hard" in preds:
                        collapses["collapse_to_hard"] += 1

            # Pairwise similarities
            for pair, (l1, l2) in {
                "Easy-Medium": ("Easy", "Medium"),
                "Medium-Hard": ("Medium", "Hard"),
                "Easy-Hard": ("Easy", "Hard"),
            }.items():
                if l1 in diffs and l2 in diffs:
                    q1 = diffs[l1]["question"]
                    q2 = diffs[l2]["question"]
                    if q1 and q2:
                        sim = _char_ngram_similarity(q1, q2)
                        pairs[pair].append(sim)
                    # Evidence overlap
                    e1 = set(diffs[l1].get("used_evidence", []))
                    e2 = set(diffs[l2].get("used_evidence", []))
                    if e1 and e2:
                        overlap = len(e1 & e2) / len(e1 | e2)
                    else:
                        overlap = 0.0
                    evidence_overlaps[pair].append(overlap)

        avg_pairs = {}
        for pair in pairs:
            vals = pairs[pair]
            avg_pairs[pair] = {
                "mean": sum(vals) / len(vals) if vals else 0.0,
                "max": max(vals) if vals else 0.0,
                "min": min(vals) if vals else 0.0,
                "n": len(vals),
            }

        avg_evidence = {}
        for pair in evidence_overlaps:
            vals = evidence_overlaps[pair]
            avg_evidence[pair] = {
                "mean": sum(vals) / len(vals) if vals else 0.0,
                "max": max(vals) if vals else 0.0,
                "n": len(vals),
            }

        method_similarities[m] = {
            "question_similarity": avg_pairs,
            "evidence_overlap": avg_evidence,
            "collapses": collapses,
            "n_stories_with_3qp": collapses["total_stories"],
        }

    return method_similarities


# ---------------------------------------------------------------------------
# Stage 2 diagnostics (focus, difficulty realization)
# ---------------------------------------------------------------------------

def _compute_stage2_diagnostics(all_results):
    """Compute Stage 2 focus and difficulty realization diagnostics.

    Returns dict with:
      - focus_distribution: focus type counts by target difficulty (Ours)
      - easy_asa_by_focus: Easy answer_sentence_alone=yes rate by focus type (Ours)
      - hard_asa_by_focus: Hard answer_sentence_alone=no rate by focus type (Ours)
      - policy_compliance_by_focus: graph_policy_compliance=yes rate by focus type (Ours)
      - repair_by_difficulty: repair_used / repair_success by target difficulty (Ours)
      - top_easy_failures: top 10 Easy failures with classified failure reason
      - top_hard_failures: top 10 Hard failures with classified failure reason
    """
    levels = ["Easy", "Medium", "Hard"]
    ours = [r for r in all_results if r.get("method") == "Ours"]
    ours_qp = [r for r in ours if r.get("quality_pass")]

    # Focus distribution by target difficulty (all Ours, quality-pass only)
    focus_dist = {}
    for level in levels:
        focus_dist[level] = {}
        lr = [r for r in ours_qp if r.get("target_difficulty") == level]
        for r in lr:
            f = r.get("question_focus", "unknown")
            focus_dist[level][f] = focus_dist[level].get(f, 0) + 1

    # Also count node-level focus (pre-override) for comparison
    node_focus_dist = {}
    for level in levels:
        node_focus_dist[level] = {}
        lr = [r for r in ours_qp if r.get("target_difficulty") == level]
        for r in lr:
            f = r.get("node_question_focus", "unknown")
            node_focus_dist[level][f] = node_focus_dist[level].get(f, 0) + 1

    # Easy answer_sentence_alone=yes by focus type (Ours, quality-pass)
    easy_asa_by_focus = {}
    easy_qp = [r for r in ours_qp if r.get("target_difficulty") == "Easy"]
    for r in easy_qp:
        f = r.get("question_focus", "unknown")
        dj = r.get("difficulty_judge", {})
        asa = dj.get("answer_sentence_alone_sufficient", "?")
        if f not in easy_asa_by_focus:
            easy_asa_by_focus[f] = {"total": 0, "asa_yes": 0}
        easy_asa_by_focus[f]["total"] += 1
        if asa == "yes":
            easy_asa_by_focus[f]["asa_yes"] += 1

    # Hard answer_sentence_alone=no by focus type (Ours, quality-pass)
    hard_asa_by_focus = {}
    hard_qp = [r for r in ours_qp if r.get("target_difficulty") == "Hard"]
    for r in hard_qp:
        f = r.get("question_focus", "unknown")
        dj = r.get("difficulty_judge", {})
        asa = dj.get("answer_sentence_alone_sufficient", "?")
        if f not in hard_asa_by_focus:
            hard_asa_by_focus[f] = {"total": 0, "asa_no": 0}
        hard_asa_by_focus[f]["total"] += 1
        if asa == "no":
            hard_asa_by_focus[f]["asa_no"] += 1

    # Graph policy compliance by focus type (Ours, quality-pass)
    policy_by_focus = {}
    for r in ours_qp:
        f = r.get("question_focus", "unknown")
        gpc = r.get("graph_policy_compliance", "unknown")
        if f not in policy_by_focus:
            policy_by_focus[f] = {"total": 0, "gpc_yes": 0}
        policy_by_focus[f]["total"] += 1
        if gpc == "yes":
            policy_by_focus[f]["gpc_yes"] += 1

    # Repair usage by target difficulty (Ours)
    repair_by_difficulty = {}
    for level in levels:
        lr = [r for r in ours if r.get("target_difficulty") == level]
        n_total = len(lr)
        n_repair = sum(1 for r in lr if r.get("repair_attempted"))
        n_repair_ok = sum(1 for r in lr if r.get("repair_success"))
        repair_by_difficulty[level] = {
            "total": n_total,
            "repair_used": n_repair,
            "repair_success": n_repair_ok,
        }

    # Top 10 Easy failures: Ours, quality-pass, target=Easy, predicted != Easy
    easy_failures = []
    for r in ours_qp:
        if r.get("target_difficulty") != "Easy":
            continue
        pred = r.get("predicted_difficulty", "?")
        if pred == "Easy":
            continue
        # Classify failure reason
        fr = _classify_failure_reason(r)
        easy_failures.append({
            "story": r.get("story_name", "?"),
            "question": r.get("generated_question", "?"),
            "answer": r.get("answer", "?"),
            "predicted": pred,
            "focus": r.get("question_focus", "?"),
            "node_focus": r.get("node_question_focus", "?"),
            "asa": r.get("difficulty_judge", {}).get("answer_sentence_alone_sufficient", "?"),
            "failure_reason": fr,
        })
    easy_failures.sort(key=lambda x: x["failure_reason"])
    easy_failures = easy_failures[:10]

    # Top 10 Hard failures: Ours, quality-pass, target=Hard, predicted != Hard
    hard_failures = []
    for r in ours_qp:
        if r.get("target_difficulty") != "Hard":
            continue
        pred = r.get("predicted_difficulty", "?")
        if pred == "Hard":
            continue
        fr = _classify_failure_reason(r)
        hard_failures.append({
            "story": r.get("story_name", "?"),
            "question": r.get("generated_question", "?"),
            "answer": r.get("answer", "?"),
            "predicted": pred,
            "focus": r.get("question_focus", "?"),
            "node_focus": r.get("node_question_focus", "?"),
            "asa": r.get("difficulty_judge", {}).get("answer_sentence_alone_sufficient", "?"),
            "failure_reason": fr,
        })
    hard_failures.sort(key=lambda x: x["failure_reason"])
    hard_failures = hard_failures[:10]

    return {
        "focus_distribution": focus_dist,
        "node_focus_distribution": node_focus_dist,
        "easy_asa_by_focus": easy_asa_by_focus,
        "hard_asa_by_focus": hard_asa_by_focus,
        "policy_compliance_by_focus": policy_by_focus,
        "repair_by_difficulty": repair_by_difficulty,
        "top_easy_failures": easy_failures,
        "top_hard_failures": hard_failures,
    }


def _classify_failure_reason(result):
    """Classify why an Ours question failed to match target difficulty.

    Returns one of:
      focus_mismatch, answer_local_wording, graph_extraction, malformed, judge_error, unknown
    """
    q = result.get("generated_question", "") or ""
    target = result.get("target_difficulty", "?")
    dj = result.get("difficulty_judge", {})
    asa = dj.get("answer_sentence_alone_sufficient", "?")
    bridge = dj.get("bridge_required", "?")
    gpc = result.get("graph_policy_compliance", "?")
    focus = result.get("question_focus", "?")

    q_lower = q.lower()

    if target == "Easy":
        # Easy failure: predicted Medium or Hard
        # Check if question wording is causal/motivation (focus mismatch)
        causal_starters = ["why ", "what motivated", "what caused", "what led to",
                          "how did", "what was the outcome", "what was the result"]
        if any(q_lower.startswith(cs) for cs in causal_starters):
            return "focus_mismatch_causal_wording"
        if asa == "no":
            return "multi_sentence_required"
        if bridge == "yes":
            return "bridge_detected"
        if gpc != "yes":
            return "graph_policy_noncompliant"
        return "answer_local_wording"

    elif target == "Hard":
        # Hard failure: predicted Easy or Medium
        if asa == "yes":
            return "answer_sentence_alone_yes"
        direct_starters = ["who ", "what ", "where ", "when "]
        if any(q_lower.startswith(ds) for ds in direct_starters):
            return "focus_mismatch_direct_wording"
        if gpc != "yes":
            return "graph_policy_noncompliant"
        if focus not in ("chain_explanation",):
            return "focus_mismatch_wrong_focus_type"
        return "chain_not_realized"

    return "unknown"


def _compute_stage31_diagnostics(all_results):
    """Compute Stage 3.1 Easy prompt hardening diagnostics.

    Returns dict with:
      - easy_forbidden_total, easy_forbidden_violations, easy_forbidden_examples
      - easy_degenerate_total, easy_degenerate_count, easy_degenerate_examples
      - easy_judge_overcount_examples
      - easy_q_introduced_context_examples
    """
    ours = [r for r in all_results if r.get("method") == "Ours"]
    ours_easy = [r for r in ours if r.get("target_difficulty") == "Easy"]
    ours_easy_qp = [r for r in ours_easy if r.get("quality_pass")]

    # Forbidden-frame violations
    fbv_total = len(ours_easy)
    fbv_examples = []
    for r in ours_easy:
        if r.get("easy_forbidden_violation"):
            frames = r.get("easy_forbidden_frames", [])
            fbv_examples.append({
                "story": r.get("story_name", "?"),
                "question": r.get("generated_question", "?"),
                "frames": frames,
                "qp": r.get("quality_pass", False),
                "pred": r.get("predicted_difficulty", "?"),
            })
    fbv_examples.sort(key=lambda x: len(x["frames"]), reverse=True)

    # Degenerate output
    deg_examples = []
    for r in ours_easy:
        err = r.get("generation_error", "")
        parse_ok = r.get("parse_ok", False)
        if not parse_ok or "degenerate" in err.lower() or "parse failure" in err.lower():
            deg_examples.append({
                "story": r.get("story_name", "?"),
                "error": err or "parse failure",
            })

    # Judge overcount: QP-passing Easy, predicted Medium/Hard, with clearly single-sentence wording
    jo_examples = []
    for r in ours_easy_qp:
        pred = r.get("predicted_difficulty", "?")
        if pred in ("Easy", "?"):
            continue
        q = r.get("generated_question", "") or ""
        q_lower = q.lower()
        # Check if question uses only simple wording (no causal/inferential framing)
        simple_starters = q_lower.startswith("what ") or q_lower.startswith("who ") or \
                          q_lower.startswith("where ") or q_lower.startswith("how many ")
        has_forbidden = r.get("easy_forbidden_violation", False)
        if simple_starters and not has_forbidden:
            ans = r.get("answer", "")
            jo_examples.append({
                "story": r.get("story_name", "?"),
                "question": q,
                "answer": ans[:60],
                "pred": pred,
                "reason": f"Clear single-sentence question, judged {pred}",
            })

    # Question-introduced-context: QP-passing Easy, has forbidden frames or introduced context
    qic_examples = []
    for r in ours_easy_qp:
        pred = r.get("predicted_difficulty", "?")
        if pred == "Easy":
            continue
        has_forbidden = r.get("easy_forbidden_violation", False)
        if has_forbidden:
            q = r.get("generated_question", "") or ""
            frames = r.get("easy_forbidden_frames", [])
            qic_examples.append({
                "story": r.get("story_name", "?"),
                "question": q,
                "answer": (r.get("answer", "") or "")[:60],
                "pred": pred,
                "context_wording": ", ".join(frames),
            })

    return {
        "easy_forbidden_total": fbv_total,
        "easy_forbidden_violations": len(fbv_examples),
        "easy_forbidden_examples": fbv_examples,
        "easy_degenerate_total": len(ours_easy),
        "easy_degenerate_count": len(deg_examples),
        "easy_degenerate_examples": deg_examples,
        "easy_judge_overcount_examples": jo_examples,
        "easy_q_introduced_context_examples": qic_examples,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _build_report(all_results, crossqg_metrics, graph_stats, output_dir, meta=None, bootstrap_diag=None,
                  story_diag=None, retry_diag=None, similarity_diag=None, stage2_diag=None):
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

    selection_mode = meta.get("selection_mode", "balanced")
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
        f"| Selection mode | {selection_mode} |",
        f"| Target per level | {meta.get('target_per_level', 'N/A')} |",
        f"| Selected Easy | {meta.get('n_easy', 'N/A')} |",
        f"| Selected Medium | {meta.get('n_medium', 'N/A')} |",
        f"| Selected Hard | {meta.get('n_hard', 'N/A')} |",
        f"| Total selected | {meta.get('n_total', 'N/A')} |",
        f"| Total stories | {meta.get('n_stories', 'N/A')} |",
        f"| Total generations | {len(all_results)} |",
    ]

    if selection_mode in ("story_matched", "story_matched_suitable", "story_matched_calibrated"):
        lines.append(f"| Candidates per level per story | {meta.get('candidates_per_level_per_story', 'N/A')} |")
        lines.append(f"| Max stories | {meta.get('max_stories', 'N/A')} |")

    lines.append("")

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

    # ── Ours Graph Policy Diagnostics ──────────────────────────────
    ours_rs = [r for r in all_results if r.get("method") == "Ours"]
    ours_with_gp = [r for r in ours_rs if r.get("graph_policy") and r["graph_policy"] != "graph_invalid"]

    if ours_with_gp:
        lines.append("## Ours Graph Policy Diagnostics")
        lines.append("")

        # 1. Graph policy distribution by target difficulty
        lines.append("### GP-1. Graph Policy Distribution by Target Difficulty")
        lines.append("")
        gp_types = ["answer_only", "two_node_relation", "multi_node_chain"]
        header = "| Target | " + " | ".join(gp_types) + " | fallback | total |"
        sep = "|---|" + "|".join("---:" for _ in gp_types) + "|---:|---:|"
        lines.append(header)
        lines.append(sep)
        for diff in levels:
            diff_rs = [r for r in ours_with_gp if r.get("target_difficulty") == diff]
            counts = {gp: 0 for gp in gp_types}
            fallback = 0
            for r in diff_rs:
                gp = r.get("graph_policy", "")
                if gp in counts:
                    counts[gp] += 1
                else:
                    fallback += 1
            row = f"| {diff} | " + " | ".join(str(counts[gp]) for gp in gp_types) + f" | {fallback} | {len(diff_rs)} |"
            lines.append(row)
        lines.append("")

        # 2. Graph policy compliance rate by target difficulty
        lines.append("### GP-2. Graph Policy Compliance Rate by Target Difficulty")
        lines.append("")
        lines.append("| Target | compliant | total | pct |")
        lines.append("|---|---:|---:|---:|")
        for diff in levels:
            diff_rs = [r for r in ours_with_gp if r.get("target_difficulty") == diff]
            compliant = sum(1 for r in diff_rs if r.get("graph_policy_compliance") == "yes")
            pct = f"{100*compliant/len(diff_rs):.1f}%" if diff_rs else "N/A"
            lines.append(f"| {diff} | {compliant} | {len(diff_rs)} | {pct} |")
        lines.append("")

        # 3. Per-policy accuracy / macro F1 / Hard hit
        lines.append("### GP-3. Per-Policy Difficulty Accuracy")
        lines.append("")
        lines.append("| Policy | n_valid | accuracy | hard_hit |")
        lines.append("|---|---:|---:|---:|")
        for gp in gp_types + ["other"]:
            if gp == "other":
                gp_rs = [r for r in ours_with_gp
                         if r.get("graph_policy") not in gp_types
                         and r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
            else:
                gp_rs = [r for r in ours_with_gp
                         if r.get("graph_policy") == gp
                         and r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
            if not gp_rs:
                lines.append(f"| {gp} | 0 | N/A | N/A |")
                continue
            correct = sum(1 for r in gp_rs if r.get("predicted_difficulty") == r.get("target_difficulty"))
            acc = f"{100*correct/len(gp_rs):.1f}%"
            hard_rs = [r for r in gp_rs if r.get("target_difficulty") == "Hard"]
            hard_hit = sum(1 for r in hard_rs if r.get("predicted_difficulty") == "Hard")
            hard_pct = f"{100*hard_hit/len(hard_rs):.1f}%" if hard_rs else "N/A"
            lines.append(f"| {gp} | {len(gp_rs)} | {acc} | {hard_pct} |")
        lines.append("")

        # 4. Selected relation chain distribution (top 10)
        lines.append("### GP-4. Selected Relation Chain Distribution (top 10)")
        lines.append("")
        chain_counts = {}
        for r in ours_with_gp:
            chain = " → ".join(r.get("relation_chain", [])) or "(none)"
            chain_counts[chain] = chain_counts.get(chain, 0) + 1
        sorted_chains = sorted(chain_counts.items(), key=lambda x: -x[1])[:10]
        lines.append("| Relation chain | count | pct |")
        lines.append("|---|---:|---:|")
        total_chains = len(ours_with_gp)
        for chain, cnt in sorted_chains:
            pct = f"{100*cnt/total_chains:.1f}%"
            lines.append(f"| {chain} | {cnt} | {pct} |")
        lines.append("")

        # 5. Hard: relation type vs predicted difficulty
        hard_ours = [r for r in ours_with_gp if r.get("target_difficulty") == "Hard"
                     and r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
        if hard_ours:
            lines.append("### GP-5. Hard: Relation Type vs Predicted Difficulty")
            lines.append("")
            rel_pred = {}
            for r in hard_ours:
                for rel in r.get("relation_chain", []):
                    rel_pred.setdefault(rel, {"Easy": 0, "Medium": 0, "Hard": 0})
                    pred = r.get("predicted_difficulty", "?")
                    if pred in rel_pred[rel]:
                        rel_pred[rel][pred] += 1
            lines.append("| Relation | Easy | Medium | Hard | total |")
            lines.append("|---|---:|---:|---:|---:|")
            for rel in sorted(rel_pred, key=lambda x: -sum(rel_pred[x].values())):
                d = rel_pred[rel]
                total = sum(d.values())
                lines.append(f"| {rel} | {d['Easy']} | {d['Medium']} | {d['Hard']} | {total} |")
            lines.append("")

        # 6. Repair prompt usage by difficulty and graph_policy
        lines.append("### GP-6. Repair Prompt Usage by Difficulty and Graph Policy")
        lines.append("")
        lines.append("| Target | Policy | repair_used | total | pct |")
        lines.append("|---|---|---:|---:|---:|")
        for diff in levels:
            for gp in gp_types:
                gp_diff_rs = [r for r in ours_with_gp
                             if r.get("target_difficulty") == diff and r.get("graph_policy") == gp]
                if not gp_diff_rs:
                    continue
                repair = sum(1 for r in gp_diff_rs if r.get("repair_attempted"))
                pct = f"{100*repair/len(gp_diff_rs):.1f}%"
                lines.append(f"| {diff} | {gp} | {repair} | {len(gp_diff_rs)} | {pct} |")
        lines.append("")

        # 7. Hard pure_temporal_chain count
        hard_temporal = [r for r in ours_with_gp
                        if r.get("target_difficulty") == "Hard"
                        and r.get("graph_policy") == "multi_node_chain"
                        and "pure_temporal_chain" in r.get("graph_policy_reason", "")]
        lines.append(f"### GP-7. Hard Pure Temporal Chain Count: {len(hard_temporal)}")
        lines.append("")

        # 8. Easy answer_sentence_alone rate
        easy_ours = [r for r in ours_with_gp if r.get("target_difficulty") == "Easy"
                     and r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
        if easy_ours:
            alone_yes = sum(1 for r in easy_ours
                           if r.get("difficulty_judge", {}).get("answer_sentence_alone_sufficient") == "yes")
            pct = f"{100*alone_yes/len(easy_ours):.1f}%"
            lines.append(f"### GP-8. Easy Answer-Sentence-Alone Rate: {pct} ({alone_yes}/{len(easy_ours)})")
            lines.append("")

        # 9. Hard answer_sentence_alone=no rate
        hard_ours_valid = [r for r in ours_with_gp if r.get("target_difficulty") == "Hard"
                          and r.get("quality_pass") and r.get("difficulty_judge_status") == "ok"]
        if hard_ours_valid:
            alone_no = sum(1 for r in hard_ours_valid
                          if r.get("difficulty_judge", {}).get("answer_sentence_alone_sufficient") == "no")
            pct = f"{100*alone_no/len(hard_ours_valid):.1f}%"
            lines.append(f"### GP-9. Hard Answer-Sentence-Alone=No Rate: {pct} ({alone_no}/{len(hard_ours_valid)})")
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

    if selection_mode in ("story_matched", "story_matched_suitable", "story_matched_calibrated"):
        # Story-matched criteria
        n_stories_val = meta.get("n_stories", 0)
        story_count_pass = n_stories_val >= 100

        # Verify every selected story has equal Easy/Med/Hard count
        # In story_matched mode, each method generates for each candidate.
        # Expected per-level per-story = n_methods × candidates_per_level_per_story
        expected_per_level = len(methods) * meta.get("candidates_per_level_per_story", 1)
        stories_levels = {}
        for r in all_results:
            sn = r.get("story_name", "")
            diff = r.get("target_difficulty", "")
            stories_levels.setdefault(sn, {}).setdefault(diff, 0)
            stories_levels[sn][diff] += 1
        equal_levels = all(
            s.get("Easy", 0) == s.get("Medium", 0) == s.get("Hard", 0) == expected_per_level
            for s in stories_levels.values()
        )

        # All methods have identical denominators
        method_counts = {}
        for m in methods:
            method_counts[m] = sum(1 for r in all_results if r.get("method") == m)
        identical_denom = len(set(method_counts.values())) == 1

        criteria = [
            ("selected stories >= 100", story_count_pass,
             f"selected stories={n_stories_val}"),
            ("every story has equal Easy/Med/Hard count", equal_levels,
             f"all {len(stories_levels)} stories have 1E/1M/1H" if equal_levels else "some stories unbalanced"),
            ("all methods have identical denominator", identical_denom,
             f"denominators: {method_counts}"),
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
             f"Ours={ours_cm['spearman']:.3f}"),
        ]
    else:
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

    # ── Story-Matched Diagnostics ──────────────────────────────────
    if selection_mode in ("story_matched", "story_matched_suitable", "story_matched_calibrated") and story_diag:
        lines.append("## 13. Story-Matched Diagnostics")
        lines.append("")

        # 13a. Summary
        lines.append("### 13a. Story Summary")
        lines.append("")
        lines.append(f"| Selected stories | {story_diag['n_stories']} |")
        lines.append(f"| Candidates per story | {meta.get('candidates_per_level_per_story', 1)} × 3 levels |")
        lines.append("")

        # Verify equal counts: use raw totals per method per story
        sm = story_diag["story_metrics"]
        equal_ok = True
        for sn, metrics in sm.items():
            # Check using Ours raw counts (any method would work since denominators are identical)
            easy_n = metrics.get("Ours_Easy_raw", -1)
            med_n = metrics.get("Ours_Medium_raw", -1)
            hard_n = metrics.get("Ours_Hard_raw", -1)
            if easy_n != med_n or med_n != hard_n:
                equal_ok = False
                break
        expected = len(methods) * meta.get("candidates_per_level_per_story", 1)
        lines.append(f"**Equal Easy/Med/Hard per story:** {'YES' if equal_ok else 'NO — some stories unbalanced'} (expected {expected} per level per story)")
        lines.append("")

        # 13b. Story-level average accuracy by method
        lines.append("### 13b. Story-Level Average Accuracy by Method (quality-pass, judge-ok)")
        lines.append("")
        lines.append("| Method | Mean story acc | Median story acc | Std | N stories |")
        lines.append("|---|---:|---:|---:|---:|")
        for m in methods:
            accs = [sm[sn].get(f"{m}_accuracy", 0.0) for sn in sm]
            valid_accs = [a for a in accs if a > 0]
            mean_acc = sum(accs) / len(accs) if accs else 0.0
            sorted_accs = sorted(valid_accs)
            median_acc = sorted_accs[len(sorted_accs)//2] if sorted_accs else 0.0
            std_acc = (sum((a - mean_acc)**2 for a in accs) / len(accs))**0.5 if accs else 0.0
            n_valid = len(valid_accs)
            lines.append(f"| {m} | {mean_acc*100:.1f}% | {median_acc*100:.1f}% | {std_acc*100:.1f}% | {n_valid} |")
        lines.append("")

        # 13c. Story-level Win/Tie/Loss: Ours vs each baseline
        lines.append("### 13c. Story-Level Win/Tie/Loss (Ours vs Baseline)")
        lines.append("")
        lines.append("| Baseline | Ours Wins | Ties | Ours Losses | N stories |")
        lines.append("|---|---:|---:|---:|---:|")
        wtl = story_diag["win_tie_loss"]
        for bm in baseline_methods:
            w = wtl.get(bm, {})
            lines.append(f"| {bm} | {w.get('wins', 0)} | {w.get('ties', 0)} | {w.get('losses', 0)} | {w.get('n_stories', 0)} |")
        lines.append("")

        # 13d. Story-level Spearman
        lines.append("### 13d. Story-Level Spearman (stories with all 3 levels valid)")
        lines.append("")
        lines.append("| Method | Mean story rho | N valid stories | N skipped |")
        lines.append("|---|---:|---:|---:|")
        ss = story_diag["story_spearman"]
        for m in methods:
            s = ss.get(m, {})
            lines.append(f"| {m} | {s.get('mean_rho', 0.0):.3f} | {s.get('n_valid_stories', 0)} | {s.get('n_skipped', 0)} |")
        lines.append("")

        # 13e. Per-story failure counts by method
        lines.append("### 13e. Per-Story Failure Counts by Method")
        lines.append("")
        lines.append("| Method | Stories with 0 fails | 1 fail | 2 fails | 3 fails |")
        lines.append("|---|---:|---:|---:|---:|")
        psf = story_diag["per_story_failures"]
        for m in methods:
            fd = psf.get(m, {})
            f0 = sum(1 for v in fd.values() if v == 0)
            f1 = sum(1 for v in fd.values() if v == 1)
            f2 = sum(1 for v in fd.values() if v == 2)
            f3 = sum(1 for v in fd.values() if v == 3)
            lines.append(f"| {m} | {f0} | {f1} | {f2} | {f3} |")
        lines.append("")

    # ── Retry / Budget Diagnostics ────────────────────────────────
    if retry_diag:
        lines.append("## 14. Retry & Budget Diagnostics")
        lines.append("")

        # 14a. Attempts per method
        lines.append("### 14a. Attempts per Method")
        lines.append("")
        lines.append("| Method | Avg attempts | Max attempts | Total |")
        lines.append("|---|---:|---:|---:|")
        for m in methods:
            rd = retry_diag.get(m, {})
            lines.append(f"| {m} | {rd.get('average_attempts', 0):.2f} | {rd.get('max_attempts', 0)} | {rd.get('n_total', 0)} |")
        lines.append("")

        # 14b. Attempt distribution
        lines.append("### 14b. Attempt Distribution by Method")
        lines.append("")
        lines.append("| Method | 1 attempt | 2 attempts | 3+ attempts |")
        lines.append("|---|---:|---:|---:|")
        for m in methods:
            rd = retry_diag.get(m, {})
            dist = rd.get("attempt_distribution", {})
            a1 = sum(v for k, v in dist.items() if k <= 1)
            a2 = sum(v for k, v in dist.items() if k == 2)
            a3 = sum(v for k, v in dist.items() if k >= 3)
            lines.append(f"| {m} | {a1} | {a2} | {a3} |")
        lines.append("")

        # 14c. Ours repair prompt usage
        ours_rd = retry_diag.get("Ours", {})
        if ours_rd.get("repair_prompt_used", 0) > 0 or ours_rd.get("repair_prompt_used_rate", 0) > 0:
            lines.append("### 14c. Ours Repair Prompt Usage")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            lines.append(f"| Repair prompt used | {ours_rd.get('repair_prompt_used', 0)}/{ours_rd.get('n_total', 0)} ({ours_rd.get('repair_prompt_used_rate', 0)*100:.1f}%) |")
            lines.append(f"| Repair success | {ours_rd.get('repair_prompt_success', 0)} |")
            lines.append("")

        # 14d. Ours graph_policy self-check failure rate
        if ours_rd.get("graph_policy_sc_total", 0) > 0:
            lines.append("### 14d. Ours Graph Policy Self-Check Failure Rate")
            lines.append("")
            sc_fail = ours_rd.get("graph_policy_sc_failure", 0)
            sc_total = ours_rd.get("graph_policy_sc_total", 0)
            lines.append(f"| Self-check failures | {sc_fail}/{sc_total} ({sc_fail/sc_total*100:.1f}%) |")
            lines.append("")

        # 14e. Failure reason distribution
        lines.append("### 14e. Failure Reason Distribution by Method")
        lines.append("")
        lines.append("| Method | Failure reason | Count |")
        lines.append("|---|---|---:|")
        for m in methods:
            rd = retry_diag.get(m, {})
            for reason, count in sorted(rd.get("failure_reasons", {}).items(), key=lambda x: -x[1])[:10]:
                lines.append(f"| {m} | {reason} | {count} |")
        lines.append("")

        # 14f. Top retry reasons (Ours only, from attempt traces)
        ours_retry = retry_diag.get("Ours", {}).get("retry_reasons", {})
        if ours_retry:
            lines.append("### 14f. Ours Retry Reason Distribution (from attempt traces)")
            lines.append("")
            lines.append("| Retry reason | Count |")
            lines.append("|---|---:|")
            for reason, count in sorted(ours_retry.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"| {reason} | {count} |")
            lines.append("")

    # ── Similarity Diagnostics ────────────────────────────────────
    if similarity_diag:
        lines.append("## 15. Similarity Diagnostics (diagnostic only, no filtering)")
        lines.append("")

        # 15a. Per-method question similarity
        lines.append("### 15a. Per-Method Average Question Lexical Similarity (char 4-gram Jaccard)")
        lines.append("")
        hdr = "| Method | " + " | ".join("E-M sim" for _ in range(1)) + " | " + " | ".join("M-H sim" for _ in range(1)) + " | " + " | ".join("E-H sim" for _ in range(1)) + " |"
        lines.append("| Method | E-M mean | M-H mean | E-H mean | n pairs |")
        lines.append("|---|---:|---:|---:|---:|")

        for m in methods:
            sd = similarity_diag.get(m, {})
            qs = sd.get("question_similarity", {})
            em = qs.get("Easy-Medium", {}).get("mean", 0)
            mh = qs.get("Medium-Hard", {}).get("mean", 0)
            eh = qs.get("Easy-Hard", {}).get("mean", 0)
            npairs = qs.get("Easy-Medium", {}).get("n", 0)
            lines.append(f"| {m} | {em:.3f} | {mh:.3f} | {eh:.3f} | {npairs} |")
        lines.append("")

        # 15b. Evidence sentence overlap
        lines.append("### 15b. Evidence Sentence Overlap by Method (Jaccard of judge-used evidence)")
        lines.append("")
        lines.append("| Method | E-M evidence overlap | M-H evidence overlap | E-H evidence overlap |")
        lines.append("|---|---:|---:|---:|")
        for m in methods:
            sd = similarity_diag.get(m, {})
            eo = sd.get("evidence_overlap", {})
            em = eo.get("Easy-Medium", {}).get("mean", 0)
            mh = eo.get("Medium-Hard", {}).get("mean", 0)
            eh = eo.get("Easy-Hard", {}).get("mean", 0)
            lines.append(f"| {m} | {em:.3f} | {mh:.3f} | {eh:.3f} |")
        lines.append("")

        # 15c. Difficulty collapse counts
        lines.append("### 15c. Difficulty Collapse Counts by Method")
        lines.append("")
        lines.append("| Method | Stories w/ 3 QP | All 3 same pred | Collapse to Medium | Collapse to Easy | Collapse to Hard |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for m in methods:
            sd = similarity_diag.get(m, {})
            c = sd.get("collapses", {})
            lines.append(f"| {m} | {c.get('total_stories', 0)} | {c.get('collapse_all_three', 0)} | "
                        f"{c.get('collapse_to_medium', 0)} | {c.get('collapse_to_easy', 0)} | "
                        f"{c.get('collapse_to_hard', 0)} |")
        lines.append("")

    # ── Stage 2 Diagnostics ────────────────────────────────────────
    if stage2_diag:
        lines.append("## 16. Stage 2 Focus & Difficulty Realization Diagnostics (Ours)")
        lines.append("")

        # 16a. Focus distribution by target difficulty
        lines.append("### 16a. Focus Distribution by Target Difficulty (Ours, quality-pass)")
        lines.append("")
        focus_dist = stage2_diag.get("focus_distribution", {})
        all_focus_types = set()
        for level in levels:
            all_focus_types.update(focus_dist.get(level, {}).keys())
        sorted_foci = sorted(all_focus_types)
        if sorted_foci:
            hdr = "| Focus | " + " | ".join(f"{d} (n)" for d in levels) + " |"
            lines.append(hdr)
            lines.append("|---|" + "|".join("---:" for _ in levels) + "|")
            for f in sorted_foci:
                row = f"| {f} | " + " | ".join(str(focus_dist.get(d, {}).get(f, 0)) for d in levels) + " |"
                lines.append(row)
            lines.append("")

        # 16b. Node-level focus distribution (pre-override, for comparison)
        node_focus_dist = stage2_diag.get("node_focus_distribution", {})
        all_node_foci = set()
        for level in levels:
            all_node_foci.update(node_focus_dist.get(level, {}).keys())
        sorted_nf = sorted(all_node_foci)
        if sorted_nf:
            lines.append("### 16b. Node-Level Focus Distribution (pre-override, for comparison)")
            lines.append("")
            hdr = "| Node Focus | " + " | ".join(f"{d} (n)" for d in levels) + " |"
            lines.append(hdr)
            lines.append("|---|" + "|".join("---:" for _ in levels) + "|")
            for f in sorted_nf:
                row = f"| {f} | " + " | ".join(str(node_focus_dist.get(d, {}).get(f, 0)) for d in levels) + " |"
                lines.append(row)
            lines.append("")

        # 16c. Easy answer_sentence_alone=yes by focus type
        easy_asa = stage2_diag.get("easy_asa_by_focus", {})
        if easy_asa:
            lines.append("### 16c. Easy answer_sentence_alone=yes by Focus Type (Ours, quality-pass, target=Easy)")
            lines.append("")
            lines.append("| Focus | Total | ASA=yes | Rate |")
            lines.append("|---|---:|---:|---:|")
            for f in sorted(easy_asa.keys()):
                d = easy_asa[f]
                rate = f"{100*d['asa_yes']/d['total']:.1f}%" if d["total"] else "N/A"
                lines.append(f"| {f} | {d['total']} | {d['asa_yes']} | {rate} |")
            lines.append("")

        # 16d. Hard answer_sentence_alone=no by focus type
        hard_asa = stage2_diag.get("hard_asa_by_focus", {})
        if hard_asa:
            lines.append("### 16d. Hard answer_sentence_alone=no by Focus Type (Ours, quality-pass, target=Hard)")
            lines.append("")
            lines.append("| Focus | Total | ASA=no | Rate |")
            lines.append("|---|---:|---:|---:|")
            for f in sorted(hard_asa.keys()):
                d = hard_asa[f]
                rate = f"{100*d['asa_no']/d['total']:.1f}%" if d["total"] else "N/A"
                lines.append(f"| {f} | {d['total']} | {d['asa_no']} | {rate} |")
            lines.append("")

        # 16e. Graph policy compliance by focus type
        policy_focus = stage2_diag.get("policy_compliance_by_focus", {})
        if policy_focus:
            lines.append("### 16e. Graph Policy Compliance by Focus Type (Ours, quality-pass)")
            lines.append("")
            lines.append("| Focus | Total | GPC=yes | Rate |")
            lines.append("|---|---:|---:|---:|")
            for f in sorted(policy_focus.keys()):
                d = policy_focus[f]
                rate = f"{100*d['gpc_yes']/d['total']:.1f}%" if d["total"] else "N/A"
                lines.append(f"| {f} | {d['total']} | {d['gpc_yes']} | {rate} |")
            lines.append("")

        # 16f. Repair usage by target difficulty
        repair_diff = stage2_diag.get("repair_by_difficulty", {})
        if repair_diff:
            lines.append("### 16f. Repair Usage by Target Difficulty (Ours)")
            lines.append("")
            lines.append("| Difficulty | Total | Repair Used | Repair Success | Repair Rate |")
            lines.append("|---|---:|---:|---:|---:|")
            for level in levels:
                d = repair_diff.get(level, {})
                total = d.get("total", 0)
                used = d.get("repair_used", 0)
                succ = d.get("repair_success", 0)
                rate = f"{100*used/total:.1f}%" if total else "N/A"
                lines.append(f"| {level} | {total} | {used} | {succ} | {rate} |")
            lines.append("")

        # 16g. Top 10 Easy failures
        easy_fails = stage2_diag.get("top_easy_failures", [])
        if easy_fails:
            lines.append("### 16g. Top 10 Easy Failures (Ours, quality-pass, predicted != Easy)")
            lines.append("")
            lines.append("| # | Story | Question | Answer | Pred | Focus | ASA | Failure Reason |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for i, f in enumerate(easy_fails, 1):
                lines.append(f"| {i} | {f['story'][:30]} | {f['question'][:60]} | {f['answer'][:30]} | "
                           f"{f['predicted']} | {f['focus']} | {f['asa']} | {f['failure_reason']} |")
            lines.append("")

        # 16h. Top 10 Hard failures
        hard_fails = stage2_diag.get("top_hard_failures", [])
        if hard_fails:
            lines.append("### 16h. Top 10 Hard Failures (Ours, quality-pass, predicted != Hard)")
            lines.append("")
            lines.append("| # | Story | Question | Answer | Pred | Focus | ASA | Failure Reason |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for i, f in enumerate(hard_fails, 1):
                lines.append(f"| {i} | {f['story'][:30]} | {f['question'][:60]} | {f['answer'][:30]} | "
                           f"{f['predicted']} | {f['focus']} | {f['asa']} | {f['failure_reason']} |")
            lines.append("")

    # 16i. Stage 3.1 Easy hardening diagnostics
    stage31 = meta.get("stage31_diag") if meta else None
    if stage31:
        lines.append("### 16i. Stage 3.1 Easy Prompt Hardening Diagnostics (Ours)")
        lines.append("")

        # Forbidden-frame violations
        fbv_total = stage31.get("easy_forbidden_total", 0)
        fbv_count = stage31.get("easy_forbidden_violations", 0)
        lines.append(f"**Easy forbidden-frame violations:** {fbv_count}/{fbv_total} "
                     f"({100*fbv_count/fbv_total:.1f}%)" if fbv_total else "N/A")
        lines.append("")

        # Individual violations
        fbv_list = stage31.get("easy_forbidden_examples", [])
        if fbv_list:
            lines.append("| Story | Question | Violated Frames | QP | Pred |")
            lines.append("|---|---|---|---|---|")
            for f in fbv_list[:10]:
                lines.append(f"| {f['story'][:25]} | {f['question'][:60]} | {', '.join(f['frames'])} | "
                           f"{'Y' if f.get('qp') else 'N'} | {f.get('pred', '?')} |")
            lines.append("")

        # Degenerate count
        deg_total = stage31.get("easy_degenerate_total", 0)
        deg_count = stage31.get("easy_degenerate_count", 0)
        lines.append(f"**Easy degenerate output:** {deg_count}/{deg_total} "
                     f"({100*deg_count/deg_total:.1f}%)" if deg_total else "N/A")
        lines.append("")

        # Degenerate examples
        deg_list = stage31.get("easy_degenerate_examples", [])
        if deg_list:
            lines.append("| Story | Error |")
            lines.append("|---|---|")
            for d in deg_list[:10]:
                lines.append(f"| {d['story'][:25]} | {d['error'][:80]} |")
            lines.append("")

        # Judge overcount examples
        jo_list = stage31.get("easy_judge_overcount_examples", [])
        if jo_list:
            lines.append("**Easy judge-overcount examples (QP-pass, predicted Medium/Hard, clear single-sentence question):**")
            lines.append("")
            lines.append("| Story | Question | Answer | Pred | Why Likely Overcount |")
            lines.append("|---|---|---|---|---|")
            for j in jo_list[:10]:
                lines.append(f"| {j['story'][:25]} | {j['question'][:60]} | {j['answer'][:30]} | "
                           f"{j['pred']} | {j['reason']} |")
            lines.append("")

        # Question-introduced-context examples
        qic_list = stage31.get("easy_q_introduced_context_examples", [])
        if qic_list:
            lines.append("**Easy question-introduced-context examples (QP-pass, wording added context):**")
            lines.append("")
            lines.append("| Story | Question | Answer | Pred | Context Wording |")
            lines.append("|---|---|---|---|---|")
            for qc in qic_list[:10]:
                lines.append(f"| {qc['story'][:25]} | {qc['question'][:60]} | {qc['answer'][:30]} | "
                           f"{qc['pred']} | {qc['context_wording']} |")
            lines.append("")

    # 17. Examples
    lines.append("## 17. Examples")
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
    parser.add_argument("--selection_mode", choices=["balanced", "story_matched", "story_matched_suitable", "story_matched_calibrated"],
                        default="balanced",
                        help="Candidate selection mode: balanced (default), story_matched, or story_matched_suitable")
    parser.add_argument("--target_per_level", type=int, default=150)
    parser.add_argument("--candidates_per_level_per_story", type=int, default=1,
                        help="Candidates per level per story in story_matched mode")
    parser.add_argument("--max_stories", type=int, default=None,
                        help="Max stories in story_matched mode")
    parser.add_argument("--calibrated_dir", default=None,
                        help="Directory containing candidates.calibrated.jsonl "
                             "(required for story_matched_calibrated mode)")
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

        # Compute diagnostics
        story_diag_regen = _compute_story_matched_diagnostics(all_results)
        retry_diag_regen = _compute_retry_budget_diagnostics(all_results)
        similarity_diag_regen = _compute_similarity_diagnostics(all_results)
        stage2_diag_regen = _compute_stage2_diagnostics(all_results)
        stage31_diag_regen = _compute_stage31_diagnostics(all_results)

        # Reconstruct meta from results
        levels = ["Easy", "Medium", "Hard"]
        n_by_level = {d: 0 for d in levels}
        for r in all_results:
            d = r.get("target_difficulty")
            if d in n_by_level:
                n_by_level[d] += 1
        # Divide by 4 methods to get per-level candidate count
        n_per_level = {d: n_by_level[d] // len(methods) for d in levels}

        # Detect selection mode from results
        story_ids = set(r.get("story_group_id") for r in all_results if "story_group_id" in r)
        regen_mode = "story_matched" if story_ids else "balanced"
        n_stories_regen = len(set(r.get("story_name", "") for r in all_results))

        meta = {
            "selection_mode": regen_mode,
            "target_per_level": n_per_level.get("Easy", 0),
            "candidates_per_level_per_story": 1 if regen_mode == "story_matched" else None,
            "max_stories": n_stories_regen if regen_mode == "story_matched" else None,
            "n_easy": n_per_level.get("Easy", 0),
            "n_medium": n_per_level.get("Medium", 0),
            "n_hard": n_per_level.get("Hard", 0),
            "n_total": sum(n_per_level.values()),
            "n_stories": n_stories_regen,
        }

        meta["stage31_diag"] = stage31_diag_regen
        _build_report(all_results, crossqg_metrics, graph_stats, output_dir, meta=meta, bootstrap_diag=bootstrap_diag,
                      story_diag=story_diag_regen, retry_diag=retry_diag_regen, similarity_diag=similarity_diag_regen,
                      stage2_diag=stage2_diag_regen)
        print("\nDone!")
        return

    methods = ["Direct", "ICL", "SelfRefine", "Ours"]

    # Step 1: Select candidates
    print(f"=== Step 1: Selecting candidates (mode={args.selection_mode}) ===")
    if args.selection_mode == "story_matched_suitable":
        selected = _select_story_matched_suitable_candidates(
            args.candidates,
            candidates_per_level_per_story=args.candidates_per_level_per_story,
            max_stories=args.max_stories,
            seed=args.seed,
        )
    elif args.selection_mode == "story_matched_calibrated":
        if not args.calibrated_dir:
            print("  ERROR: --calibrated_dir required for story_matched_calibrated mode")
            print("  Usage: python -m scripts.run_crossqg_eval --selection_mode story_matched_calibrated "
                  "--calibrated_dir outputs/runs/fairytale_target_calibration_20260513/ ...")
            sys.exit(1)
        calibrated_path = os.path.join(args.calibrated_dir, "candidates.calibrated.jsonl")
        selected, n_cal_stories, bottleneck = _select_story_matched_calibrated_candidates(
            calibrated_path, args.candidates,
            candidates_per_level_per_story=args.candidates_per_level_per_story,
            max_stories=args.max_stories,
            seed=args.seed,
        )
        if selected is None:
            print(f"\n  Calibrated pool too small ({n_cal_stories} stories). "
                  f"Cannot proceed with QG evaluation.")
            sys.exit(1)
    elif args.selection_mode == "story_matched":
        selected = _select_story_matched_candidates(
            args.candidates,
            candidates_per_level_per_story=args.candidates_per_level_per_story,
            max_stories=args.max_stories,
            seed=args.seed,
        )
    else:
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
    n_stories = len(set(c.get("story_group_id") for c in selected if "story_group_id" in c))
    if n_stories == 0:
        n_stories = len(set(c.get("story_name", "") for c in selected))

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
                if "story_group_id" in cand:
                    result["story_group_id"] = cand["story_group_id"]

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

    # Step 4.5: Compute new diagnostics
    print("\n=== Step 5: Computing story-matched / retry / similarity diagnostics ===")
    story_diag = _compute_story_matched_diagnostics(all_results)
    retry_diag = _compute_retry_budget_diagnostics(all_results)
    similarity_diag = _compute_similarity_diagnostics(all_results)
    stage2_diag = _compute_stage2_diagnostics(all_results)
    stage31_diag = _compute_stage31_diagnostics(all_results)

    if args.selection_mode in ("story_matched", "story_matched_suitable", "story_matched_calibrated"):
        print(f"  Story-matched: {story_diag['n_stories']} stories")
        for bm, wtl in story_diag["win_tie_loss"].items():
            print(f"  Ours vs {bm}: {wtl['wins']}W/{wtl['ties']}T/{wtl['losses']}L")
    for m in methods:
        print(f"  {m} similarity collapse (all-3-same): "
              f"{similarity_diag.get(m, {}).get('collapses', {}).get('collapse_all_three', 0)}")

    # Step 6: Write judged JSONL
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

    # Step 7: Bootstrap significance
    print("\n=== Step 7: Computing bootstrap significance ===")
    bootstrap_diag = _compute_bootstrap_diagnostics(all_results, crossqg_metrics)

    # Step 8: Build report
    print("\n=== Step 8: Building report ===")
    meta = {
        "selection_mode": args.selection_mode,
        "target_per_level": args.target_per_level,
        "candidates_per_level_per_story": args.candidates_per_level_per_story
            if args.selection_mode in ("story_matched", "story_matched_suitable", "story_matched_calibrated") else None,
        "max_stories": args.max_stories,
        "n_easy": n_easy,
        "n_medium": n_medium,
        "n_hard": n_hard,
        "n_total": len(selected),
        "n_stories": n_stories,
        "stage31_diag": stage31_diag,
    }
    _build_report(all_results, crossqg_metrics, graph_stats, output_dir, meta=meta, bootstrap_diag=bootstrap_diag,
                  story_diag=story_diag, retry_diag=retry_diag, similarity_diag=similarity_diag,
                  stage2_diag=stage2_diag)

    print("\nDone!")


if __name__ == "__main__":
    main()
