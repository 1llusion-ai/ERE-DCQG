"""Hard Rescue Pilot: multi-strategy Hard question generation with full evaluation.

Expands the Hard candidate pool, generates K candidates per path using 5 strategies,
runs quality filter + independent difficulty judge + path dependency judge on ALL candidates.

Usage:
    python -m scripts.run_hard_rescue_pilot
    python -m scripts.run_hard_rescue_pilot --k_candidates 2 --limit_paths 3  # smoke test
    python -m scripts.run_hard_rescue_pilot --k_candidates 5                  # full run
"""
import argparse
import json
import time
import random
from collections import defaultdict
from pathlib import Path
from datetime import datetime

from dcqg.utils.jsonl import read_jsonl, write_jsonl
from dcqg.utils.config import get_api_config
from dcqg.generation.generator import generate_multi_strategy, STRATEGY_PROMPT_MAP, check_hard_path_suitability
from dcqg.question_filter.pipeline import quality_filter_pipeline
from dcqg.evaluation.judge import independent_difficulty_judge, independent_path_dependency_judge, blind_difficulty_judge, hard_answer_alignment_judge
from dcqg.question_filter.pipeline import apply_quality_filter_with_judges
from dcqg.tracing.record import TraceRecord
from dcqg.tracing.writer import write_full_trace
from dcqg.tracing.render import write_readable_trace, build_trace_from_pipeline_result


STRATEGIES = ["hidden_endpoint", "relation_composition"]


def _get_template_family_name(hard_answer_type):
    """Map hard_answer_type to template family name for reporting."""
    _MAP = {
        "restriction_policy": "restriction",
        "agreement_resolution": "agreement",
        "investigation_outcome": "investigation",
        "casualty_damage": "casualty",
        "movement_action_outcome": "action",
        "unsuitable": "unsuitable",
    }
    return _MAP.get(hard_answer_type, "generic")


# ── Counters ──
api_call_count = 0
generation_count = 0
filter_count = 0
judge_count = 0


# ═══════════════════════════════════════════════════════════════
# Step 1: Load and Select Hard Paths
# ═══════════════════════════════════════════════════════════════

def load_hard_paths(strict_path, relaxed_path, limit=None):
    """Load strict + relaxed Hard paths, deduplicate, priority sort."""
    strict_items = [x for x in read_jsonl(strict_path) if x.get("difficulty") == "Hard"]
    relaxed_items = [x for x in read_jsonl(relaxed_path) if x.get("difficulty") == "Hard"]

    # Deduplicate by dedup_key, keep strict when both exist
    strict_keys = set()
    for item in strict_items:
        key = item.get("dedup_key", f"{item.get('doc_id', '')}::{item.get('answer_event_id', '')}")
        strict_keys.add(key)

    merged = list(strict_items)
    for item in relaxed_items:
        key = item.get("dedup_key", f"{item.get('doc_id', '')}::{item.get('answer_event_id', '')}")
        if key not in strict_keys:
            merged.append(item)
            strict_keys.add(key)

    # Priority sort: non_temporal_count >= 1 first, then larger support_span, then CAUSE/SUBEVENT
    def priority(item):
        nt = item.get("non_temporal_count", 0)
        sp = item.get("support_span", 0)
        rg = item.get("relation_group", "NONE")
        rg_score = 1 if rg in ("CAUSE", "SUBEVENT", "MIXED") else 0
        return (-(1 if nt >= 1 else 0), -sp, -rg_score)

    merged.sort(key=priority)

    if limit:
        merged = merged[:limit]

    return merged


# ═══════════════════════════════════════════════════════════════
# Step 2: Generate K Candidates per Path x Strategy
# ═══════════════════════════════════════════════════════════════

def generate_candidates(paths, k_candidates, strategies, seed=42, model_config=None):
    """Generate K candidates per path per strategy."""
    global generation_count
    rng = random.Random(seed)
    all_results = []
    total_drift_fails = 0
    total_drift_repaired = 0
    total_unsupported_type = 0
    total_too_direct_cue = 0

    total_tasks = len(paths) * len(strategies) * k_candidates
    done = 0

    for path_idx, path_item in enumerate(paths):
        doc_id = path_item.get("doc_id", "")[:12]

        # Check path suitability before generating
        suitable, suit_reason = check_hard_path_suitability(path_item)
        if not suitable:
            print(f"  [SKIP] doc={doc_id} path unsuitable: {suit_reason}")
            if "unsupported_answer_type" in suit_reason:
                total_unsupported_type += 1
            # Mark all candidates for this path as unsuitable
            for strategy in strategies:
                for k in range(k_candidates):
                    result = {
                        "generation_error": True,
                        "grammar_pass": False,
                        "grammar_reason": f"path_unsuitable: {suit_reason}",
                        "hard_strategy": strategy,
                        "generated_question": "",
                        "events": path_item.get("events", []),
                        "supporting_sentences": path_item.get("supporting_sentences", []),
                        "hard_answer_type": "unsuitable",
                        "template_family": "unsuitable",
                    }
                    result["_path_idx"] = path_idx
                    result["_candidate_idx"] = k
                    result["dedup_key"] = path_item.get("dedup_key", "")
                    result["relation_group"] = path_item.get("relation_group", "")
                    result["support_span"] = path_item.get("support_span", 0)
                    result["non_temporal_count"] = path_item.get("non_temporal_count", 0)
                    all_results.append(result)
            continue

        for strategy in strategies:
            for k in range(k_candidates):
                done += 1
                print(f"  [{done}/{total_tasks}] doc={doc_id} strategy={strategy} k={k+1}", end=" ")

                result, attempts = generate_multi_strategy(path_item, strategy, max_attempts=5, model_config=model_config)
                generation_count += attempts

                # Track drift stats
                drift_fails = result.get("drift_check_fail", 0)
                drift_reps = result.get("drift_repaired", 0)
                total_drift_fails += drift_fails
                total_drift_repaired += drift_reps
                total_too_direct_cue += result.get("too_direct_cue", 0)

                # Attach path metadata
                result["_path_idx"] = path_idx
                result["_candidate_idx"] = k
                result["dedup_key"] = path_item.get("dedup_key", "")
                result["relation_group"] = path_item.get("relation_group", "")
                result["support_span"] = path_item.get("support_span", 0)
                result["non_temporal_count"] = path_item.get("non_temporal_count", 0)

                status = "OK" if result["grammar_pass"] else f"FAIL:{result['grammar_reason']}"
                print(f"-> {status} ({attempts} attempts)")

                all_results.append(result)
                time.sleep(0.1)

    print(f"  Drift check: {total_drift_fails} failures, {total_drift_repaired} repaired")
    print(f"  Too direct cue: {total_too_direct_cue} rejected")
    print(f"  Unsupported answer types: {total_unsupported_type} paths rejected")
    return all_results, total_drift_fails, total_drift_repaired, total_unsupported_type, total_too_direct_cue


# ═══════════════════════════════════════════════════════════════
# Step 3: Run Quality Filter on ALL Candidates
# ═══════════════════════════════════════════════════════════════

def run_filters(results, skip_llm=False):
    """Run quality_filter_pipeline on ALL candidates."""
    global filter_count
    for i, r in enumerate(results):
        if (i + 1) % 20 == 0:
            print(f"  Filtering [{i+1}/{len(results)}]...")

        # Skip filter for generation errors
        if r.get("generation_error"):
            r["final_filter_pass"] = False
            r["final_filter_reason"] = "generation_error"
            continue

        try:
            quality_filter_pipeline(r, skip_llm=skip_llm)
            filter_count += 1
        except Exception as exc:
            r["final_filter_pass"] = False
            r["final_filter_reason"] = f"filter_exception: {exc}"

    return results


# ═══════════════════════════════════════════════════════════════
# Step 4: Run Independent Judges on ALL Candidates
# ═══════════════════════════════════════════════════════════════

def run_judges(results, model_config):
    """Run blind difficulty + path dependency + alignment judges on ALL candidates."""
    global judge_count
    for i, r in enumerate(results):
        if (i + 1) % 20 == 0:
            print(f"  Judging [{i+1}/{len(results)}]...")

        # Skip judge for generation errors
        if r.get("generation_error"):
            r["difficulty_judge_status"] = "skipped"
            r["path_dependency_judge_status"] = "skipped"
            r["blind_difficulty_judge_status"] = "skipped"
            r["hard_alignment_status"] = "skipped"
            continue

        q = r.get("generated_question", "")
        if not q:
            r["difficulty_judge_status"] = "skipped"
            r["path_dependency_judge_status"] = "skipped"
            r["blind_difficulty_judge_status"] = "skipped"
            r["hard_alignment_status"] = "skipped"
            continue

        # Blind difficulty judge (context + question + answer ONLY)
        blind_result = blind_difficulty_judge(r, model_config)
        r.update(blind_result)
        judge_count += 1

        # Path dependency judge
        path_result = independent_path_dependency_judge(r, model_config)
        r.update(path_result)
        judge_count += 1

        # Hard answer-alignment judge
        align_result = hard_answer_alignment_judge(r, model_config)
        r.update(align_result)
        judge_count += 1

        blind_pred = r.get("blind_difficulty_judge", {}).get("predicted_difficulty", "?")
        dep = r.get("path_dependency_judge", {}).get("path_dependency", "?")
        asks = r.get("hard_alignment", {}).get("asks_expected_answer", "?")
        b_status = r.get("blind_difficulty_judge_status", "?")
        p_status = r.get("path_dependency_judge_status", "?")
        a_status = r.get("hard_alignment_status", "?")

        if (i + 1) % 20 == 0:
            print(f"    -> blind={blind_pred}({b_status}) dep={dep}({p_status}) asks={asks}({a_status})")

        time.sleep(0.15)

    return results


def apply_quality_filter(results):
    """Apply quality-only filter to all candidates."""
    for r in results:
        if r.get("generation_error") or not r.get("generated_question"):
            r["quality_filter_pass"] = False
            r["quality_filter_reason"] = "generation_error_or_empty"
            continue
        apply_quality_filter_with_judges(r)
    return results


# ═══════════════════════════════════════════════════════════════
# Step 5: Build Traces
# ═══════════════════════════════════════════════════════════════

def build_traces(results, output_dir):
    """Build and write trace artifacts."""
    traces = []
    for i, r in enumerate(results):
        try:
            trace = build_trace_from_pipeline_result(r, item_id=i)
            traces.append(trace)
        except Exception:
            pass

    trace_dir = output_dir / "debug_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    write_full_trace(traces, trace_dir)
    write_readable_trace(traces, trace_dir)
    print(f"  Wrote {len(traces)} traces to {trace_dir}")


# ═══════════════════════════════════════════════════════════════
# Step 6: Generate Report
# ═══════════════════════════════════════════════════════════════

def generate_report(results, selected_paths, output_dir, k_candidates, drift_fails=0, drift_repaired=0, unsupported_type=0, too_direct_cue=0):
    """Generate HARD_RESCUE_QUALITY_REPORT.md with quality-only filtering."""
    lines = []
    lines.append("# Hard Rescue Pilot Report (Quality Filter Edition)\n")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Paths:** {len(selected_paths)}")
    lines.append(f"**Strategies:** {', '.join(STRATEGIES)}")
    lines.append(f"**K candidates per path per strategy:** {k_candidates}")
    lines.append(f"**Total candidates generated:** {len(results)}")
    lines.append(f"**API calls:** generation={generation_count}, filter={filter_count}, judge={judge_count}")
    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # Section 1: Pool Stats
    # ══════════════════════════════════════════════════════════════
    lines.append("## 1. Pool Statistics\n")
    gen_errors = sum(1 for r in results if r.get("generation_error"))
    grammar_ok = sum(1 for r in results if r.get("grammar_pass"))
    quality_pass = sum(1 for r in results if r.get("quality_filter_pass"))

    lines.append("| Stage | Count |")
    lines.append("|-------|------:|")
    lines.append(f"| Selected Hard paths | {len(selected_paths)} |")
    lines.append(f"| Total candidates | {len(results)} |")
    lines.append(f"| Generation errors | {gen_errors} |")
    lines.append(f"| Grammar pass | {grammar_ok} |")
    lines.append(f"| Drift check failures | {drift_fails} |")
    lines.append(f"| Drift repaired | {drift_repaired} |")
    if too_direct_cue > 0:
        lines.append(f"| Too direct answer-type cue | {too_direct_cue} |")
    if unsupported_type > 0:
        lines.append(f"| Unsupported answer type (skipped) | {unsupported_type} |")
    lines.append(f"| **Quality filter pass** | **{quality_pass}** |")
    lines.append("")

    # Quality filter fail reason distribution
    qfail_reasons = defaultdict(int)
    for r in results:
        if not r.get("quality_filter_pass") and not r.get("generation_error"):
            reason = r.get("quality_filter_reason", "unknown")
            for part in reason.split(";"):
                qfail_reasons[part.strip()] += 1
    if qfail_reasons:
        lines.append("### Quality Filter Fail Reasons\n")
        lines.append("| Reason | Count |")
        lines.append("|--------|------:|")
        for reason, count in sorted(qfail_reasons.items(), key=lambda x: -x[1]):
            lines.append(f"| {reason} | {count} |")
        lines.append("")

    # ══════════════════════════════════════════════════════════════
    # Section 2: Quality-Pass Difficulty Distribution
    # ══════════════════════════════════════════════════════════════
    qp_blind = [r for r in results
                if r.get("quality_filter_pass")
                and r.get("blind_difficulty_judge_status") == "ok"]
    lines.append("## 2. Quality-Pass Difficulty Distribution\n")
    lines.append("*(Computed over quality_filter_pass candidates only)*\n")

    if qp_blind:
        qp_total = len(qp_blind)
        qp_hard = sum(1 for r in qp_blind if r["blind_difficulty_judge"]["predicted_difficulty"] == "Hard")
        qp_med = sum(1 for r in qp_blind if r["blind_difficulty_judge"]["predicted_difficulty"] == "Medium")
        qp_easy = sum(1 for r in qp_blind if r["blind_difficulty_judge"]["predicted_difficulty"] == "Easy")

        lines.append("| Metric | Count | Rate |")
        lines.append("|--------|------:|-----:|")
        lines.append(f"| Quality-pass (judged) | {qp_total} | — |")
        lines.append(f"| Blind Pred Easy | {qp_easy} | {qp_easy/qp_total*100:.1f}% |")
        lines.append(f"| Blind Pred Medium | {qp_med} | {qp_med/qp_total*100:.1f}% |")
        lines.append(f"| Blind Pred Hard | {qp_hard} | {qp_hard/qp_total*100:.1f}% |")
        lines.append("")

        # Required steps distribution
        lines.append("### Required Steps (Blind Judge)\n")
        steps_counts = defaultdict(int)
        for r in qp_blind:
            rs = r.get("blind_difficulty_judge", {}).get("required_steps", "?")
            steps_counts[rs] += 1
        lines.append("| Steps | Count | Rate |")
        lines.append("|------:|------:|-----:|")
        for steps in sorted(steps_counts.keys()):
            c = steps_counts[steps]
            lines.append(f"| {steps} | {c} | {c/qp_total*100:.1f}% |")
        lines.append("")

        # Single sentence answerable
        lines.append("### Single Sentence Answerable (Blind Judge)\n")
        ssa_counts = defaultdict(int)
        for r in qp_blind:
            ssa = r.get("blind_difficulty_judge", {}).get("single_sentence_answerable", "?")
            ssa_counts[ssa] += 1
        lines.append("| SSA | Count | Rate |")
        lines.append("|-----|------:|-----:|")
        for ssa in ["no", "partial", "yes"]:
            if ssa in ssa_counts:
                c = ssa_counts[ssa]
                lines.append(f"| {ssa} | {c} | {c/qp_total*100:.1f}% |")
        lines.append("")
    else:
        lines.append("*No quality-pass candidates with blind judge results.*\n")

    # ══════════════════════════════════════════════════════════════
    # Section 3: Quality-Pass Structural Metrics
    # ══════════════════════════════════════════════════════════════
    lines.append("## 3. Quality-Pass Structural Metrics\n")
    if qp_blind:
        # Path dependency
        pd_counts = defaultdict(int)
        for r in qp_blind:
            pd = r.get("path_dependency_judge", {}).get("path_dependency", "?")
            pd_counts[pd] += 1
        lines.append("### Path Dependency\n")
        lines.append("| Level | Count | Rate |")
        lines.append("|-------|------:|-----:|")
        for level in ["strong", "partial", "none"]:
            if level in pd_counts:
                c = pd_counts[level]
                lines.append(f"| {level} | {c} | {c/qp_total*100:.1f}% |")
        lines.append("")

        # Shortcut without path
        swp_counts = defaultdict(int)
        for r in qp_blind:
            swp = r.get("shortcut_without_path", "?")
            swp_counts[swp] += 1
        lines.append("### Shortcut Without Path\n")
        lines.append("| Value | Count | Rate |")
        lines.append("|-------|------:|-----:|")
        for val in ["no", "partial", "yes"]:
            if val in swp_counts:
                c = swp_counts[val]
                lines.append(f"| {val} | {c} | {c/qp_total*100:.1f}% |")
        lines.append("")
    else:
        lines.append("*No quality-pass candidates with structural metrics.*\n")

    # ══════════════════════════════════════════════════════════════
    # Section 4: Quality Metrics
    # ══════════════════════════════════════════════════════════════
    lines.append("## 4. Quality Metrics (Among Quality-Pass)\n")
    if qp_blind:
        def _qp_metric(getter, label):
            ok = sum(1 for r in qp_blind if getter(r))
            return f"| {label} | {ok}/{qp_total} ({ok/qp_total*100:.1f}%) |"

        lines.append("| Metric | Value |")
        lines.append("|--------|------:|")
        lines.append(_qp_metric(lambda r: r.get("blind_difficulty_judge", {}).get("answerable") in ("yes", "partial"), "Answerable (yes/partial)"))
        lines.append(_qp_metric(lambda r: r.get("blind_difficulty_judge", {}).get("final_event_consistent") in ("yes", "partial"), "Final-Event Consistent (yes/partial)"))
        lines.append(_qp_metric(lambda r: r.get("hard_alignment", {}).get("asks_expected_answer") in ("yes", "partial"), "Alignment asks (yes/partial)"))
        lines.append(_qp_metric(lambda r: r.get("hard_alignment", {}).get("target_drift") != "yes", "Target drift != yes"))
        lines.append(_qp_metric(lambda r: r.get("answer_consistency_label", "no") != "no", "Answer consistency != no"))
        lines.append("")
    else:
        lines.append("*No quality-pass candidates.*\n")

    # ══════════════════════════════════════════════════════════════
    # Section 5: Per Answer Type
    # ══════════════════════════════════════════════════════════════
    lines.append("## 5. Per Answer Type (Quality-Pass Only)\n")
    qp_valid = [r for r in results if r.get("quality_filter_pass") and not r.get("generation_error")]
    if qp_valid:
        lines.append("| Hard Answer Type | N QP | Blind Easy | Blind Med | Blind Hard | FEC yes/partial | SSA=no | PathDep strong |")
        lines.append("|-----------------|-----:|----------:|---------:|----------:|----------------|-------:|---------------:|")
        type_stats = defaultdict(lambda: {"n": 0, "easy": 0, "med": 0, "hard": 0, "fec": 0, "ssa_no": 0, "pd_strong": 0})
        for r in qp_valid:
            hat = r.get("hard_answer_type", "unknown")
            type_stats[hat]["n"] += 1
            blind = r.get("blind_difficulty_judge", {})
            pred = blind.get("predicted_difficulty", "?") if isinstance(blind, dict) else "?"
            if pred == "Easy":
                type_stats[hat]["easy"] += 1
            elif pred == "Medium":
                type_stats[hat]["med"] += 1
            elif pred == "Hard":
                type_stats[hat]["hard"] += 1
            fec = blind.get("final_event_consistent", "") if isinstance(blind, dict) else ""
            if fec in ("yes", "partial"):
                type_stats[hat]["fec"] += 1
            ssa = blind.get("single_sentence_answerable", "") if isinstance(blind, dict) else ""
            if ssa == "no":
                type_stats[hat]["ssa_no"] += 1
            pd = r.get("path_dependency_judge", {}).get("path_dependency", "")
            if pd == "strong":
                type_stats[hat]["pd_strong"] += 1
        for hat in sorted(type_stats.keys()):
            s = type_stats[hat]
            fec_pct = f"{s['fec']/s['n']*100:.0f}%" if s["n"] > 0 else "—"
            lines.append(f"| {hat} | {s['n']} | {s['easy']} | {s['med']} | {s['hard']} | {s['fec']} ({fec_pct}) | {s['ssa_no']} | {s['pd_strong']} |")
        lines.append("")
    else:
        lines.append("*No quality-pass candidates.*\n")

    # ══════════════════════════════════════════════════════════════
    # Section 6: Per Strategy
    # ══════════════════════════════════════════════════════════════
    lines.append("## 6. Per Strategy\n")
    lines.append("| Strategy | N judged | QP Rate | Blind Hard | Blind Med | Blind Easy | FEC% | SSA=no% | PathDep strong% |")
    lines.append("|----------|--------:|--------:|----------:|---------:|----------:|-----:|--------:|----------------:|")
    for strategy in STRATEGIES:
        strat_all = [r for r in results if r.get("hard_strategy") == strategy]
        strat_qp = [r for r in qp_blind if r.get("hard_strategy") == strategy]
        strat_judged = [r for r in results if r.get("hard_strategy") == strategy and r.get("blind_difficulty_judge_status") == "ok"]
        if not strat_all:
            lines.append(f"| {strategy} | 0 | — | — | — | — | — | — | — |")
            continue
        n_all = len(strat_all)
        n_judged = len(strat_judged)
        n_qp = len(strat_qp)
        qp_rate = f"{n_qp/n_all*100:.0f}%" if n_all > 0 else "—"
        if n_judged > 0:
            hard_c = sum(1 for r in strat_judged if r["blind_difficulty_judge"]["predicted_difficulty"] == "Hard")
            med_c = sum(1 for r in strat_judged if r["blind_difficulty_judge"]["predicted_difficulty"] == "Medium")
            easy_c = sum(1 for r in strat_judged if r["blind_difficulty_judge"]["predicted_difficulty"] == "Easy")
            fec_c = sum(1 for r in strat_judged if r["blind_difficulty_judge"].get("final_event_consistent") in ("yes", "partial"))
            ssa_c = sum(1 for r in strat_judged if r["blind_difficulty_judge"].get("single_sentence_answerable") == "no")
            pd_c = sum(1 for r in strat_judged if r.get("path_dependency_judge", {}).get("path_dependency") == "strong")
            lines.append(
                f"| {strategy} | {n_judged} | {qp_rate} | "
                f"{hard_c} ({hard_c/n_judged*100:.0f}%) | "
                f"{med_c} ({med_c/n_judged*100:.0f}%) | "
                f"{easy_c} ({easy_c/n_judged*100:.0f}%) | "
                f"{fec_c/n_judged*100:.0f}% | {ssa_c/n_judged*100:.0f}% | {pd_c/n_judged*100:.0f}% |"
            )
        else:
            lines.append(f"| {strategy} | {n_all} | {qp_rate} | — | — | — | — | — | — |")
    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # Section 7: Samples
    # ══════════════════════════════════════════════════════════════
    lines.append("## 7. Quality-Pass Samples by Predicted Difficulty\n")
    for tier in ["Easy", "Medium", "Hard"]:
        tier_samples = [r for r in qp_blind
                        if r["blind_difficulty_judge"]["predicted_difficulty"] == tier]
        tier_samples.sort(key=lambda r: len(r.get("generated_question", "")))
        lines.append(f"### {tier} Samples (top 3)\n")
        if tier_samples:
            for i, r in enumerate(tier_samples[:3]):
                _append_sample(lines, i, r, tier)
        else:
            lines.append(f"*No quality-pass {tier.lower()} samples.*\n")

    # ══════════════════════════════════════════════════════════════
    # Section 8: Quality-Pass Easy Diagnostic
    # ══════════════════════════════════════════════════════════════
    lines.append("## 8. Quality-Pass Easy Diagnostic\n")
    lines.append("*Why are quality-pass candidates judged Easy by the blind judge?*\n")
    qp_easy = [r for r in qp_blind
               if r["blind_difficulty_judge"]["predicted_difficulty"] == "Easy"]
    if qp_easy:
        lines.append("| # | Strategy | Answer Type | Ans Sent ID | SSA | Blind Reason (truncated) | PathDep | Question (truncated) |")
        lines.append("|--:|----------|------------|-------------|-----|--------------------------|---------|---------------------|")
        for i, r in enumerate(qp_easy[:10]):
            blind = r.get("blind_difficulty_judge", {})
            pd = r.get("path_dependency_judge", {})
            # Find answer sentence id
            answer_event_id = r.get("answer_event_id", "")
            ans_sent_id = "?"
            for e in r.get("events", []):
                if e.get("id") == answer_event_id:
                    ans_sent_id = e.get("sent_id", "?")
                    break
            ssa = blind.get("single_sentence_answerable", "?")
            reason = blind.get("reason", "")[:80]
            dep = pd.get("path_dependency", "?")
            q = r.get("generated_question", "")[:50]
            hat = r.get("hard_answer_type", "?")
            lines.append(
                f"| {i+1} | {r.get('hard_strategy', '?')} | {hat} | "
                f"S{ans_sent_id} | {ssa} | {reason} | {dep} | {q} |"
            )
        lines.append("")
    else:
        lines.append("*No quality-pass Easy samples.*\n")

    # ══════════════════════════════════════════════════════════════
    # Section 9: Path-Level Diagnostic
    # ══════════════════════════════════════════════════════════════
    lines.append("## 9. Path-Level Diagnostic (Selected Hard Paths)\n")
    lines.append("*For each selected hard path, show path-level risk factors.*\n")

    # Compute path-level stats from the raw path data
    path_risk_stats = defaultdict(lambda: {
        "single_sentence_risk": "unknown",
        "answer_in_final_sent": "?",
        "doc_id": "",
        "n_candidates": 0,
        "n_qp": 0,
        "best_blind_pred": "Easy",
    })

    for r in results:
        key = r.get("dedup_key", r.get("doc_id", ""))
        path_risk_stats[key]["n_candidates"] += 1
        if r.get("quality_filter_pass"):
            path_risk_stats[key]["n_qp"] += 1
        path_risk_stats[key]["doc_id"] = r.get("doc_id", "")[:12]
        blind = r.get("blind_difficulty_judge", {})
        pred = blind.get("predicted_difficulty", "Easy") if isinstance(blind, dict) else "Easy"
        rank = {"Hard": 2, "Medium": 1, "Easy": 0}.get(pred, 0)
        best_rank = {"Hard": 2, "Medium": 1, "Easy": 0}.get(
            path_risk_stats[key]["best_blind_pred"], 0)
        if rank > best_rank:
            path_risk_stats[key]["best_blind_pred"] = pred

    # single_sentence_risk from path data if available
    for p in selected_paths:
        key = p.get("dedup_key", f"{p.get('doc_id', '')}::{p.get('answer_event_id', '')}")
        ssr = p.get("single_sentence_risk", p.get("llm_path_judge", {}).get("single_sentence_risk", "unknown"))
        path_risk_stats[key]["single_sentence_risk"] = ssr

    # SSR distribution
    ssr_counts = defaultdict(int)
    for stats in path_risk_stats.values():
        ssr_counts[stats["single_sentence_risk"]] += 1

    lines.append("### Single Sentence Risk Distribution\n")
    lines.append("| Risk Level | Count | Rate |")
    lines.append("|------------|------:|-----:|")
    n_paths = len(path_risk_stats)
    for level in ["high", "medium", "low", "unknown"]:
        if level in ssr_counts:
            c = ssr_counts[level]
            lines.append(f"| {level} | {c} | {c/n_paths*100:.1f}% |")
    lines.append("")

    # Path detail table
    lines.append("### Path Detail\n")
    lines.append("| Doc ID | Candidates | QP | Best Blind Pred | SSR |")
    lines.append("|--------|----------:|---:|----------------:|-----|")
    for key, stats in sorted(path_risk_stats.items(), key=lambda x: x[1]["best_blind_pred"]):
        lines.append(
            f"| {stats['doc_id']} | {stats['n_candidates']} | {stats['n_qp']} | "
            f"{stats['best_blind_pred']} | {stats['single_sentence_risk']} |"
        )
    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # Section 10: Oracle Top-1 Diagnostic (NOT used for main metrics)
    # ══════════════════════════════════════════════════════════════
    lines.append("## 10. Oracle Top-1 Diagnostic\n")
    lines.append("> **NOT USED FOR MAIN METRICS. Diagnostic only.**\n")

    path_groups = defaultdict(list)
    blind_judged_all = [r for r in results if r.get("blind_difficulty_judge_status") == "ok"]
    for r in blind_judged_all:
        key = r.get("dedup_key", r.get("doc_id", ""))
        path_groups[key].append(r)

    oracle_top = []
    for key, candidates in path_groups.items():
        valid = [c for c in candidates if not c.get("generation_error") and c.get("generated_question")]
        if not valid:
            continue
        def _oracle_sort(r):
            blind = r.get("blind_difficulty_judge", {})
            pd = r.get("path_dependency_judge", {})
            return (
                0 if blind.get("predicted_difficulty") == "Hard" else (1 if blind.get("predicted_difficulty") == "Medium" else 2),
                0 if blind.get("single_sentence_answerable") == "no" else 1,
                0 if pd.get("path_dependency") == "strong" else 1,
                0 if r.get("quality_filter_pass") else 1,
                len(r.get("generated_question", "")),
            )
        valid.sort(key=_oracle_sort)
        oracle_top.append(valid[0])

    if oracle_top:
        o_total = len(oracle_top)
        o_hard = sum(1 for r in oracle_top if r["blind_difficulty_judge"]["predicted_difficulty"] == "Hard")
        o_ssa_no = sum(1 for r in oracle_top if r["blind_difficulty_judge"].get("single_sentence_answerable") == "no")
        o_pd_strong = sum(1 for r in oracle_top if r.get("path_dependency_judge", {}).get("path_dependency") == "strong")
        o_qp = sum(1 for r in oracle_top if r.get("quality_filter_pass"))

        lines.append("| Metric | Count | Rate |")
        lines.append("|--------|------:|-----:|")
        lines.append(f"| Total paths | {o_total} | — |")
        lines.append(f"| Oracle Blind Hard | {o_hard} | {o_hard/o_total*100:.1f}% |")
        lines.append(f"| Oracle SSA=no | {o_ssa_no} | {o_ssa_no/o_total*100:.1f}% |")
        lines.append(f"| Oracle PathDep strong | {o_pd_strong} | {o_pd_strong/o_total*100:.1f}% |")
        lines.append(f"| Oracle quality-filter-pass | {o_qp} | {o_qp/o_total*100:.1f}% |")
        lines.append("")

        lines.append("| # | Strategy | Blind Pred | SSA | PathDep | QP | Question (truncated) |")
        lines.append("|--:|----------|-----------:|-----|---------|----|---------------------|")
        for i, r in enumerate(oracle_top[:5]):
            blind = r.get("blind_difficulty_judge", {})
            pd = r.get("path_dependency_judge", {})
            q = r.get("generated_question", "")[:50]
            lines.append(
                f"| {i+1} | {r.get('hard_strategy', '?')} | "
                f"{blind.get('predicted_difficulty', '?')} | "
                f"{blind.get('single_sentence_answerable', '?')} | "
                f"{pd.get('path_dependency', '?')} | "
                f"{'Y' if r.get('quality_filter_pass') else 'N'} | "
                f"{q} |"
            )
        lines.append("")

    # ══════════════════════════════════════════════════════════════
    # Success / Readiness Criteria
    # ══════════════════════════════════════════════════════════════
    lines.append("## Success / Readiness Criteria\n")

    non_error = [r for r in results if not r.get("generation_error")]
    qp_count = sum(1 for r in results if r.get("quality_filter_pass"))
    qp_rate = qp_count / len(non_error) * 100 if non_error else 0

    if qp_blind:
        qp_fec = sum(1 for r in qp_blind if r["blind_difficulty_judge"].get("final_event_consistent") in ("yes", "partial"))
        qp_ans = sum(1 for r in qp_blind if r.get("hard_alignment", {}).get("asks_expected_answer") in ("yes", "partial"))
        fec_rate = qp_fec / qp_total * 100
        ans_rate = qp_ans / qp_total * 100
    else:
        fec_rate = 0
        ans_rate = 0

    lines.append(f"- Quality filter pass rate: {qp_count}/{len(non_error)} ({qp_rate:.1f}%)")
    if fec_rate >= 80:
        lines.append(f"- [PASS] FEC among quality-pass >= 80% ({fec_rate:.1f}%)")
    else:
        lines.append(f"- [FAIL] FEC among quality-pass = {fec_rate:.1f}% (need >= 80%)")
    if ans_rate >= 80:
        lines.append(f"- [PASS] Alignment (asks_expected_answer) among quality-pass >= 80% ({ans_rate:.1f}%)")
    else:
        lines.append(f"- [FAIL] Alignment among quality-pass = {ans_rate:.1f}% (need >= 80%)")

    if qp_blind:
        lines.append(f"- [INFO] Difficulty distribution (quality-pass): Easy={qp_easy}, Medium={qp_med}, Hard={qp_hard}")
    lines.append("- [NOTE] No difficulty metric used to select main evaluation set")
    lines.append("")

    report_path = Path(output_dir) / "HARD_RESCUE_QUALITY_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report: {report_path}")


def _select_top1_per_path(path_groups):
    """Select top-1 candidate per path by quality-first ranking.

    Sort order:
    1. quality_filter_pass=True first
    2. answerable + FEC yes/partial first
    3. blind Pred Hard first (informational tiebreaker, NOT a gate)
    4. shorter question first
    """
    top1 = []
    for key, candidates in path_groups.items():
        valid = [c for c in candidates if not c.get("generation_error") and c.get("generated_question")]
        if not valid:
            continue

        def _sort_key(r):
            blind = r.get("blind_difficulty_judge", {})
            return (
                0 if r.get("quality_filter_pass") else 1,
                0 if blind.get("answerable") in ("yes", "partial")
                    and blind.get("final_event_consistent") in ("yes", "partial") else 1,
                0 if blind.get("predicted_difficulty") == "Hard"
                    else (1 if blind.get("predicted_difficulty") == "Medium" else 2),
                len(r.get("generated_question", "")),
            )

        valid.sort(key=_sort_key)
        top1.append(valid[0])

    return top1


def _append_sample(lines, idx, r, label):
    """Append a sample entry to the report."""
    question = r.get("generated_question", "")[:200]
    answer = (r.get("gold_answer_phrase", "") or r.get("gold_answer_trigger", ""))[:200]
    events = r.get("events", [])
    event_path = " -> ".join(f"{e.get('trigger', '?')}" for e in events)
    strategy = r.get("hard_strategy", "?")
    blind = r.get("blind_difficulty_judge", {})
    pj = r.get("path_dependency_judge", {})
    pred = blind.get("predicted_difficulty", "?")
    dep = pj.get("path_dependency", "?")
    ans = blind.get("answerable", "?")
    fec = blind.get("final_event_consistent", "?")
    ssa = blind.get("single_sentence_answerable", "?")
    shortcut = r.get("shortcut_without_path", "?")
    quality_pass = r.get("quality_filter_pass", False)
    quality_reason = r.get("quality_filter_reason", "")
    blind_reason = blind.get("reason", "")

    lines.append(f"### {label} #{idx+1} [{strategy}]\n")
    lines.append(f"- **Question:** {question}")
    lines.append(f"- **Answer:** {answer}")
    lines.append(f"- **Event path:** {event_path}")
    lines.append(f"- **Blind Pred:** {pred} | **PathDep:** {dep} | **Answerable:** {ans} | **FEC:** {fec} | **SSA:** {ssa} | **Shortcut:** {shortcut}")
    lines.append(f"- **Quality Filter Pass:** {quality_pass}")
    if quality_reason:
        lines.append(f"- **Quality Reason:** {quality_reason}")
    if blind_reason:
        lines.append(f"- **Blind Judge Reason:** {blind_reason}")
    lines.append("")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    global api_call_count, generation_count, filter_count, judge_count

    parser = argparse.ArgumentParser(description="Hard Rescue Pilot")
    parser.add_argument("--strict_paths",
                        default="outputs/runs/path_filter_strict_pilot/paths.filtered.strict.jsonl")
    parser.add_argument("--relaxed_paths",
                        default="outputs/runs/path_filter_strict_pilot/paths.filtered.relaxed.jsonl")
    parser.add_argument("--pool_path", default=None,
                        help="Direct pool JSONL (bypasses strict+relaxed loading)")
    parser.add_argument("--output_dir", default=None,
                        help="Output dir. Default: outputs/runs/hard_rescue_pilot_<timestamp>")
    parser.add_argument("--k_candidates", type=int, default=3,
                        help="Number of candidates per path per strategy")
    parser.add_argument("--limit_paths", type=int, default=None,
                        help="Limit number of paths (for testing)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_llm_filters", action="store_true",
                        help="Skip LLM-based filter stages")
    args = parser.parse_args()

    # Output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"outputs/runs/hard_rescue_pilot_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model config for independent judges
    cfg = get_api_config()
    model_config = {
        "api_url": cfg["AIHUBMIX_API_URL"],
        "api_key": cfg["AIHUBMIX_API_KEY"],
        "model": cfg["AIHUBMIX_MODEL"],
    }
    print(f"Judge model: {cfg['AIHUBMIX_MODEL']}")

    # Step 1: Load Hard paths
    print("\n=== Step 1: Loading Hard paths ===")
    if args.pool_path:
        paths = read_jsonl(args.pool_path)
        if args.limit_paths:
            paths = paths[:args.limit_paths]
        print(f"  Loaded {len(paths)} paths from pool: {args.pool_path}")
    else:
        paths = load_hard_paths(args.strict_paths, args.relaxed_paths, limit=args.limit_paths)
    print(f"  Selected {len(paths)} Hard paths")

    # Write selected paths
    write_jsonl(output_dir / "selected_hard_paths.jsonl", paths)

    # Step 2: Generate candidates
    print(f"\n=== Step 2: Generating {args.k_candidates} candidates x {len(STRATEGIES)} strategies ===")
    total_tasks = len(paths) * len(STRATEGIES) * args.k_candidates
    print(f"  Total generation tasks: {total_tasks}")
    results, drift_fails, drift_repaired, unsupported_type, too_direct_cue = generate_candidates(paths, args.k_candidates, STRATEGIES, seed=args.seed, model_config=model_config)
    write_jsonl(output_dir / "questions.raw.jsonl", results)
    print(f"  Generated {len(results)} candidates")

    # Step 3: Quality filter
    print(f"\n=== Step 3: Running quality filter on {len(results)} candidates ===")
    results = run_filters(results, skip_llm=args.skip_llm_filters)
    write_jsonl(output_dir / "questions.filtered.jsonl", results)
    fp = sum(1 for r in results if r.get("final_filter_pass"))
    print(f"  Filter pass: {fp}/{len(results)}")

    # Step 4: Independent judges (blind difficulty + path dependency)
    print(f"\n=== Step 4: Running blind difficulty + path dependency judges on {len(results)} candidates ===")
    results = run_judges(results, model_config)
    write_jsonl(output_dir / "questions.judged.jsonl", results)

    # Step 4b: Apply quality filter (post-judges)
    print(f"\n=== Step 4b: Applying quality filter ===")
    results = apply_quality_filter(results)
    write_jsonl(output_dir / "questions.quality.jsonl", results)
    quality_fp = sum(1 for r in results if r.get("quality_filter_pass"))
    print(f"  Quality filter pass: {quality_fp}/{len(results)}")

    # Step 5: Traces
    print(f"\n=== Step 5: Building traces ===")
    build_traces(results, output_dir)

    # Step 6: Report
    print(f"\n=== Step 6: Generating report ===")
    generate_report(results, paths, output_dir, args.k_candidates, drift_fails, drift_repaired, unsupported_type, too_direct_cue)

    # Summary
    qp_blind = [r for r in results if r.get("quality_filter_pass") and r.get("blind_difficulty_judge_status") == "ok"]
    qp_hard = sum(1 for r in qp_blind if r.get("blind_difficulty_judge", {}).get("predicted_difficulty") == "Hard")
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Paths: {len(paths)}")
    print(f"  Candidates: {len(results)}")
    print(f"  Quality filter pass: {quality_fp}")
    print(f"  Quality-pass with blind judge: {len(qp_blind)}")
    print(f"  Quality-pass Blind Pred Hard: {qp_hard}")
    print(f"  API calls: gen={generation_count} filter={filter_count} judge={judge_count}")
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
