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
from dcqg.question_filter.pipeline import apply_strict_hard_filter, apply_relaxed_hard_filter
from dcqg.tracing.record import TraceRecord
from dcqg.tracing.writer import write_full_trace
from dcqg.tracing.render import write_readable_trace, build_trace_from_pipeline_result


STRATEGIES = ["hidden_endpoint", "relation_composition"]

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

    total_tasks = len(paths) * len(strategies) * k_candidates
    done = 0

    for path_idx, path_item in enumerate(paths):
        doc_id = path_item.get("doc_id", "")[:12]

        # Check path suitability before generating
        suitable, suit_reason = check_hard_path_suitability(path_item)
        if not suitable:
            print(f"  [SKIP] doc={doc_id} path unsuitable: {suit_reason}")
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
    return all_results, total_drift_fails, total_drift_repaired


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


def apply_new_hard_filter(results):
    """Apply both strict and relaxed Hard filters."""
    for r in results:
        if r.get("generation_error") or not r.get("generated_question"):
            r["strict_new_hard_filter_pass"] = False
            r["strict_new_hard_filter_reason"] = "generation_error_or_empty"
            r["relaxed_new_hard_filter_pass"] = False
            r["relaxed_new_hard_filter_reason"] = "generation_error_or_empty"
            r["new_hard_filter_pass"] = False
            r["new_hard_filter_reason"] = "generation_error_or_empty"
            continue
        apply_strict_hard_filter(r)
        apply_relaxed_hard_filter(r)
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

def generate_report(results, selected_paths, output_dir, k_candidates, drift_fails=0, drift_repaired=0):
    """Generate HARD_RESCUE_REPORT.md with 3-level reporting: all / new-filter-passing / top-1 per path."""
    lines = []
    lines.append("# Hard Rescue Pilot Report (Blind Judge Edition)\n")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Paths:** {len(selected_paths)} (strict + relaxed Hard)")
    lines.append(f"**Strategies:** {', '.join(STRATEGIES)}")
    lines.append(f"**K candidates per path per strategy:** {k_candidates}")
    lines.append(f"**Total candidates generated:** {len(results)}")
    lines.append(f"**API calls:** generation={generation_count}, filter={filter_count}, judge={judge_count}")
    lines.append("")

    # ── Pool Statistics ──
    lines.append("## 1. Pool Statistics\n")
    lines.append("| Stage | Count |")
    lines.append("|-------|------:|")
    lines.append(f"| Selected Hard paths | {len(selected_paths)} |")
    gen_errors = sum(1 for r in results if r.get("generation_error"))
    grammar_ok = sum(1 for r in results if r.get("grammar_pass"))
    strict_pass = sum(1 for r in results if r.get("strict_new_hard_filter_pass"))
    relaxed_pass = sum(1 for r in results if r.get("relaxed_new_hard_filter_pass"))
    lines.append(f"| Total candidates | {len(results)} |")
    lines.append(f"| Grammar pass | {grammar_ok} |")
    lines.append(f"| Generation errors | {gen_errors} |")
    lines.append(f"| Drift check failures | {drift_fails} |")
    lines.append(f"| Drift repaired | {drift_repaired} |")
    lines.append(f"| Strict Hard filter pass | {strict_pass} |")
    lines.append(f"| Relaxed Hard filter pass | {relaxed_pass} |")
    lines.append("")

    # ── Blind Judge: All Candidates ──
    lines.append("## 2. Blind Difficulty Judge — All Candidates\n")
    blind_judged = [r for r in results if r.get("blind_difficulty_judge_status") == "ok"]

    def _blind_rate(items):
        if not items:
            return 0, 0, 0, 0
        total = len(items)
        hard = sum(1 for r in items if r["blind_difficulty_judge"]["predicted_difficulty"] == "Hard")
        med = sum(1 for r in items if r["blind_difficulty_judge"]["predicted_difficulty"] == "Medium")
        easy = sum(1 for r in items if r["blind_difficulty_judge"]["predicted_difficulty"] == "Easy")
        return total, hard, med, easy

    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    total, hard, med, easy = _blind_rate(blind_judged)
    lines.append(f"| Judged candidates | {total} |")
    lines.append(f"| Blind Pred Hard | {hard} ({hard/total*100:.1f}%) |" if total else "| Blind Pred Hard | 0 |")
    lines.append(f"| Blind Pred Medium | {med} ({med/total*100:.1f}%) |" if total else "| Blind Pred Medium | 0 |")
    lines.append(f"| Blind Pred Easy | {easy} ({easy/total*100:.1f}%) |" if total else "| Blind Pred Easy | 0 |")
    lines.append("")

    # ── Blind Judge: New Filter-Passing ──
    lines.append("## 3. Blind Difficulty Judge — New Filter-Passing Candidates\n")
    new_pass = [r for r in blind_judged if r.get("new_hard_filter_pass")]
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    total_p, hard_p, med_p, easy_p = _blind_rate(new_pass)
    lines.append(f"| Filter-passing candidates | {total_p} |")
    lines.append(f"| Blind Pred Hard | {hard_p} ({hard_p/total_p*100:.1f}%) |" if total_p else "| Blind Pred Hard | 0 |")
    lines.append(f"| Blind Pred Medium | {med_p} ({med_p/total_p*100:.1f}%) |" if total_p else "| Blind Pred Medium | 0 |")
    lines.append(f"| Blind Pred Easy | {easy_p} ({easy_p/total_p*100:.1f}%) |" if total_p else "| Blind Pred Easy | 0 |")
    lines.append("")

    # ── Path-level yield (using blind judge) ──
    lines.append("## 4. Path-Level Blind Pred Hard Yield\n")
    path_groups = defaultdict(list)
    for r in blind_judged:
        key = r.get("dedup_key", r.get("doc_id", ""))
        path_groups[key].append(r)

    paths_with_hard = 0
    paths_with_hard_answerable = 0
    paths_with_hard_full = 0  # answerable + fec + pathdep strong
    paths_with_strict_filter = 0
    paths_with_relaxed_filter = 0

    for key, candidates in path_groups.items():
        hard_cands = [c for c in candidates if c["blind_difficulty_judge"]["predicted_difficulty"] == "Hard"]
        if hard_cands:
            paths_with_hard += 1
            ans_ok = [c for c in hard_cands if c["blind_difficulty_judge"].get("answerable") in ("yes", "partial")]
            if ans_ok:
                paths_with_hard_answerable += 1
                fec_ok = [c for c in ans_ok if c["blind_difficulty_judge"].get("final_event_consistent") in ("yes", "partial")]
                dep_ok = [c for c in fec_ok if c.get("path_dependency_judge", {}).get("path_dependency") == "strong"]
                if dep_ok:
                    paths_with_hard_full += 1
        if any(c.get("strict_new_hard_filter_pass") for c in candidates):
            paths_with_strict_filter += 1
        if any(c.get("relaxed_new_hard_filter_pass") for c in candidates):
            paths_with_relaxed_filter += 1

    n_paths = len(path_groups)
    lines.append("| Metric | Count | Rate |")
    lines.append("|--------|------:|-----:|")
    lines.append(f"| Total unique paths | {n_paths} | — |")
    lines.append(f"| Paths with >= 1 Blind Pred Hard | {paths_with_hard} | {paths_with_hard/n_paths*100:.1f}% |" if n_paths else "| Paths with >= 1 Blind Pred Hard | 0 | — |")
    lines.append(f"| Paths with Blind Hard + answerable | {paths_with_hard_answerable} | {paths_with_hard_answerable/n_paths*100:.1f}% |" if n_paths else "| Paths with Blind Hard + answerable | 0 | — |")
    lines.append(f"| Paths with Blind Hard + ans + fec + pathdep strong | {paths_with_hard_full} | {paths_with_hard_full/n_paths*100:.1f}% |" if n_paths else "| Paths with Blind Hard + full | 0 | — |")
    lines.append(f"| Paths with >= 1 strict filter pass | {paths_with_strict_filter} | {paths_with_strict_filter/n_paths*100:.1f}% |" if n_paths else "| Paths with >= 1 strict filter pass | 0 | — |")
    lines.append(f"| Paths with >= 1 relaxed filter pass | {paths_with_relaxed_filter} | {paths_with_relaxed_filter/n_paths*100:.1f}% |" if n_paths else "| Paths with >= 1 relaxed filter pass | 0 | — |")
    lines.append("")

    # ── Per-Strategy Comparison ──
    lines.append("## 5. Per-Strategy Comparison (Blind Judge)\n")
    lines.append("| Strategy | N judged | Blind Hard | Blind Med | Blind Easy | Ans% | FEC% | PathDep Strong% | Strict Pass% | Relaxed Pass% |")
    lines.append("|----------|--------:|----------:|---------:|----------:|-----:|-----:|----------------:|-------------:|--------------:|")

    for strategy in STRATEGIES:
        strat_items = [r for r in blind_judged if r.get("hard_strategy") == strategy]
        strat_all = [r for r in results if r.get("hard_strategy") == strategy]
        if not strat_items:
            lines.append(f"| {strategy} | 0 | — | — | — | — | — | — | — | — |")
            continue
        n = len(strat_items)
        hard_c = sum(1 for r in strat_items if r["blind_difficulty_judge"]["predicted_difficulty"] == "Hard")
        med_c = sum(1 for r in strat_items if r["blind_difficulty_judge"]["predicted_difficulty"] == "Medium")
        easy_c = sum(1 for r in strat_items if r["blind_difficulty_judge"]["predicted_difficulty"] == "Easy")
        ans_c = sum(1 for r in strat_items if r["blind_difficulty_judge"].get("answerable") in ("yes", "partial"))
        fec_c = sum(1 for r in strat_items if r["blind_difficulty_judge"].get("final_event_consistent") in ("yes", "partial"))
        dep_c = sum(1 for r in strat_items if r.get("path_dependency_judge", {}).get("path_dependency") == "strong")
        sp_c = sum(1 for r in strat_all if r.get("strict_new_hard_filter_pass"))
        rp_c = sum(1 for r in strat_all if r.get("relaxed_new_hard_filter_pass"))
        lines.append(
            f"| {strategy} | {n} | {hard_c} ({hard_c/n*100:.0f}%) | "
            f"{med_c} ({med_c/n*100:.0f}%) | {easy_c} ({easy_c/n*100:.0f}%) | "
            f"{ans_c/n*100:.0f}% | {fec_c/n*100:.0f}% | {dep_c/n*100:.0f}% | "
            f"{sp_c/len(strat_all)*100:.0f}% | {rp_c/len(strat_all)*100:.0f}% |" if strat_all else ""
        )
    lines.append("")

    # ── Strict Hard Filter Fail Reason Distribution ──
    lines.append("## 6. Strict Hard Filter Fail Reason Distribution\n")
    strict_fail_reasons = defaultdict(int)
    for r in results:
        if not r.get("strict_new_hard_filter_pass") and not r.get("generation_error"):
            reason = r.get("strict_new_hard_filter_reason", "unknown")
            for part in reason.split(";"):
                strict_fail_reasons[part.strip()] += 1
    if strict_fail_reasons:
        lines.append("| Reason | Count |")
        lines.append("|--------|------:|")
        for reason, count in sorted(strict_fail_reasons.items(), key=lambda x: -x[1]):
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("No filter failures.")
    lines.append("")

    # ── Quality Metrics (Blind Judge) ──
    strict_pass_items = [r for r in blind_judged if r.get("strict_new_hard_filter_pass")]
    relaxed_pass_items = [r for r in blind_judged if r.get("relaxed_new_hard_filter_pass")]

    lines.append("## 7. Quality Metrics Summary (Blind Judge)\n")
    lines.append("| Metric | All Judged | Strict Pass | Relaxed Pass |")
    lines.append("|--------|----------:|------------:|-------------:|")

    def _metric(items, getter):
        ok = [r for r in items if getter(r)]
        return f"{len(ok)}/{len(items)} ({len(ok)/len(items)*100:.0f}%)" if items else "—"

    lines.append(f"| Blind Answerable (yes/partial) | {_metric(blind_judged, lambda r: r.get('blind_difficulty_judge', {}).get('answerable') in ('yes', 'partial'))} | {_metric(strict_pass_items, lambda r: r.get('blind_difficulty_judge', {}).get('answerable') in ('yes', 'partial'))} | {_metric(relaxed_pass_items, lambda r: r.get('blind_difficulty_judge', {}).get('answerable') in ('yes', 'partial'))} |")
    lines.append(f"| Blind Final-Event Consistent | {_metric(blind_judged, lambda r: r.get('blind_difficulty_judge', {}).get('final_event_consistent') in ('yes', 'partial'))} | {_metric(strict_pass_items, lambda r: r.get('blind_difficulty_judge', {}).get('final_event_consistent') in ('yes', 'partial'))} | {_metric(relaxed_pass_items, lambda r: r.get('blind_difficulty_judge', {}).get('final_event_consistent') in ('yes', 'partial'))} |")
    lines.append(f"| PathDep Strong | {_metric(blind_judged, lambda r: r.get('path_dependency_judge', {}).get('path_dependency') == 'strong')} | {_metric(strict_pass_items, lambda r: r.get('path_dependency_judge', {}).get('path_dependency') == 'strong')} | {_metric(relaxed_pass_items, lambda r: r.get('path_dependency_judge', {}).get('path_dependency') == 'strong')} |")
    lines.append(f"| Single-Sent Answerable=no | {_metric(blind_judged, lambda r: r.get('blind_difficulty_judge', {}).get('single_sentence_answerable') == 'no')} | {_metric(strict_pass_items, lambda r: r.get('blind_difficulty_judge', {}).get('single_sentence_answerable') == 'no')} | {_metric(relaxed_pass_items, lambda r: r.get('blind_difficulty_judge', {}).get('single_sentence_answerable') == 'no')} |")
    lines.append(f"| Alignment asks=yes/partial | {_metric(blind_judged, lambda r: r.get('hard_alignment', {}).get('asks_expected_answer') in ('yes', 'partial'))} | {_metric(strict_pass_items, lambda r: r.get('hard_alignment', {}).get('asks_expected_answer') in ('yes', 'partial'))} | {_metric(relaxed_pass_items, lambda r: r.get('hard_alignment', {}).get('asks_expected_answer') in ('yes', 'partial'))} |")
    lines.append("")

    # ── Selected Top-1 Per Path ──
    lines.append("## 8. Selected Top-1 Per Path\n")
    top1_per_path = _select_top1_per_path(path_groups)
    lines.append(f"**{len(top1_per_path)} paths with selected top-1 candidates.**\n")

    # Top-1 aggregate stats
    if top1_per_path:
        t1_total = len(top1_per_path)
        t1_hard = sum(1 for r in top1_per_path if r["blind_difficulty_judge"]["predicted_difficulty"] == "Hard")
        t1_ans_fec = sum(1 for r in top1_per_path
                         if r["blind_difficulty_judge"].get("answerable") in ("yes", "partial")
                         and r["blind_difficulty_judge"].get("final_event_consistent") in ("yes", "partial"))
        t1_pathdep = sum(1 for r in top1_per_path if r.get("path_dependency_judge", {}).get("path_dependency") == "strong")
        t1_strict = sum(1 for r in top1_per_path if r.get("strict_new_hard_filter_pass"))
        t1_relaxed = sum(1 for r in top1_per_path if r.get("relaxed_new_hard_filter_pass"))

        lines.append("| Metric | Count | Rate |")
        lines.append("|--------|------:|-----:|")
        lines.append(f"| Total paths | {t1_total} | — |")
        lines.append(f"| Top-1 Blind Pred Hard | {t1_hard} | {t1_hard/t1_total*100:.1f}% |")
        lines.append(f"| Top-1 Answerable + FEC | {t1_ans_fec} | {t1_ans_fec/t1_total*100:.1f}% |")
        lines.append(f"| Top-1 PathDep Strong | {t1_pathdep} | {t1_pathdep/t1_total*100:.1f}% |")
        lines.append(f"| Top-1 Strict Filter Pass | {t1_strict} | {t1_strict/t1_total*100:.1f}% |")
        lines.append(f"| Top-1 Relaxed Filter Pass | {t1_relaxed} | {t1_relaxed/t1_total*100:.1f}% |")
        lines.append("")

        lines.append("### Top-1 Per Path Detail\n")
        lines.append("| # | Strategy | Blind Pred | Ans | FEC | PathDep | SSA | Strict | Relaxed | Question (truncated) |")
        lines.append("|--:|----------|-----------:|-----|-----|---------|-----|--------|---------|---------------------|")
        for i, r in enumerate(top1_per_path):
            blind = r.get("blind_difficulty_judge", {})
            pd = r.get("path_dependency_judge", {})
            q = r.get("generated_question", "")[:60]
            lines.append(
                f"| {i+1} | {r.get('hard_strategy', '?')} | "
                f"{blind.get('predicted_difficulty', '?')} | "
                f"{blind.get('answerable', '?')} | "
                f"{blind.get('final_event_consistent', '?')} | "
                f"{pd.get('path_dependency', '?')} | "
                f"{blind.get('single_sentence_answerable', '?')} | "
                f"{'Y' if r.get('strict_new_hard_filter_pass') else 'N'} | "
                f"{'Y' if r.get('relaxed_new_hard_filter_pass') else 'N'} | "
                f"{q} |"
            )
        lines.append("")

    # ── Blind Context Audit ──
    lines.append("## 9. Blind Context Audit\n")
    blind_with_answer = sum(1 for r in blind_judged if r.get("blind_context_contains_answer_sentence"))
    lines.append(f"- **blind_context_contains_answer_sentence=true:** {blind_with_answer}/{len(blind_judged)}")
    lines.append(f"- **blind_context_contains_answer_sentence=false:** {len(blind_judged) - blind_with_answer}/{len(blind_judged)}")
    lines.append("")

    # ── Blind Prompt Samples (for audit) ──
    lines.append("## 10. Blind Difficulty Judge Prompt Samples (for audit)\n")
    prompt_samples = [r for r in blind_judged if r.get("blind_difficulty_judge_prompt")]
    for i, r in enumerate(prompt_samples[:3]):
        lines.append(f"### Sample {i+1}\n")
        lines.append(f"**Question:** {r.get('generated_question', '')[:150]}")
        lines.append(f"**Blind Pred:** {r.get('blind_difficulty_judge', {}).get('predicted_difficulty', '?')}")
        lines.append(f"\n```\n{r['blind_difficulty_judge_prompt'][:1500]}\n```\n")

    # ── Best Samples ──
    lines.append("## 11. Best Samples (Blind Hard + Answerable + PathDep)\n")
    best = [r for r in blind_judged
            if r.get("blind_difficulty_judge", {}).get("predicted_difficulty") == "Hard"
            and r.get("blind_difficulty_judge", {}).get("answerable") in ("yes", "partial")
            and r.get("path_dependency_judge", {}).get("path_dependency") in ("strong",)]
    best.sort(key=lambda r: (
        0 if r.get("relaxed_new_hard_filter_pass") else 1,
        0 if r.get("strict_new_hard_filter_pass") else 1,
        0 if r.get("path_dependency_judge", {}).get("path_dependency") == "strong" else 1,
        len(r.get("generated_question", "")),
    ))
    for i, r in enumerate(best[:5]):
        _append_sample(lines, i, r, "Best")

    if not best:
        lines.append("*No Blind Hard + answerable + path-dependent candidates found.*\n")

    # ── Worst Samples ──
    lines.append("## 12. Worst Samples\n")
    worst = [r for r in results if not r.get("generation_error")]
    worst.sort(key=lambda r: (
        0 if not r.get("relaxed_new_hard_filter_pass") else 1,
        0 if r.get("blind_difficulty_judge", {}).get("predicted_difficulty") == "Easy" else 1,
    ))
    for i, r in enumerate(worst[:5]):
        _append_sample(lines, i, r, "Worst")

    # ── Success Criteria ──
    lines.append("## 13. Success Criteria Evaluation\n")

    # Top-1 metrics
    if top1_per_path:
        t1_pred_hard_rate = t1_hard / t1_total * 100
        t1_ans_fec_rate = t1_ans_fec / t1_total * 100
        t1_pathdep_rate = t1_pathdep / t1_total * 100
        t1_strict_rate = t1_strict / t1_total * 100
        t1_relaxed_rate = t1_relaxed / t1_total * 100
    else:
        t1_pred_hard_rate = 0
        t1_ans_fec_rate = 0
        t1_pathdep_rate = 0
        t1_strict_rate = 0
        t1_relaxed_rate = 0

    all_blind_hard_rate = hard / total * 100 if total else 0

    lines.append(f"- **Top-1 per path: Blind Pred Hard rate:** {t1_pred_hard_rate:.1f}%")
    lines.append(f"- **Top-1 per path: Answerable + FEC rate:** {t1_ans_fec_rate:.1f}%")
    lines.append(f"- **Top-1 per path: PathDep Strong rate:** {t1_pathdep_rate:.1f}%")
    lines.append(f"- **Top-1 per path: Strict Hard Filter Pass rate:** {t1_strict_rate:.1f}%")
    lines.append(f"- **Top-1 per path: Relaxed Hard Filter Pass rate:** {t1_relaxed_rate:.1f}%")
    lines.append(f"- **All candidates: Blind Pred Hard rate:** {all_blind_hard_rate:.1f}%")
    lines.append("")

    criteria_met = 0
    criteria_total = 6

    if t1_pred_hard_rate >= 20:
        lines.append("- [PASS] Top-1 Blind Pred Hard >= 20%")
        criteria_met += 1
    else:
        lines.append(f"- [FAIL] Top-1 Blind Pred Hard = {t1_pred_hard_rate:.1f}% (need >= 20%)")

    if t1_ans_fec_rate >= 80:
        lines.append("- [PASS] Top-1 Answerable + FEC >= 80%")
        criteria_met += 1
    else:
        lines.append(f"- [FAIL] Top-1 Answerable + FEC = {t1_ans_fec_rate:.1f}% (need >= 80%)")

    if t1_pathdep_rate >= 50:
        lines.append("- [PASS] Top-1 PathDep Strong >= 50%")
        criteria_met += 1
    else:
        lines.append(f"- [FAIL] Top-1 PathDep Strong = {t1_pathdep_rate:.1f}% (need >= 50%)")

    if t1_strict_rate >= 20:
        lines.append("- [PASS] Top-1 Strict Hard Filter Pass >= 20%")
        criteria_met += 1
    else:
        lines.append(f"- [FAIL] Top-1 Strict Hard Filter Pass = {t1_strict_rate:.1f}% (need >= 20%)")

    if t1_relaxed_rate >= 20:
        lines.append("- [PASS] Top-1 Relaxed Hard Filter Pass >= 20%")
        criteria_met += 1
    else:
        lines.append(f"- [FAIL] Top-1 Relaxed Hard Filter Pass = {t1_relaxed_rate:.1f}% (need >= 20%)")

    if all_blind_hard_rate > 0:
        lines.append("- [PASS] All candidates: Blind Pred Hard > 0")
        criteria_met += 1
    else:
        lines.append("- [FAIL] All candidates: Blind Pred Hard = 0")

    lines.append(f"\n**Criteria met: {criteria_met}/{criteria_total}**\n")

    if criteria_met == criteria_total:
        lines.append("**RESULT: All 6 criteria met. Strong Hard candidate evidence across strict and relaxed filters.**")
    elif t1_strict_rate >= 20 and t1_relaxed_rate >= 20:
        lines.append("**RESULT: Both strict and relaxed Hard filters show viable candidates.**")
    elif t1_relaxed_rate >= 20 and t1_strict_rate < 20:
        lines.append("**RESULT: Relaxed Hard filter viable; strict Hard filter below threshold.**")
    elif t1_relaxed_rate == 0 and all_blind_hard_rate > 0:
        lines.append("**RESULT: Blind Pred Hard candidates exist, but none pass relaxed Hard filter.**")
    elif t1_pred_hard_rate == 0:
        lines.append("**RESULT: Pred Hard = 0. No Hard-difficulty evidence.**")
    else:
        lines.append(f"**RESULT: Partial. {criteria_met}/{criteria_total} criteria met.**")

    report_path = Path(output_dir) / "HARD_RESCUE_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report: {report_path}")


def _select_top1_per_path(path_groups):
    """Select top-1 candidate per path by publishable quality ranking.

    Sort order:
    1. relaxed_new_hard_filter_pass=True first
    2. strict_new_hard_filter_pass=True first
    3. blind Pred Hard first
    4. pathdep strong first
    5. single_sentence_answerable=no first
    6. answerable + FEC (yes/partial) first
    7. shorter question first
    """
    top1 = []
    for key, candidates in path_groups.items():
        valid = [c for c in candidates if not c.get("generation_error") and c.get("generated_question")]
        if not valid:
            continue

        def _sort_key(r):
            blind = r.get("blind_difficulty_judge", {})
            pd = r.get("path_dependency_judge", {})
            return (
                0 if r.get("relaxed_new_hard_filter_pass") else 1,
                0 if r.get("strict_new_hard_filter_pass") else 1,
                0 if blind.get("predicted_difficulty") == "Hard" else (1 if blind.get("predicted_difficulty") == "Medium" else 2),
                0 if pd.get("path_dependency") == "strong" else 1,
                0 if blind.get("single_sentence_answerable") == "no" else 1,
                0 if blind.get("answerable") in ("yes", "partial") and blind.get("final_event_consistent") in ("yes", "partial") else 1,
                len(r.get("generated_question", "")),
            )

        valid.sort(key=_sort_key)
        top1.append(valid[0])

    return top1


def _append_sample(lines, idx, r, label):
    """Append a sample entry to the report using blind judge fields."""
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
    strict_pass = r.get("strict_new_hard_filter_pass", False)
    relaxed_pass = r.get("relaxed_new_hard_filter_pass", False)
    reason = blind.get("reason", "")
    filter_reason = r.get("strict_new_hard_filter_reason", "") or r.get("relaxed_new_hard_filter_reason", "")

    lines.append(f"### {label} #{idx+1} [{strategy}]\n")
    lines.append(f"- **Question:** {question}")
    lines.append(f"- **Answer:** {answer}")
    lines.append(f"- **Event path:** {event_path}")
    lines.append(f"- **Blind Pred:** {pred} | **PathDep:** {dep} | **Answerable:** {ans} | **FEC:** {fec} | **SSA:** {ssa}")
    lines.append(f"- **Strict Hard Filter Pass:** {strict_pass}")
    lines.append(f"- **Relaxed Hard Filter Pass:** {relaxed_pass}")
    if reason:
        lines.append(f"- **Blind Judge Reason:** {reason}")
    if filter_reason:
        lines.append(f"- **Filter Reason:** {filter_reason}")
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
    paths = load_hard_paths(args.strict_paths, args.relaxed_paths, limit=args.limit_paths)
    print(f"  Selected {len(paths)} Hard paths")

    # Write selected paths
    write_jsonl(output_dir / "selected_hard_paths.jsonl", paths)

    # Step 2: Generate candidates
    print(f"\n=== Step 2: Generating {args.k_candidates} candidates x {len(STRATEGIES)} strategies ===")
    total_tasks = len(paths) * len(STRATEGIES) * args.k_candidates
    print(f"  Total generation tasks: {total_tasks}")
    results, drift_fails, drift_repaired = generate_candidates(paths, args.k_candidates, STRATEGIES, seed=args.seed, model_config=model_config)
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

    # Step 4b: Apply new Hard filter
    print(f"\n=== Step 4b: Applying new Hard filter ===")
    results = apply_new_hard_filter(results)
    write_jsonl(output_dir / "questions.new_filter.jsonl", results)
    strict_fp = sum(1 for r in results if r.get("strict_new_hard_filter_pass"))
    relaxed_fp = sum(1 for r in results if r.get("relaxed_new_hard_filter_pass"))
    print(f"  Strict Hard filter pass: {strict_fp}/{len(results)}")
    print(f"  Relaxed Hard filter pass: {relaxed_fp}/{len(results)}")

    # Step 5: Traces
    print(f"\n=== Step 5: Building traces ===")
    build_traces(results, output_dir)

    # Step 6: Report
    print(f"\n=== Step 6: Generating report ===")
    generate_report(results, paths, output_dir, args.k_candidates, drift_fails, drift_repaired)

    # Summary
    blind_judged = [r for r in results if r.get("blind_difficulty_judge_status") == "ok"]
    blind_hard = sum(1 for r in blind_judged if r.get("blind_difficulty_judge", {}).get("predicted_difficulty") == "Hard")
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Paths: {len(paths)}")
    print(f"  Candidates: {len(results)}")
    print(f"  Strict Hard filter pass: {strict_fp}")
    print(f"  Relaxed Hard filter pass: {relaxed_fp}")
    print(f"  Blind judged OK: {len(blind_judged)}")
    print(f"  Blind Pred Hard: {blind_hard}")
    print(f"  API calls: gen={generation_count} filter={filter_count} judge={judge_count}")
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
