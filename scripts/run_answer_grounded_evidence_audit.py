"""Answer-Grounded Evidence Planning Audit.

Runs the answer-grounded evidence planner on story-matched candidates
and produces ANSWER_GROUNDED_EVIDENCE_AUDIT.md.

The original FairytaleQA question is NOT used in any evidence planning prompt.
"""

import json
import os
import sys
import time
import argparse
from collections import Counter, defaultdict

from dcqg.path.answer_grounded_evidence import plan_evidence


# ── Audit runner ──────────────────────────────────────────────────

def run_audit(candidates_path, output_dir, model=None, limit=None):
    """Run evidence planning audit on all candidates."""
    os.makedirs(output_dir, exist_ok=True)

    # Load candidates
    with open(candidates_path, encoding="utf-8") as f:
        candidates = [json.loads(l) for l in f if l.strip()]

    if limit:
        candidates = candidates[:limit]

    print(f"Candidates: {len(candidates)}")
    print(f"Model: {model or 'default JUDGE_MODEL'}")

    results_path = os.path.join(output_dir, "answer_grounded_evidence_plans.jsonl")
    out_f = open(results_path, "w", encoding="utf-8")

    for i, c in enumerate(candidates):
        story_name = c.get("story_name", "?")
        story_section = c.get("story_section", "")
        target_answer = c.get("answer", "") or c.get("answer1", "")
        target_difficulty = c.get("target_difficulty", c.get("evidence_difficulty", "Medium"))

        plan = plan_evidence(
            story_name=story_name,
            story_section=story_section,
            target_answer=target_answer,
            target_difficulty=target_difficulty,
            model=model,
            local_or_sum=c.get("local_or_sum", ""),
            attribute=c.get("attribute", ""),
            ex_or_im=c.get("ex_or_im", ""),
        )

        # Merge candidate metadata
        out = {
            "story_name": story_name,
            "target_difficulty": target_difficulty,
            "target_answer": target_answer,
            "original_question": c.get("question", ""),
            "local_or_sum": c.get("local_or_sum", ""),
            "attribute": c.get("attribute", ""),
            "ex_or_im": c.get("ex_or_im", ""),
            "story_group_id": c.get("story_group_id"),
            # Old evidence fields for comparison
            "old_required_evidence_sentences": c.get("required_evidence_sentences", []),
            "old_bridge_sentence_ids": c.get("bridge_sentence_ids", []),
            "old_answer_sentence_alone_sufficient": c.get("answer_sentence_alone_sufficient", ""),
            "old_necessity_type": c.get("necessity_type", ""),
            "old_reasoning_operation": c.get("reasoning_operation", ""),
        }
        out.update(plan)

        out_f.write(json.dumps(out, ensure_ascii=False) + "\n")

        if (i + 1) % 30 == 0:
            out_f.flush()
            print(f"  [{i + 1}/{len(candidates)}] processed")

        time.sleep(0.5)

    out_f.close()
    print(f"\nResults: {results_path}")
    return results_path


# ── Report building ───────────────────────────────────────────────

def compute_jaccard(a, b):
    """Jaccard similarity between two lists."""
    sa = set(a) if a else set()
    sb = set(b) if b else set()
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def build_report(results_path, output_dir):
    """Build ANSWER_GROUNDED_EVIDENCE_AUDIT.md."""
    with open(results_path, encoding="utf-8") as f:
        plans = [json.loads(l) for l in f if l.strip()]

    report_path = os.path.join(output_dir, "ANSWER_GROUNDED_EVIDENCE_AUDIT.md")
    total = len(plans)
    levels = ["Easy", "Medium", "Hard"]

    # Group by difficulty
    by_diff = {d: [p for p in plans if p.get("target_difficulty") == d] for d in levels}

    # ── 1. Run summary
    parse_ok = sum(1 for p in plans if p.get("answer_grounded_evidence_parse_ok"))
    plan_valid = sum(1 for p in plans if p.get("evidence_plan_valid") == "yes")
    model_used = plans[0].get("answer_grounded_evidence_model", "?") if plans else "?"

    # ── 2. Leakage audit
    leakage_count = sum(1 for p in plans if p.get("original_question_present_in_prompt"))

    # ── 3. Parse/validity stats
    parse_by_diff = {d: sum(1 for p in by_diff[d] if p.get("answer_grounded_evidence_parse_ok"))
                     for d in levels}
    valid_by_diff = {d: sum(1 for p in by_diff[d] if p.get("evidence_plan_valid") == "yes")
                     for d in levels}

    # ── 4. Feasibility by difficulty
    feasible_by_diff = {}
    for d in levels:
        feas = Counter()
        for p in by_diff[d]:
            feas[p.get("target_difficulty_feasible", "?")] += 1
        feasible_by_diff[d] = feas

    # ── 5. Required evidence count distribution
    num_req_dist = {d: Counter() for d in levels}
    for d in levels:
        for p in by_diff[d]:
            num_req_dist[d][p.get("num_required_sentences", 0)] += 1

    # ── 6. Easy diagnostics
    easy = by_diff["Easy"]
    easy_len1 = sum(1 for p in easy if p.get("num_required_sentences") == 1)
    easy_asa_yes = sum(1 for p in easy if p.get("answer_sentence_alone_sufficient") == "yes")
    easy_infeasible = [p for p in easy
                       if p.get("target_difficulty_feasible") in ("no", "partial")][:10]

    # ── 7. Medium diagnostics
    medium = by_diff["Medium"]
    med_len2 = sum(1 for p in medium if p.get("num_required_sentences") == 2)
    med_one_rel = sum(1 for p in medium
                      if p.get("necessity_type") in ("answer_local", "one_relation"))

    # ── 8. Hard diagnostics
    hard = by_diff["Hard"]
    hard_len3 = sum(1 for p in hard if p.get("num_required_sentences", 0) >= 3)
    hard_bridge = sum(1 for p in hard if p.get("bridge_required") == "yes")
    hard_causal = sum(1 for p in hard
                      if p.get("reasoning_operation") in
                      ("causal_chain", "motivation_chain", "summary_synthesis"))
    hard_feasible = sum(1 for p in hard
                        if p.get("target_difficulty_feasible") in ("yes", "partial"))
    hard_infeasible = [p for p in hard
                       if p.get("target_difficulty_feasible") == "no"][:10]

    # ── 9. Comparison to old evidence
    ans_id_match = 0
    jaccard_scores = []
    bridge_overlap_scores = []
    for p in plans:
        old_req = p.get("old_required_evidence_sentences", [])
        new_req = p.get("required_evidence_sentences", [])
        if old_req and new_req:
            jaccard_scores.append(compute_jaccard(old_req, new_req))
        old_bridge = p.get("old_bridge_sentence_ids", [])
        new_bridge = p.get("bridge_sentence_ids", [])
        if old_bridge or new_bridge:
            bridge_overlap_scores.append(compute_jaccard(old_bridge, new_bridge))
        # Answer sentence match: old required contains answer_sentence_id (heuristic)
        if old_req and p.get("answer_sentence_id") is not None:
            if p["answer_sentence_id"] in old_req:
                ans_id_match += 1

    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0
    avg_bridge_overlap = sum(bridge_overlap_scores) / len(bridge_overlap_scores) if bridge_overlap_scores else 0

    # ── 10. Examples per difficulty
    examples = {}
    for d in levels:
        pool = by_diff[d]
        # Prefer valid plans, diverse stories
        valid_pool = [p for p in pool if p.get("evidence_plan_valid") == "yes"]
        if not valid_pool:
            valid_pool = pool
        seen_stories = set()
        ex = []
        for p in valid_pool:
            sn = p.get("story_name", "")
            if sn not in seen_stories:
                seen_stories.add(sn)
                ex.append(p)
            if len(ex) >= 5:
                break
        examples[d] = ex

    # Contradiction stats
    total_contradictions = sum(len(p.get("contradictions", [])) for p in plans)
    plans_with_contradictions = sum(1 for p in plans if p.get("contradictions", []))

    # ── Write report ──────────────────────────────────────────────

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Answer-Grounded Evidence Planning Audit\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Section 1: Run summary
        f.write("## 1. Run Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|---|---|\n")
        f.write(f"| Total candidates | {total} |\n")
        f.write(f"| Model | {model_used} |\n")
        f.write(f"| Parse OK | {parse_ok}/{total} ({100*parse_ok/total:.1f}%) |\n")
        f.write(f"| Evidence plan valid | {plan_valid}/{total} ({100*plan_valid/total:.1f}%) |\n")
        f.write(f"| Plans with contradictions | {plans_with_contradictions}/{total} "
                f"({100*plans_with_contradictions/total:.1f}%) |\n")
        f.write(f"| Total contradiction count | {total_contradictions} |\n")
        f.write("\n")

        # Section 2: Prompt leakage audit
        f.write("## 2. Prompt Leakage Audit\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|---|---|\n")
        f.write(f"| original_question_present_in_prompt=True | {leakage_count} |\n")
        f.write(f"| Status | {'**PASS**' if leakage_count == 0 else '**FAIL**'} |\n")
        f.write("\n")

        # Section 3: Parse/validity stats
        f.write("## 3. Parse and Validity Stats\n\n")
        f.write("| Difficulty | Total | Parse OK | Pct | Plan Valid | Pct |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for d in levels:
            d_total = len(by_diff[d])
            f.write(f"| {d} | {d_total} | {parse_by_diff[d]} | "
                    f"{100*parse_by_diff[d]/d_total:.1f}% | "
                    f"{valid_by_diff[d]} | {100*valid_by_diff[d]/d_total:.1f}% |\n")
        f.write("\n")

        # Section 4: Feasibility by difficulty
        f.write("## 4. Evidence Plan Feasibility by Target Difficulty\n\n")
        f.write("| Difficulty | yes | partial | no | ? |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for d in levels:
            feas = feasible_by_diff[d]
            d_total = len(by_diff[d])
            f.write(f"| {d} | {feas.get('yes', 0)} | {feas.get('partial', 0)} | "
                    f"{feas.get('no', 0)} | {feas.get('?', 0)} |\n")
        f.write("\n")

        # Section 5: Required evidence count distribution
        f.write("## 5. Required Evidence Count Distribution\n\n")
        f.write("| Count | Easy | Medium | Hard |\n")
        f.write("|---|---:|---:|---:|\n")
        all_counts = sorted(set(
            c for d in levels for c in num_req_dist[d].keys()
        ))
        for cnt in all_counts:
            f.write(f"| {cnt} | {num_req_dist['Easy'].get(cnt, 0)} | "
                    f"{num_req_dist['Medium'].get(cnt, 0)} | "
                    f"{num_req_dist['Hard'].get(cnt, 0)} |\n")
        f.write("\n")

        # Section 6: Easy diagnostics
        f.write("## 6. Easy Diagnostics\n\n")
        n_easy = len(easy)
        f.write(f"| Metric | Count | Pct | Target |\n")
        f.write(f"|---|---:|---:|---:|\n")
        f.write(f"| num_req=1 | {easy_len1}/{n_easy} | {100*easy_len1/n_easy:.1f}% "
                f"| >=70% |\n")
        f.write(f"| ASA=yes | {easy_asa_yes}/{n_easy} | {100*easy_asa_yes/n_easy:.1f}% "
                f"| >=70% |\n")
        f.write("\n")

        if easy_infeasible:
            f.write("### Infeasible Easy Examples\n\n")
            f.write("| # | Story | Answer | Num Req | ASA | Feasible | Reason |\n")
            f.write("|---|---|---|---|---|---|\n")
            for i, p in enumerate(easy_infeasible, 1):
                f.write(f"| {i} | {p['story_name'][:25]} | {p['target_answer'][:30]} | "
                        f"{p.get('num_required_sentences', '?')} | "
                        f"{p.get('answer_sentence_alone_sufficient', '?')} | "
                        f"{p.get('target_difficulty_feasible', '?')} | "
                        f"{p.get('evidence_plan_reason', '')[:60]} |\n")
            f.write("\n")

        # Section 7: Medium diagnostics
        f.write("## 7. Medium Diagnostics\n\n")
        n_med = len(medium)
        f.write(f"| Metric | Count | Pct | Target |\n")
        f.write(f"|---|---:|---:|---:|\n")
        f.write(f"| num_req=2 | {med_len2}/{n_med} | {100*med_len2/n_med:.1f}% "
                f"| >=50% |\n")
        f.write(f"| necessity one_relation/answer_local | {med_one_rel}/{n_med} | "
                f"{100*med_one_rel/n_med:.1f}% | — |\n")
        f.write("\n")

        # Necessity type distribution for Medium
        med_necessity = Counter(p.get("necessity_type", "?") for p in medium)
        f.write("### Medium Necessity Types\n\n")
        f.write("| Type | Count |\n")
        f.write("|---|---:|\n")
        for nt, cnt in med_necessity.most_common():
            f.write(f"| {nt} | {cnt} |\n")
        f.write("\n")

        # Section 8: Hard diagnostics
        f.write("## 8. Hard Diagnostics\n\n")
        n_hard = len(hard)
        f.write(f"| Metric | Count | Pct | Target |\n")
        f.write(f"|---|---:|---:|---:|\n")
        f.write(f"| num_req>=3 | {hard_len3}/{n_hard} | {100*hard_len3/n_hard:.1f}% "
                f"| >=45% |\n")
        f.write(f"| bridge_required=yes | {hard_bridge}/{n_hard} | "
                f"{100*hard_bridge/n_hard:.1f}% | >=45% |\n")
        f.write(f"| causal/motivation/summary | {hard_causal}/{n_hard} | "
                f"{100*hard_causal/n_hard:.1f}% | — |\n")
        f.write(f"| feasible yes/partial | {hard_feasible}/{n_hard} | "
                f"{100*hard_feasible/n_hard:.1f}% | >=60% |\n")
        f.write("\n")

        # Reasoning operation distribution for Hard
        hard_reasoning = Counter(p.get("reasoning_operation", "?") for p in hard)
        f.write("### Hard Reasoning Operations\n\n")
        f.write("| Operation | Count |\n")
        f.write("|---|---:|\n")
        for op, cnt in hard_reasoning.most_common():
            f.write(f"| {op} | {cnt} |\n")
        f.write("\n")

        if hard_infeasible:
            f.write("### Infeasible Hard Examples\n\n")
            f.write("| # | Story | Answer | Num Req | ASA | Bridge | Reason |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for i, p in enumerate(hard_infeasible, 1):
                f.write(f"| {i} | {p['story_name'][:25]} | {p['target_answer'][:30]} | "
                        f"{p.get('num_required_sentences', '?')} | "
                        f"{p.get('answer_sentence_alone_sufficient', '?')} | "
                        f"{p.get('bridge_required', '?')} | "
                        f"{p.get('evidence_plan_reason', '')[:60]} |\n")
            f.write("\n")

        # Section 9: Comparison to old question-conditioned evidence
        f.write("## 9. Comparison to Old Question-Conditioned Evidence\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|---|---|\n")
        f.write(f"| Answer sentence ID match (in old required) | "
                f"{ans_id_match}/{total} ({100*ans_id_match/total:.1f}%) |\n")
        f.write(f"| Avg Jaccard (required evidence) | {avg_jaccard:.3f} |\n")
        f.write(f"| Avg Jaccard (bridge) | {avg_bridge_overlap:.3f} |\n")
        f.write("\n")

        # Jaccard distribution
        jaccard_bins = Counter()
        for s in jaccard_scores:
            if s == 0:
                jaccard_bins["0.0 (disjoint)"] += 1
            elif s < 0.3:
                jaccard_bins["0.01-0.29"] += 1
            elif s < 0.5:
                jaccard_bins["0.30-0.49"] += 1
            elif s < 0.7:
                jaccard_bins["0.50-0.69"] += 1
            elif s < 1.0:
                jaccard_bins["0.70-0.99"] += 1
            else:
                jaccard_bins["1.0 (identical)"] += 1
        if jaccard_bins:
            f.write("### Required Evidence Jaccard Distribution\n\n")
            f.write("| Range | Count |\n")
            f.write("|---|---:|\n")
            for bin_name in ["0.0 (disjoint)", "0.01-0.29", "0.30-0.49",
                             "0.50-0.69", "0.70-0.99", "1.0 (identical)"]:
                if bin_name in jaccard_bins:
                    f.write(f"| {bin_name} | {jaccard_bins[bin_name]} |\n")
            f.write("\n")

        # Section 10: Examples per difficulty
        f.write("## 10. Examples per Difficulty\n\n")
        for d in levels:
            f.write(f"### {d}\n\n")
            for i, p in enumerate(examples[d], 1):
                f.write(f"**Example {i}: {p['story_name']}**\n\n")
                f.write(f"- Target answer: \"{p['target_answer']}\"\n")
                f.write(f"- Original question (not in prompt): \"{p['original_question'][:100]}\"\n")
                f.write(f"- Answer sentence: S{p.get('answer_sentence_id', '?')}\n")
                f.write(f"- Required evidence: {p.get('required_evidence_sentences', [])}\n")
                f.write(f"- Anchor: {p.get('anchor_sentence_ids', [])}\n")
                f.write(f"- Bridge: {p.get('bridge_sentence_ids', [])}\n")
                f.write(f"- ASA: {p.get('answer_sentence_alone_sufficient', '?')}\n")
                f.write(f"- Bridge required: {p.get('bridge_required', '?')}\n")
                f.write(f"- Reasoning: {p.get('reasoning_operation', '?')}\n")
                f.write(f"- Necessity: {p.get('necessity_type', '?')}\n")
                f.write(f"- Feasible: {p.get('target_difficulty_feasible', '?')}\n")
                f.write(f"- Reason: {p.get('evidence_plan_reason', '')}\n\n")

        # ── Success Criteria Summary ──────────────────────────────
        f.write("## 11. Success Criteria Summary\n\n")
        f.write("| Criterion | Actual | Target | Status |\n")
        f.write("|---|---:|---:|---|\n")

        def sc(name, actual, target, higher_better=True, fmt="pct"):
            if higher_better:
                status = "**PASS**" if actual >= target else "**FAIL**"
            else:
                status = "**PASS**" if actual <= target else "**FAIL**"
            if fmt == "pct":
                f.write(f"| {name} | {actual:.1f}% | >={target:.0f}% | {status} |\n")
            elif fmt == "raw_count":
                f.write(f"| {name} | {actual} | {target} | {status} |\n")

        sc("original_question_present_in_prompt=0",
           f"{leakage_count}/{total}", "0/318", False, fmt="raw_count")
        sc("parse_ok >= 95%",
           100*parse_ok/total, 95)
        sc("evidence_plan_valid >= 85%",
           100*plan_valid/total, 85)
        sc("Easy len=1 >= 70%",
           100*easy_len1/n_easy, 70)
        sc("Easy ASA=yes >= 70%",
           100*easy_asa_yes/n_easy, 70)
        sc("Medium len=2 >= 50%",
           100*med_len2/n_med, 50)
        sc("Hard len>=3 >= 45%",
           100*hard_len3/n_hard, 45)
        sc("Hard bridge_required >= 45%",
           100*hard_bridge/n_hard, 45)
        sc("Hard feasible yes/partial >= 60%",
           100*hard_feasible/n_hard, 60)

        f.write("\n")

    print(f"Report: {report_path}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Answer-Grounded Evidence Planning Audit")
    parser.add_argument("--candidates", required=True,
                        help="Path to selected_candidates.jsonl")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory")
    parser.add_argument("--model", default=None,
                        help="Model override (default: JUDGE_MODEL from config)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of candidates")
    parser.add_argument("--report_only", action="store_true",
                        help="Only build report from existing plans")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results_path = os.path.join(args.output_dir,
                                "answer_grounded_evidence_plans.jsonl")

    if not args.report_only:
        print("=== Answer-Grounded Evidence Planning Audit ===")
        print(f"Candidates: {args.candidates}")
        print(f"Output: {args.output_dir}")
        if args.model:
            print(f"Model override: {args.model}")

        results_path = run_audit(
            args.candidates, args.output_dir,
            model=args.model, limit=args.limit,
        )

    if os.path.exists(results_path):
        build_report(results_path, args.output_dir)
    else:
        print("No plans found, skipping report.")


if __name__ == "__main__":
    main()
