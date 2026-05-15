"""Stage C.1 Smoke Test: Answer-Grounded Evidence → Graph → QG (with repairs).

Runs the full answer-grounded pipeline with Fix 1-4 applied:
  Fix 1: Force answer-role node in graph extraction
  Fix 2: Easy prompt with local grammar context + answer-type templates
  Fix 3: Active Easy forbidden-frame validator in retry loop
  Fix 4: Hard summary/count exception for bridge_required

Diagnoses failure modes in detail.
"""

import json
import os
import sys
import time
import argparse
from collections import Counter, defaultdict

from dcqg.path.answer_grounded_evidence import plan_evidence
from dcqg.graph.narrative_graph import NarrativeGraphExtractor
from dcqg.generation.fairytale_qg import (
    generate_ours, generate_direct, generate_icl,
    detect_easy_forbidden_frames,
)


# ── Candidate loading ─────────────────────────────────────────────

def load_candidates(candidates_path, max_per_difficulty=10):
    """Load candidates, selecting up to N per difficulty level.

    Accepts either calibrated JSONL or selected_candidates JSONL.
    """
    with open(candidates_path, encoding="utf-8") as f:
        all_c = [json.loads(l) for l in f]

    # Detect format
    if all_c and "evidence_difficulty" in all_c[0]:
        diff_key = "evidence_difficulty"
    elif all_c and "target_difficulty" in all_c[0]:
        diff_key = "target_difficulty"
    else:
        diff_key = "evidence_difficulty"

    by_diff = defaultdict(list)
    for c in all_c:
        diff = c.get(diff_key, "Medium")
        if diff in ("Easy", "Medium", "Hard"):
            by_diff[diff].append(c)

    selected = []
    for diff in ["Easy", "Medium", "Hard"]:
        pool = by_diff[diff]
        # Prefer diverse stories
        seen_stories = set()
        sel = []
        for c in pool:
            sn = c.get("story_name", "?")
            if sn not in seen_stories:
                sel.append(c)
                seen_stories.add(sn)
                if len(sel) >= max_per_difficulty:
                    break
        selected.extend(sel)
        print(f"  {diff}: {len(sel)}/{len(pool)} candidates selected")

    print(f"Total selected: {len(selected)}")
    return selected


# ── Answer-Grounded Evidence + Graph + QG ─────────────────────────

def run_answer_grounded_pipeline(candidate, model=None):
    """Run the full answer-grounded pipeline on one candidate."""
    story_name = candidate.get("story_name", "?")
    story_section = candidate.get("story_section", "")
    target_answer = candidate.get("answer", "") or candidate.get("answer1", "")
    target_difficulty = candidate.get("evidence_difficulty") or candidate.get(
        "target_difficulty", "Medium")
    original_question = candidate.get("question", "")

    trace = {
        "story_name": story_name,
        "target_answer": target_answer,
        "target_difficulty": target_difficulty,
        "original_question": original_question,
    }

    # ── Step 1: Evidence Planning ──
    evidence_plan = plan_evidence(
        story_name=story_name, story_section=story_section,
        target_answer=target_answer, target_difficulty=target_difficulty,
        model=model,
    )
    trace["evidence_plan"] = {
        "parse_ok": evidence_plan.get("answer_grounded_evidence_parse_ok", False),
        "valid": evidence_plan.get("evidence_plan_valid", "?"),
        "required_evidence_sentences": evidence_plan.get("required_evidence_sentences", []),
        "bridge_sentence_ids": evidence_plan.get("bridge_sentence_ids", []),
        "anchor_sentence_ids": evidence_plan.get("anchor_sentence_ids", []),
        "answer_sentence_id": evidence_plan.get("answer_sentence_id"),
        "reasoning_operation": evidence_plan.get("reasoning_operation", ""),
        "necessity_type": evidence_plan.get("necessity_type", ""),
        "num_required_sentences": evidence_plan.get("num_required_sentences", 0),
        "answer_sentence_alone_sufficient": evidence_plan.get("answer_sentence_alone_sufficient", ""),
        "bridge_required": evidence_plan.get("bridge_required", ""),
        "target_difficulty_feasible": evidence_plan.get("target_difficulty_feasible", ""),
        "contradictions": evidence_plan.get("contradictions", []),
    }

    if not evidence_plan.get("answer_grounded_evidence_parse_ok"):
        trace["graph_result"] = {"graph_valid": False, "reason": "evidence plan parse failed"}
        trace["qg_result"] = {"generated_question": "", "error": "evidence plan parse failed"}
        return trace

    # ── Step 2: Build graph candidate ──
    graph_candidate = {
        "story_name": story_name,
        "story_section": story_section,
        "question": "",
        "answer": target_answer,
        "answer_sentence_id": evidence_plan.get("answer_sentence_id"),
        "required_evidence_sentences": evidence_plan.get("required_evidence_sentences", []),
        "bridge_sentence_ids": evidence_plan.get("bridge_sentence_ids", []),
        "reasoning_operation": evidence_plan.get("reasoning_operation", ""),
        "necessity_type": evidence_plan.get("necessity_type", ""),
    }

    # ── Step 3: Graph Extraction ──
    extractor = NarrativeGraphExtractor()
    try:
        graph_record = extractor.extract(graph_candidate, difficulty=target_difficulty)
    except Exception as e:
        graph_record = {
            "graph_valid": False,
            "graph_validation_reason": f"extraction_error: {e}",
            "nodes": [], "edges": [],
        }

    trace["graph_result"] = {
        "graph_valid": graph_record.get("graph_valid", False),
        "graph_validation_reason": graph_record.get("graph_validation_reason", ""),
        "num_nodes": len(graph_record.get("nodes", [])),
        "num_edges": len(graph_record.get("edges", [])),
        "graph_role_repair_applied": graph_record.get("graph_role_repair_applied", False),
        "graph_role_repair_reason": graph_record.get("graph_role_repair_reason", ""),
        "diagnostics": graph_record.get("diagnostics", {}),
    }

    if not graph_record.get("graph_valid"):
        trace["qg_result"] = {
            "method": "Ours_AnswerGrounded",
            "generated_question": "",
            "attempts": 0, "parse_ok": False,
            "error": f"graph_invalid: {graph_record.get('graph_validation_reason', '?')}",
        }
        return trace

    # ── Step 4: QG Generation ──
    qg_result, attempts = generate_ours(
        story_section=story_section,
        target_answer=target_answer,
        difficulty=target_difficulty,
        nodes=graph_record.get("nodes", []),
        edges=graph_record.get("edges", []),
        required_evidence_sentences=evidence_plan.get("required_evidence_sentences", []),
        bridge_sentence_ids=evidence_plan.get("bridge_sentence_ids", []),
        reasoning_operation=evidence_plan.get("reasoning_operation", ""),
        necessity_type=evidence_plan.get("necessity_type", ""),
    )

    gen_q = qg_result.get("generated_question", "")
    # Classify Easy failure modes
    easy_forbidden = False
    easy_forbidden_frames = []
    easy_malformed = False
    if target_difficulty == "Easy" and gen_q:
        easy_forbidden, easy_forbidden_frames = detect_easy_forbidden_frames(gen_q)
        # Check malformed: not a proper question
        q_clean = gen_q.strip()
        if not q_clean.endswith("?") or len(q_clean.split()) < 3:
            easy_malformed = True

    trace["qg_result"] = {
        "method": "Ours_AnswerGrounded",
        "generated_question": gen_q,
        "attempts": attempts,
        "parse_ok": qg_result.get("parse_ok", False),
        "generation_error": qg_result.get("generation_error", ""),
        "self_check_pass": qg_result.get("self_check_pass", False),
        "graph_policy": qg_result.get("graph_policy", ""),
        "graph_policy_reason": qg_result.get("graph_policy_reason", ""),
        "evidence_roles_used": qg_result.get("evidence_roles_used", []),
        "relation_chain": qg_result.get("relation_chain", []),
        "focus_match": qg_result.get("focus_match", "?"),
        "graph_policy_compliance": qg_result.get("graph_policy_compliance", "?"),
        "attempts_trace": qg_result.get("attempts_trace", []),
        # Easy-specific diagnostics
        "easy_forbidden_violation": easy_forbidden,
        "easy_forbidden_frames": easy_forbidden_frames,
        "easy_malformed": easy_malformed,
    }

    return trace


# ── Report ────────────────────────────────────────────────────────

def print_report(results, output_dir):
    """Print detailed diagnostic report."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "STAGE_C1_SMOKE_REPORT.md")

    n = len(results)
    by_diff = defaultdict(list)
    for r in results:
        by_diff[r["target_difficulty"]].append(r)

    def _pct(num, den):
        return f"{100*num/den:.1f}%" if den else "N/A"

    lines = []
    lines.append("# Stage C.1 Smoke Test: Answer-Grounded Pipeline with Repairs")
    lines.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\nTotal candidates: {n}")
    for diff in ["Easy", "Medium", "Hard"]:
        lines.append(f"  {diff}: {len(by_diff[diff])}")
    lines.append("")

    # ── 1. Pipeline Summary ──
    lines.append("## 1. Pipeline Summary\n")
    ep_ok = sum(1 for r in results if r["evidence_plan"].get("parse_ok"))
    ep_valid = sum(1 for r in results if r["evidence_plan"].get("valid") == "yes")
    graph_valid = sum(1 for r in results if r["graph_result"].get("graph_valid"))
    graph_repair = sum(1 for r in results if r["graph_result"].get("graph_role_repair_applied"))
    qg_gen = sum(1 for r in results if r["qg_result"].get("generated_question"))
    qg_parse = sum(1 for r in results if r["qg_result"].get("parse_ok"))
    qg_sc = sum(1 for r in results if r["qg_result"].get("self_check_pass"))
    orig_q = sum(1 for r in results if r["evidence_plan"].get("original_question_present_in_prompt"))

    lines.append("| Stage | Pass | Total | Pct |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Evidence parse | {ep_ok} | {n} | {_pct(ep_ok, n)} |")
    lines.append(f"| Evidence valid | {ep_valid} | {n} | {_pct(ep_valid, n)} |")
    lines.append(f"| Graph valid | {graph_valid} | {n} | {_pct(graph_valid, n)} |")
    lines.append(f"| Graph role repair | {graph_repair} | {n} | {_pct(graph_repair, n)} |")
    lines.append(f"| QG question generated | {qg_gen} | {n} | {_pct(qg_gen, n)} |")
    lines.append(f"| QG parse OK | {qg_parse} | {n} | {_pct(qg_parse, n)} |")
    lines.append(f"| QG self-check pass | {qg_sc} | {n} | {_pct(qg_sc, n)} |")
    lines.append(f"| Original Q leakage | {orig_q} | {n} | {_pct(orig_q, n)} |")
    lines.append(f"| **End-to-end pass** | **{qg_sc}** | **{n}** | **{_pct(qg_sc, n)}** |")
    lines.append("")

    # ── 2. By Difficulty ──
    lines.append("## 2. By Difficulty\n")
    lines.append("| Difficulty | N | Ev Parse | Ev Valid | Graph Valid | Graph Repair | QG Gen | QG Parse | QG Self-Check |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for diff in ["Easy", "Medium", "Hard"]:
        drs = by_diff[diff]
        nd = len(drs)
        if nd == 0:
            continue
        dep = sum(1 for r in drs if r["evidence_plan"].get("parse_ok"))
        dev = sum(1 for r in drs if r["evidence_plan"].get("valid") == "yes")
        dgv = sum(1 for r in drs if r["graph_result"].get("graph_valid"))
        dgr = sum(1 for r in drs if r["graph_result"].get("graph_role_repair_applied"))
        dqgen = sum(1 for r in drs if r["qg_result"].get("generated_question"))
        dqp = sum(1 for r in drs if r["qg_result"].get("parse_ok"))
        dqs = sum(1 for r in drs if r["qg_result"].get("self_check_pass"))
        lines.append(f"| {diff} | {nd} | {dep} | {dev} | {dgv} | {dgr} | {dqgen} | {dqp} | {dqs} |")
    lines.append("")

    # ── 3. Easy Diagnostics ──
    lines.append("## 3. Easy Diagnostics\n")
    easy_results = by_diff.get("Easy", [])
    n_easy = len(easy_results)
    if n_easy:
        easy_mal = sum(1 for r in easy_results if r["qg_result"].get("easy_malformed"))
        easy_ban = sum(1 for r in easy_results if r["qg_result"].get("easy_forbidden_violation"))
        easy_sc = sum(1 for r in easy_results if r["qg_result"].get("self_check_pass"))
        easy_parse = sum(1 for r in easy_results if r["qg_result"].get("parse_ok"))
        lines.append("| Metric | Count | Pct |")
        lines.append("|---|---:|---:|")
        lines.append(f"| Easy malformed | {easy_mal} | {_pct(easy_mal, n_easy)} |")
        lines.append(f"| Easy banned frames | {easy_ban} | {_pct(easy_ban, n_easy)} |")
        lines.append(f"| Easy parse OK | {easy_parse} | {_pct(easy_parse, n_easy)} |")
        lines.append(f"| Easy self-check pass | {easy_sc} | {_pct(easy_sc, n_easy)} |")
        # Banned frame breakdown
        frame_counts = Counter()
        for r in easy_results:
            for f in r["qg_result"].get("easy_forbidden_frames", []):
                frame_counts[f] += 1
        if frame_counts:
            lines.append("\n### Banned Frame Breakdown\n")
            lines.append("| Frame | Count |")
            lines.append("|---|---:|")
            for f, c in frame_counts.most_common():
                lines.append(f"| {f} | {c} |")
        lines.append("")

    # ── 4. Hard Diagnostics ──
    lines.append("## 4. Hard Diagnostics\n")
    hard_results = by_diff.get("Hard", [])
    n_hard = len(hard_results)
    if n_hard:
        hard_sc = sum(1 for r in hard_results if r["qg_result"].get("self_check_pass"))
        hard_bridge_no = sum(1 for r in hard_results
                             if r["evidence_plan"].get("bridge_required") != "yes"
                             and r["evidence_plan"].get("valid") == "yes")
        hard_ro_counts = Counter(
            r["evidence_plan"].get("reasoning_operation", "?") for r in hard_results)
        lines.append("| Metric | Count | Pct |")
        lines.append("|---|---:|---:|")
        lines.append(f"| Hard self-check pass | {hard_sc} | {_pct(hard_sc, n_hard)} |")
        lines.append(f"| Hard bridge_required!=yes | {hard_bridge_no} | {_pct(hard_bridge_no, n_hard)} |")
        lines.append("\n### Hard Reasoning Operations\n")
        lines.append("| Operation | Count |")
        lines.append("|---|---:|")
        for op, cnt in hard_ro_counts.most_common():
            lines.append(f"| {op} | {cnt} |")
        lines.append("")

    # ── 5. Graph Role Repair Details ──
    lines.append("## 5. Graph Role Repair Details\n")
    repaired = [r for r in results if r["graph_result"].get("graph_role_repair_applied")]
    if repaired:
        lines.append(f"Total repairs: {len(repaired)}\n")
        for r in repaired:
            lines.append(f"- **{r['story_name']}** [{r['target_difficulty']}]: "
                         f"{r['graph_result']['graph_role_repair_reason']}")
    else:
        lines.append("No graph role repairs needed.")
    lines.append("")

    # ── 6. Self-Check Details ──
    lines.append("## 6. QG Self-Check Details\n")
    for diff in ["Easy", "Medium", "Hard"]:
        lines.append(f"### {diff}\n")
        drs = by_diff[diff]
        sc_pass = [r for r in drs if r["qg_result"].get("self_check_pass")]
        sc_fail = [r for r in drs if r["qg_result"].get("generated_question")
                   and not r["qg_result"].get("self_check_pass")]
        lines.append(f"Pass: {len(sc_pass)}/{len(drs)}, Fail: {len(sc_fail)}/{len(drs)}\n")
        if sc_pass:
            lines.append("**Passing:**\n")
            for r in sc_pass:
                q = r["qg_result"]["generated_question"][:100]
                lines.append(f"- {r['story_name']}: \"{q}\"")
        if sc_fail:
            lines.append("\n**Failing:**\n")
            for r in sc_fail:
                q = r["qg_result"]["generated_question"][:100]
                err = r["qg_result"].get("generation_error", "?")
                traces = r["qg_result"].get("attempts_trace", [])
                last_trace = traces[-1] if traces else {}
                sc_reason = last_trace.get("self_check_reason", err)[:80]
                lines.append(f"- {r['story_name']}: \"{q}\" | reason: {sc_reason}")
        lines.append("")

    # ── 7. Per-Candidate Table ──
    lines.append("## 7. Per-Candidate Table\n")
    lines.append("| # | Story | Diff | Ev OK | Graph OK | Repair | QG | Parse | SC | Question |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(results, 1):
        ep = r["evidence_plan"]
        gr = r["graph_result"]
        qg = r["qg_result"]
        q_short = qg.get("generated_question", "")[:60]
        lines.append(
            f"| {i} | {r['story_name'][:20]} | {r['target_difficulty'][:6]} | "
            f"{'Y' if ep.get('parse_ok') else 'N'} | "
            f"{'Y' if gr.get('graph_valid') else 'N'} | "
            f"{'Y' if gr.get('graph_role_repair_applied') else '-'} | "
            f"{'Y' if qg.get('generated_question') else 'N'} | "
            f"{'Y' if qg.get('parse_ok') else 'N'} | "
            f"{'Y' if qg.get('self_check_pass') else 'N'} | "
            f"{q_short} |"
        )
    lines.append("")

    # ── 8. Success Criteria ──
    lines.append("## 8. Success Criteria\n")
    lines.append("| Criterion | Actual | Target | Status |")
    lines.append("|---|---:|---:|---|")
    targets = [
        ("Evidence valid >= 95%", _pct(ep_valid, n), ">=95%",
         ep_valid / n >= 0.95 if n else False),
        ("Graph valid >= 90%", _pct(graph_valid, n), ">=90%",
         graph_valid / n >= 0.90 if n else False),
        ("QG parse OK >= 85%", _pct(qg_parse, n), ">=85%",
         qg_parse / n >= 0.85 if n else False),
        ("QG self-check >= 50%", _pct(qg_sc, n), ">=50%",
         qg_sc / n >= 0.50 if n else False),
    ]
    # Per-difficulty
    for diff in ["Easy", "Medium", "Hard"]:
        drs = by_diff[diff]
        nd = len(drs)
        if nd == 0:
            continue
        d_sc = sum(1 for r in drs if r["qg_result"].get("self_check_pass"))
        targets.append(
            (f"{diff} self-check >= 40%" if diff == "Easy" else f"{diff} self-check >= 50%",
             _pct(d_sc, nd),
             ">=40%" if diff == "Easy" else ">=50%",
             d_sc / nd >= (0.40 if diff == "Easy" else 0.50))
        )
    targets.append(
        ("Original Q leakage = 0", str(orig_q), "0", orig_q == 0)
    )
    for name, actual, target, ok in targets:
        status = "**PASS**" if ok else "**FAIL**"
        lines.append(f"| {name} | {actual} | {target} | {status} |")

    # ── Write ──
    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)
    print(f"\nReport: {report_path}")

    # Save traces
    trace_path = os.path.join(output_dir, "stage_c1_smoke_traces.json")
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Traces: {trace_path}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage C.1: Answer-Grounded Evidence → Graph → QG smoke test")
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates JSONL")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory")
    parser.add_argument("--max_per_difficulty", type=int, default=10,
                        help="Max candidates per difficulty level")
    parser.add_argument("--model", default=None,
                        help="Model override for evidence planner")
    args = parser.parse_args()

    print("=" * 70)
    print("Stage C.1: Answer-Grounded Pipeline Smoke Test (with repairs)")
    print("=" * 70)

    print(f"\nLoading candidates from: {args.candidates}")
    candidates = load_candidates(args.candidates,
                                 max_per_difficulty=args.max_per_difficulty)

    if not candidates:
        print("ERROR: No candidates found!")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for i, c in enumerate(candidates):
        diff = c.get("evidence_difficulty") or c.get("target_difficulty", "?")
        sn = c.get("story_name", "?")
        print(f"\n[{i+1}/{len(candidates)}] {sn} [{diff}]")
        try:
            result = run_answer_grounded_pipeline(c, model=args.model)
        except Exception as e:
            result = {
                "story_name": sn, "target_difficulty": diff,
                "target_answer": c.get("answer", ""),
                "original_question": c.get("question", ""),
                "evidence_plan": {"parse_ok": False, "valid": "no",
                                  "contradictions": [f"exception: {e}"]},
                "graph_result": {"graph_valid": False,
                                 "graph_role_repair_applied": False},
                "qg_result": {"generated_question": "", "attempts": 0,
                              "parse_ok": False, "error": str(e)},
            }
        results.append(result)

        ep = result["evidence_plan"]
        gr = result["graph_result"]
        qg = result["qg_result"]
        repair_tag = " [REPAIR]" if gr.get("graph_role_repair_applied") else ""
        print(f"  Ev:{'OK' if ep.get('parse_ok') else 'FAIL'} "
              f"Gr:{'OK' if gr.get('graph_valid') else 'FAIL'}{repair_tag} "
              f"QG:{'OK' if qg.get('generated_question') else 'FAIL'} "
              f"SC:{'PASS' if qg.get('self_check_pass') else 'FAIL'} "
              f"({qg.get('attempts','?')}a)")

        time.sleep(0.5)

    print("\n" + "=" * 70)
    print_report(results, args.output_dir)


if __name__ == "__main__":
    main()
