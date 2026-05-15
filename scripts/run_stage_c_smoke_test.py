"""Stage C Smoke Test: Answer-Grounded Evidence → Graph → QG.

Runs the full answer-grounded pipeline on a small set of calibrated candidates:
  1. plan_evidence() — identify evidence sentences from story + answer + difficulty
  2. NarrativeGraphExtractor — build graph from answer-grounded evidence
  3. generate_ours() — generate question using answer-grounded graph

Compares answer-grounded results to old question-conditioned baselines.
"""

import json
import os
import sys
import time
import argparse
from collections import Counter, defaultdict

from dcqg.path.answer_grounded_evidence import plan_evidence
from dcqg.graph.narrative_graph import NarrativeGraphExtractor
from dcqg.generation.fairytale_qg import generate_ours, generate_direct, generate_icl


# ── Candidate loading ─────────────────────────────────────────────

def load_calibrated_candidates(calibrated_path, n_per_difficulty=3):
    """Load calibrated candidates, selecting N per difficulty level."""
    with open(calibrated_path, encoding="utf-8") as f:
        all_cal = [json.loads(l) for l in f]

    by_diff = defaultdict(list)
    for c in all_cal:
        if c.get("calibrated"):
            by_diff[c["evidence_difficulty"]].append(c)

    selected = []
    for diff in ["Easy", "Medium", "Hard"]:
        pool = by_diff[diff]
        # Prefer diverse stories
        seen_stories = set()
        sel = []
        for c in pool:
            sn = c["story_name"]
            if sn not in seen_stories:
                sel.append(c)
                seen_stories.add(sn)
                if len(sel) >= n_per_difficulty:
                    break
        selected.extend(sel)
        print(f"  {diff}: {len(sel)}/{len(pool)} calibrated candidates selected")

    print(f"Total selected: {len(selected)}")
    return selected


# ── Answer-Grounded Evidence + Graph + QG ─────────────────────────

def run_answer_grounded_pipeline(candidate, model=None):
    """Run the full answer-grounded pipeline on one candidate.

    Returns dict with full trace.
    """
    story_name = candidate.get("story_name", "?")
    story_section = candidate.get("story_section", "")
    target_answer = candidate.get("answer", "")
    target_difficulty = candidate.get("evidence_difficulty", "Medium")
    original_question = candidate.get("question", "")

    trace = {
        "story_name": story_name,
        "target_answer": target_answer,
        "target_difficulty": target_difficulty,
        "original_question": original_question,
    }

    # ── Step 1: Answer-Grounded Evidence Planning ──
    evidence_plan = plan_evidence(
        story_name=story_name,
        story_section=story_section,
        target_answer=target_answer,
        target_difficulty=target_difficulty,
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
        "contradictions": evidence_plan.get("contradictions", []),
    }

    if not evidence_plan.get("answer_grounded_evidence_parse_ok"):
        trace["graph_result"] = {"graph_valid": False, "reason": "evidence plan parse failed"}
        trace["qg_result"] = {"generated_question": "", "error": "evidence plan parse failed"}
        return trace

    # ── Step 2: Build modified candidate for graph extraction ──
    # Suppress original question to prevent leakage into graph prompt
    graph_candidate = {
        "story_name": story_name,
        "story_section": story_section,
        "question": "",  # NO original question in graph prompt
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
            "nodes": [],
            "edges": [],
        }

    trace["graph_result"] = {
        "graph_valid": graph_record.get("graph_valid", False),
        "graph_validation_reason": graph_record.get("graph_validation_reason", ""),
        "num_nodes": len(graph_record.get("nodes", [])),
        "num_edges": len(graph_record.get("edges", [])),
        "diagnostics": graph_record.get("diagnostics", {}),
    }

    if not graph_record.get("graph_valid"):
        trace["qg_result"] = {
            "method": "Ours_AnswerGrounded",
            "generated_question": "",
            "error": f"graph_invalid: {graph_record.get('graph_validation_reason', '?')}",
        }
        return trace

    # ── Step 4: Question Generation ──
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

    trace["qg_result"] = {
        "method": "Ours_AnswerGrounded",
        "generated_question": qg_result.get("generated_question", ""),
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
    }

    return trace


def run_baseline_qg(candidate, method="Direct"):
    """Run a baseline QG method for comparison."""
    story_section = candidate.get("story_section", "")
    target_answer = candidate.get("answer", "")
    target_difficulty = candidate.get("evidence_difficulty", "Medium")

    if method == "Direct":
        result, attempts = generate_direct(story_section, target_answer, target_difficulty)
    else:
        result, attempts = generate_icl(story_section, target_answer, target_difficulty)

    return {
        "method": method,
        "generated_question": result.get("generated_question", ""),
        "attempts": attempts,
        "parse_ok": result.get("parse_ok", False),
        "generation_error": result.get("generation_error", ""),
    }


# ── Report ────────────────────────────────────────────────────────

def print_report(results, output_dir):
    """Print a summary report of smoke test results."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "STAGE_C_SMOKE_TEST_REPORT.md")

    n = len(results)

    # Compute stats
    ep_ok = sum(1 for r in results if r["evidence_plan"].get("parse_ok"))
    ep_valid = sum(1 for r in results if r["evidence_plan"].get("valid") == "yes")
    graph_valid = sum(1 for r in results if r["graph_result"].get("graph_valid"))
    qg_ok = sum(1 for r in results if r["qg_result"].get("generated_question"))
    qg_parse = sum(1 for r in results if r["qg_result"].get("parse_ok"))
    qg_selfcheck = sum(1 for r in results if r["qg_result"].get("self_check_pass"))

    # By difficulty
    by_diff = defaultdict(list)
    for r in results:
        by_diff[r["target_difficulty"]].append(r)

    lines = []
    lines.append("# Stage C Smoke Test: Answer-Grounded Evidence → Graph → QG")
    lines.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\nTotal candidates: {n}")

    # ── Summary ──
    lines.append("\n## 1. Pipeline Summary\n")
    lines.append("| Stage | Pass | Total | Pct |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Evidence Plan parse | {ep_ok} | {n} | {100*ep_ok/n:.1f}% |")
    lines.append(f"| Evidence Plan valid | {ep_valid} | {n} | {100*ep_valid/n:.1f}% |")
    lines.append(f"| Graph valid | {graph_valid} | {n} | {100*graph_valid/n:.1f}% |")
    lines.append(f"| QG question generated | {qg_ok} | {n} | {100*qg_ok/n:.1f}% |")
    lines.append(f"| QG parse ok | {qg_parse} | {n} | {100*qg_parse/n:.1f}% |")
    lines.append(f"| QG self-check pass | {qg_selfcheck} | {n} | {100*qg_selfcheck/n:.1f}% |")
    lines.append(f"| **End-to-end pass** | **{qg_selfcheck}** | **{n}** | **{100*qg_selfcheck/n:.1f}%** |")

    # ── By difficulty ──
    lines.append("\n## 2. By Difficulty\n")
    lines.append("| Difficulty | Evidence Parse | Evidence Valid | Graph Valid | QG Question | QG Parse | QG Self-Check |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for diff in ["Easy", "Medium", "Hard"]:
        drs = by_diff[diff]
        nd = len(drs)
        if nd == 0:
            continue
        dep = sum(1 for r in drs if r["evidence_plan"].get("parse_ok"))
        dev = sum(1 for r in drs if r["evidence_plan"].get("valid") == "yes")
        dgv = sum(1 for r in drs if r["graph_result"].get("graph_valid"))
        dqg = sum(1 for r in drs if r["qg_result"].get("generated_question"))
        dqp = sum(1 for r in drs if r["qg_result"].get("parse_ok"))
        dqs = sum(1 for r in drs if r["qg_result"].get("self_check_pass"))
        lines.append(f"| {diff} | {dep}/{nd} | {dev}/{nd} | {dgv}/{nd} | {dqg}/{nd} | {dqp}/{nd} | {dqs}/{nd} |")

    # ── Per-candidate details ──
    lines.append("\n## 3. Per-Candidate Details\n")
    for i, r in enumerate(results, 1):
        ep = r["evidence_plan"]
        gr = r["graph_result"]
        qg = r["qg_result"]

        lines.append(f"### {i}. {r['story_name']} [{r['target_difficulty']}]\n")
        lines.append(f"- **Original question:** {r['original_question'][:100]}")
        lines.append(f"- **Target answer:** {r['target_answer'][:80]}")
        lines.append(f"- **Evidence plan:** parse={'OK' if ep.get('parse_ok') else 'FAIL'}, "
                     f"valid={ep.get('valid','?')}, num_req={ep.get('num_required_sentences','?')}")
        lines.append(f"  - Required: {ep.get('required_evidence_sentences', [])}")
        lines.append(f"  - Bridge: {ep.get('bridge_sentence_ids', [])}")
        lines.append(f"  - Reasoning: {ep.get('reasoning_operation', '?')}")
        if ep.get("contradictions"):
            lines.append(f"  - Contradictions: {ep['contradictions']}")
        lines.append(f"- **Graph:** valid={'yes' if gr.get('graph_valid') else 'no'}, "
                     f"nodes={gr.get('num_nodes','?')}, edges={gr.get('num_edges','?')}")
        if not gr.get("graph_valid"):
            lines.append(f"  - Reason: {gr.get('graph_validation_reason', '?')[:120]}")
        lines.append(f"- **Generated question:** {qg.get('generated_question', '(none)')[:120]}")
        lines.append(f"- **QG status:** parse={'OK' if qg.get('parse_ok') else 'FAIL'}, "
                     f"self_check={'PASS' if qg.get('self_check_pass') else 'FAIL'}, "
                     f"attempts={qg.get('attempts','?')}")
        if qg.get("generation_error"):
            lines.append(f"  - Error: {qg.get('generation_error')}")
        lines.append(f"  - Graph policy: {qg.get('graph_policy', '?')}")
        lines.append(f"  - Policy reason: {qg.get('graph_policy_reason', '?')[:100]}")
        lines.append("")

    # ── Write ──
    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {report_path}")

    # Also save full trace JSON
    trace_path = os.path.join(output_dir, "stage_c_smoke_traces.json")
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Full traces saved to: {trace_path}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage C: Answer-Grounded Evidence → Graph → QG smoke test")
    parser.add_argument("--calibrated",
                        default="outputs/runs/fairytale_target_calibration_20260513/candidates.calibrated.jsonl",
                        help="Path to calibrated candidates JSONL")
    parser.add_argument("--output_dir",
                        default="outputs/runs/stage_c_answer_grounded_smoke_20260514",
                        help="Output directory")
    parser.add_argument("--n_per_difficulty", type=int, default=3,
                        help="Number of candidates per difficulty level")
    parser.add_argument("--model", default=None,
                        help="Model override for evidence planner")
    args = parser.parse_args()

    print("=" * 70)
    print("Stage C: Answer-Grounded Evidence → Graph → QG Smoke Test")
    print("=" * 70)

    # Load calibrated candidates
    print(f"\nLoading calibrated candidates from: {args.calibrated}")
    candidates = load_calibrated_candidates(
        args.calibrated, n_per_difficulty=args.n_per_difficulty)

    if not candidates:
        print("ERROR: No calibrated candidates found!")
        sys.exit(1)

    # Run answer-grounded pipeline on each
    results = []
    for i, c in enumerate(candidates):
        print(f"\n[{i+1}/{len(candidates)}] {c['story_name']} [{c['evidence_difficulty']}]")
        try:
            result = run_answer_grounded_pipeline(c, model=args.model)
        except Exception as e:
            result = {
                "story_name": c.get("story_name", "?"),
                "target_difficulty": c.get("evidence_difficulty", "?"),
                "target_answer": c.get("answer", ""),
                "original_question": c.get("question", ""),
                "evidence_plan": {"parse_ok": False, "valid": "no"},
                "graph_result": {"graph_valid": False},
                "qg_result": {"generated_question": "", "attempts": 0, "parse_ok": False, "error": str(e)},
            }
        results.append(result)

        # Brief status
        ep = result["evidence_plan"]
        gr = result["graph_result"]
        qg = result["qg_result"]
        print(f"  Evidence: {'OK' if ep.get('parse_ok') else 'FAIL'} | "
              f"Graph: {'OK' if gr.get('graph_valid') else 'FAIL'} | "
              f"QG: {'OK' if qg.get('generated_question') else 'FAIL'} "
              f"({qg.get('attempts', '?')} attempts)")

        time.sleep(0.5)

    # Print report
    print("\n" + "=" * 70)
    print_report(results, args.output_dir)


if __name__ == "__main__":
    main()
