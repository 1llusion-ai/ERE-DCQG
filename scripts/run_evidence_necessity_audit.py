"""Evidence Necessity Audit.

Mines Easy / Medium / Hard evidence candidates from validation documents
and produces an audit report.  Phase 1: audit only, no question generation.

Includes ablation probe to distinguish background context from true
answer-identification necessity.

Usage:
    python -m scripts.run_evidence_necessity_audit --limit 200
    python -m scripts.run_evidence_necessity_audit --limit 50 --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dcqg.utils.jsonl import read_jsonl, write_jsonl
from dcqg.path.evidence_necessity import EvidenceNecessityMiner


def parse_args():
    p = argparse.ArgumentParser(description="Evidence Necessity Audit")
    p.add_argument("--input", default="data/raw/maven_ere/valid.jsonl",
                   help="Input JSONL file")
    p.add_argument("--limit", type=int, default=200,
                   help="Max documents to process")
    p.add_argument("--max_candidates", type=int, default=15,
                   help="Max candidates per document")
    p.add_argument("--context_window", type=int, default=5,
                   help="Sentence window around answer sentence")
    p.add_argument("--model", default=None,
                   help="LLM model (default: JUDGE_MODEL from .env)")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: auto-generated)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing partial output")
    return p.parse_args()


def _default_output_dir():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"outputs/runs/evidence_necessity_audit_{ts}"


def _write_report(candidates, output_path):
    """Write the audit report to a markdown file."""
    total = len(candidates)
    if total == 0:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Evidence Necessity Audit Report\n\nNo candidates found.\n")
        return

    # --- Potential difficulty counts ---
    pot_counts = Counter(c.get("potential_evidence_difficulty", "Easy") for c in candidates)
    pot_easy = pot_counts.get("Easy", 0)
    pot_med = pot_counts.get("Medium", 0)
    pot_hard = pot_counts.get("Hard", 0)

    # --- Verified difficulty counts ---
    ver_counts = Counter(c.get("verified_evidence_difficulty", "Easy") for c in candidates)
    ver_easy = ver_counts.get("Easy", 0)
    ver_med = ver_counts.get("Medium", 0)
    ver_hard = ver_counts.get("Hard", 0)

    # --- Ablation field distributions ---
    answer_only_counts = Counter(c.get("answer_only_can_identify_answer", "N/A") for c in candidates)
    anchor_answer_counts = Counter(c.get("anchor_answer_can_identify_answer", "N/A") for c in candidates)
    full_evidence_counts = Counter(c.get("full_evidence_can_identify_answer", "N/A") for c in candidates)
    removal_counts = Counter(c.get("bridge_removal_effect", "N/A") for c in candidates)
    nec_type_counts = Counter(c.get("necessity_type", "N/A") for c in candidates)

    # --- Hard-only ablation distributions (potential Hard) ---
    pot_hard_cands = [c for c in candidates if c.get("potential_evidence_difficulty") == "Hard"]
    pot_hard_answer_only = Counter(c.get("answer_only_can_identify_answer", "N/A") for c in pot_hard_cands)
    pot_hard_removal = Counter(c.get("bridge_removal_effect", "N/A") for c in pot_hard_cands)
    pot_hard_nec_type = Counter(c.get("necessity_type", "N/A") for c in pot_hard_cands)

    # --- Verified Hard candidates ---
    ver_hard_cands = [c for c in candidates if c.get("verified_evidence_difficulty") == "Hard"]

    # --- Conversion rate ---
    potential_to_verified = (100 * ver_hard / pot_hard) if pot_hard > 0 else 0

    # --- Other distributions ---
    reasoning_op_counts = Counter(c.get("reasoning_operation", "N/A") for c in candidates)
    answer_type_counts = Counter(c.get("answer_event_type", "N/A") for c in candidates)
    locality_counts = Counter(c.get("answer_locality", "N/A") for c in candidates)
    necessity_counts = Counter(c.get("evidence_necessity", "N/A") for c in candidates)
    num_req_counts = Counter(c.get("num_required_sentences", 1) for c in candidates)

    # Assessment status
    status_counts = Counter(c.get("assessment_status", "N/A") for c in candidates)

    # Consistency diagnostics
    invalid_span_count = 0
    contradictory_count = 0
    parse_fail_count = 0
    for c in candidates:
        span = c.get("evidence_span", [])
        ans_id = c.get("answer_sent_id", -1)
        if ans_id not in span:
            invalid_span_count += 1
        contradictory_count += c.get("contradiction_count", 0)
        if not c.get("evidence_assessment_parse_ok", True):
            parse_fail_count += 1

    # Unique documents
    unique_docs = len(set(c.get("doc_id", "") for c in candidates))

    # ─── Build report ───
    lines = []
    lines.append("# Evidence Necessity Audit Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Section 1: Overview
    lines.append("## 1. Overview")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|---|---:|")
    lines.append(f"| Total documents processed | {unique_docs} |")
    lines.append(f"| Total candidates | {total} |")
    lines.append(f"| Assessment OK | {status_counts.get('ok', 0)} |")
    lines.append(f"| Assessment LLM error | {status_counts.get('llm_error', 0)} |")
    lines.append("")

    # Section 2: Consistency Diagnostics
    lines.append("## 2. Consistency Diagnostics")
    lines.append("")
    lines.append("| Diagnostic | Count |")
    lines.append("|---|---:|")
    lines.append(f"| Parse failures | {parse_fail_count} |")
    lines.append(f"| Invalid spans | {invalid_span_count} |")
    lines.append(f"| Contradictions fixed | {contradictory_count} |")
    lines.append(f"| Status = ok | {status_counts.get('ok', 0)} |")
    lines.append(f"| Status = llm_error | {status_counts.get('llm_error', 0)} |")
    lines.append("")

    # Section 3: Potential vs Verified Difficulty
    lines.append("## 3. Potential vs Verified Difficulty")
    lines.append("")
    lines.append("| Level | Potential | Verified |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Easy | {pot_easy} ({100*pot_easy/total:.1f}%) | {ver_easy} ({100*ver_easy/total:.1f}%) |")
    lines.append(f"| Medium | {pot_med} ({100*pot_med/total:.1f}%) | {ver_med} ({100*ver_med/total:.1f}%) |")
    lines.append(f"| Hard | {pot_hard} ({100*pot_hard/total:.1f}%) | {ver_hard} ({100*ver_hard/total:.1f}%) |")
    lines.append("")
    lines.append(f"**Potential Hard -> Verified Hard conversion:** {potential_to_verified:.1f}% ({ver_hard}/{pot_hard})")
    lines.append("")

    # Section 4: answer_only_can_identify_answer distribution
    lines.append("## 4. Ablation: answer_only_can_identify_answer")
    lines.append("")
    lines.append("| Value | All | Potential Hard only |")
    lines.append("|---|---:|---:|")
    for val in ["yes", "partial", "no"]:
        all_cnt = answer_only_counts.get(val, 0)
        hard_cnt = pot_hard_answer_only.get(val, 0)
        lines.append(f"| {val} | {all_cnt} | {hard_cnt} |")
    lines.append("")

    # Section 5: bridge_removal_effect distribution
    lines.append("## 5. Ablation: bridge_removal_effect")
    lines.append("")
    lines.append("| Value | All | Potential Hard only |")
    lines.append("|---|---:|---:|")
    for val in ["none", "harder", "ambiguous", "unanswerable"]:
        all_cnt = removal_counts.get(val, 0)
        hard_cnt = pot_hard_removal.get(val, 0)
        lines.append(f"| {val} | {all_cnt} | {hard_cnt} |")
    lines.append("")

    # Section 6: necessity_type distribution
    lines.append("## 6. Ablation: necessity_type")
    lines.append("")
    lines.append("| Type | All | Potential Hard only |")
    lines.append("|---|---:|---:|")
    for val in sorted(set(list(nec_type_counts.keys()) + list(pot_hard_nec_type.keys()))):
        all_cnt = nec_type_counts.get(val, 0)
        hard_cnt = pot_hard_nec_type.get(val, 0)
        lines.append(f"| {val} | {all_cnt} | {hard_cnt} |")
    lines.append("")

    # Section 7: Hard candidate detail
    lines.append("## 7. Hard Candidate Detail")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|---|---:|")
    lines.append(f"| Potential Hard total | {pot_hard} |")
    lines.append(f"| Potential Hard + alone_sufficient=no | {sum(1 for c in pot_hard_cands if c.get('answer_sentence_alone_sufficient') == 'no')} |")
    lines.append(f"| Verified Hard total | {ver_hard} |")
    lines.append(f"| Verified Hard + answer_only=no | {sum(1 for c in ver_hard_cands if c.get('answer_only_can_identify_answer') == 'no')} |")
    lines.append("")

    # Section 8: Distribution by reasoning operation
    ver_hard_op = Counter(c.get("reasoning_operation", "N/A") for c in ver_hard_cands)
    lines.append("## 8. Distribution by Reasoning Operation")
    lines.append("")
    lines.append("| Operation | All | Verified Hard |")
    lines.append("|---|---:|---:|")
    for op in sorted(set(list(reasoning_op_counts.keys()) + list(ver_hard_op.keys()))):
        lines.append(f"| {op} | {reasoning_op_counts.get(op, 0)} | {ver_hard_op.get(op, 0)} |")
    lines.append("")

    # Section 9: Distribution by answer event type
    ver_hard_type = Counter(c.get("answer_event_type", "N/A") for c in ver_hard_cands)
    lines.append("## 9. Distribution by Answer Event Type")
    lines.append("")
    lines.append("| Event Type | All | Verified Hard |")
    lines.append("|---|---:|---:|")
    for at in sorted(set(list(answer_type_counts.keys()) + list(ver_hard_type.keys()))):
        lines.append(f"| {at} | {answer_type_counts.get(at, 0)} | {ver_hard_type.get(at, 0)} |")
    lines.append("")

    # Section 10: Answer locality / num_required / evidence_necessity
    lines.append("## 10. Answer Locality / num_required / evidence_necessity")
    lines.append("")
    lines.append("| Locality | Count | Pct |")
    lines.append("|---|---:|---:|")
    for loc in ["single_sentence", "two_sentence", "multi_sentence"]:
        cnt = locality_counts.get(loc, 0)
        lines.append(f"| {loc} | {cnt} | {100*cnt/total:.1f}% |")
    lines.append("")
    lines.append("| # Required | Count | Pct |")
    lines.append("|---|---:|---:|")
    for n_req in sorted(num_req_counts.keys()):
        cnt = num_req_counts[n_req]
        lines.append(f"| {n_req} | {cnt} | {100*cnt/total:.1f}% |")
    lines.append("")
    lines.append("| Necessity | Count | Pct |")
    lines.append("|---|---:|---:|")
    for nec in ["weak", "partial", "strong"]:
        cnt = necessity_counts.get(nec, 0)
        lines.append(f"| {nec} | {cnt} | {100*cnt/total:.1f}% |")
    lines.append("")

    # Section 11: Verified Hard trace examples
    lines.append("## 11. Verified Hard Trace Examples (up to 10)")
    lines.append("")

    vhard_examples = sorted(
        ver_hard_cands,
        key=lambda c: (
            0 if c.get("bridge_removal_effect") == "unanswerable" else 1,
            0 if c.get("necessity_type") == "answer_identification" else 1,
        )
    )[:10]

    if not vhard_examples:
        lines.append("No verified Hard candidates found.\n")
    else:
        for idx, c in enumerate(vhard_examples, 1):
            lines.append(f"### Example {idx}")
            lines.append("")
            lines.append(f"**Document:** {c.get('title', 'N/A')} (doc_id={c.get('doc_id', 'N/A')})")
            lines.append(f"**Answer trigger:** \"{c.get('answer_trigger', '')}\" ({c.get('answer_event_type', '')})")
            lines.append(f"**Answer phrase:** \"{c.get('answer_phrase', '')}\"")
            lines.append(f"**Answer sentence [S{c.get('answer_sent_id', -1)}]:** {c.get('answer_sentence', '')}")
            lines.append("")
            lines.append("**Ablation assessment:**")
            lines.append(f"- potential_evidence_difficulty: {c.get('potential_evidence_difficulty', 'N/A')}")
            lines.append(f"- verified_evidence_difficulty: {c.get('verified_evidence_difficulty', 'N/A')}")
            lines.append(f"- answer_only_can_identify_answer: {c.get('answer_only_can_identify_answer', 'N/A')}")
            lines.append(f"- anchor_answer_can_identify_answer: {c.get('anchor_answer_can_identify_answer', 'N/A')}")
            lines.append(f"- full_evidence_can_identify_answer: {c.get('full_evidence_can_identify_answer', 'N/A')}")
            lines.append(f"- bridge_removal_effect: {c.get('bridge_removal_effect', 'N/A')}")
            lines.append(f"- necessity_type: {c.get('necessity_type', 'N/A')}")
            lines.append(f"- ablation_reason: {c.get('ablation_reason', 'N/A')}")
            lines.append("")
            lines.append("**Evidence structure:**")
            lines.append(f"- anchor_sentence_ids: {c.get('anchor_sentence_ids', [])}")
            lines.append(f"- bridge_sentence_ids: {c.get('bridge_sentence_ids', [])}")
            lines.append(f"- evidence_span: {c.get('evidence_span', [])}")
            lines.append(f"- num_required_sentences: {c.get('num_required_sentences', 1)}")
            lines.append(f"- reasoning_operation: {c.get('reasoning_operation', 'N/A')}")
            lines.append("")

            # Show anchor/bridge sentences
            ctx = c.get("context_sentences", [])
            ctx_dict = {sid: text for sid, text in ctx}
            anchor_ids = c.get("anchor_sentence_ids", [])
            bridge_ids = c.get("bridge_sentence_ids", [])
            if anchor_ids:
                lines.append("**Anchor sentences:**")
                for sid in anchor_ids:
                    lines.append(f"- [S{sid}] {ctx_dict.get(sid, '(not in context window)')}")
                lines.append("")
            if bridge_ids:
                lines.append("**Bridge sentences:**")
                for sid in bridge_ids:
                    lines.append(f"- [S{sid}] {ctx_dict.get(sid, '(not in context window)')}")
                lines.append("")

            lines.append("---")
            lines.append("")

    # Section 12: Success criteria
    lines.append("## 12. Success Criteria Check")
    lines.append("")
    lines.append(f"- Potential Hard >= 100: **{'PASS' if pot_hard >= 100 else 'FAIL'}** ({pot_hard})")
    lines.append(f"- Verified Hard >= 100: **{'PASS' if ver_hard >= 100 else 'FAIL'}** ({ver_hard})")
    lines.append(f"- Verified Hard + answer_only=no: **{'PASS' if sum(1 for c in ver_hard_cands if c.get('answer_only_can_identify_answer') == 'no') >= 50 else 'NEEDS CHECK'}** ({sum(1 for c in ver_hard_cands if c.get('answer_only_can_identify_answer') == 'no')})")
    lines.append(f"- Potential->Verified conversion: {potential_to_verified:.1f}%")
    lines.append("")
    if ver_hard >= 100:
        lines.append("**Conclusion:** Enough verified Hard candidates exist. "
                      "Proceed to design evidence-role-aware QG prompt.")
    elif pot_hard >= 100 and ver_hard < 100:
        lines.append("**Conclusion:** Many potential Hard candidates are "
                      "background-context-only. Review necessity_type distribution "
                      "before proceeding.")
    else:
        lines.append("**Conclusion:** Too few Hard evidence candidates. "
                      "Need to change data construction method or answer types.")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _load_done_ids(output_jsonl):
    """Load doc_ids already processed (for resume)."""
    done = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec.get("doc_id", ""))
                except json.JSONDecodeError:
                    pass
    return done


def main():
    args = parse_args()

    output_dir = args.output_dir or _default_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    output_jsonl = os.path.join(output_dir, "candidates.jsonl")
    report_path = os.path.join(output_dir, "audit_report.md")

    print(f"=== Evidence Necessity Audit ===")
    print(f"Input:        {args.input}")
    print(f"Limit:        {args.limit} documents")
    print(f"Model:        {args.model or '(default JUDGE_MODEL)'}")
    print(f"Output dir:   {output_dir}")
    print()

    # Load documents
    docs = read_jsonl(args.input, n=args.limit)
    print(f"Loaded {len(docs)} documents")

    # Resume support
    done_ids = set()
    existing_candidates = []
    if args.resume:
        done_ids = _load_done_ids(output_jsonl)
        if done_ids:
            with open(output_jsonl, encoding="utf-8") as f:
                for line in f:
                    try:
                        existing_candidates.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            print(f"Resuming: {len(done_ids)} docs already processed, "
                  f"{len(existing_candidates)} candidates loaded")

    # Filter out already-processed docs
    remaining = [d for d in docs if d.get("id", "") not in done_ids]
    print(f"Processing {len(remaining)} remaining documents...")
    print()

    miner = EvidenceNecessityMiner(
        context_window=args.context_window,
        max_candidates_per_doc=args.max_candidates,
        model=args.model,
    )

    all_candidates = list(existing_candidates)
    stats = {"ok": 0, "error": 0, "total_candidates": 0}

    # Open output file for incremental writing
    out_f = open(output_jsonl, "a" if args.resume else "w", encoding="utf-8")

    try:
        for i, doc in enumerate(remaining):
            doc_id = doc.get("id", f"doc_{i}")
            doc_title = doc.get("title", "N/A")

            t0 = time.time()
            try:
                candidates = miner.mine_document(doc)
            except Exception as e:
                print(f"  [{i+1}/{len(remaining)}] ERROR {doc_title[:40]}: {e}")
                stats["error"] += 1
                continue
            elapsed = time.time() - t0

            for c in candidates:
                out_f.write(json.dumps(c, ensure_ascii=False) + "\n")
            out_f.flush()
            all_candidates.extend(candidates)

            n_cands = len(candidates)
            stats["total_candidates"] += n_cands
            stats["ok"] += 1

            # Progress — use verified difficulty for display
            vhard = sum(1 for c in candidates if c.get("verified_evidence_difficulty") == "Hard")
            vmed = sum(1 for c in candidates if c.get("verified_evidence_difficulty") == "Medium")
            veasy = sum(1 for c in candidates if c.get("verified_evidence_difficulty") == "Easy")

            if (i + 1) % 10 == 0 or n_cands > 0:
                print(f"  [{i+1}/{len(remaining)}] {doc_title[:45]:45s} "
                      f"cands={n_cands:2d} (vE={veasy} vM={vmed} vH={vhard}) "
                      f"{elapsed:.1f}s")

            # Rate limiting
            time.sleep(0.2)

    finally:
        out_f.close()

    print()
    print(f"=== Audit Complete ===")
    print(f"Documents processed: {stats['ok']}")
    print(f"Documents with errors: {stats['error']}")
    print(f"Total candidates: {len(all_candidates)}")

    # Potential difficulty summary
    pot_counts = Counter(c.get("potential_evidence_difficulty", "Easy") for c in all_candidates)
    print(f"\nPotential difficulty:")
    print(f"  Easy:   {pot_counts.get('Easy', 0)}")
    print(f"  Medium: {pot_counts.get('Medium', 0)}")
    print(f"  Hard:   {pot_counts.get('Hard', 0)}")

    # Verified difficulty summary
    ver_counts = Counter(c.get("verified_evidence_difficulty", "Easy") for c in all_candidates)
    print(f"\nVerified difficulty:")
    print(f"  Easy:   {ver_counts.get('Easy', 0)}")
    print(f"  Medium: {ver_counts.get('Medium', 0)}")
    print(f"  Hard:   {ver_counts.get('Hard', 0)}")

    # Conversion
    pot_h = pot_counts.get('Hard', 0)
    ver_h = ver_counts.get('Hard', 0)
    if pot_h > 0:
        print(f"\nPotential->Verified Hard conversion: {100*ver_h/pot_h:.1f}% ({ver_h}/{pot_h})")

    # Write report
    _write_report(all_candidates, report_path)
    print(f"\nReport written to: {report_path}")
    print(f"Candidates written to: {output_jsonl}")


if __name__ == "__main__":
    main()
