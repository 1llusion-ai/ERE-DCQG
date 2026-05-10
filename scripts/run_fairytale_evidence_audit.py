"""FairytaleQA Evidence Audit.

Mines Easy / Medium / Hard evidence candidates from FairytaleQA QA pairs
and produces an audit report.  Phase 1: audit only, no question generation.

Usage:
    python -m scripts.run_fairytale_evidence_audit --split validation --limit 50
    python -m scripts.run_fairytale_evidence_audit --split train --limit 200
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dcqg.datasets.fairytaleqa_loader import load_fairytaleqa
from dcqg.path.fairytale_evidence_audit import FairytaleEvidenceAuditor, _split_sentences


def parse_args():
    p = argparse.ArgumentParser(description="FairytaleQA Evidence Audit")
    p.add_argument("--split", default="validation",
                   help="Dataset split: train, validation, test")
    p.add_argument("--limit", type=int, default=50,
                   help="Max QA pairs to process")
    p.add_argument("--batch_size", type=int, default=8,
                   help="QA pairs per LLM call")
    p.add_argument("--model", default=None,
                   help="LLM model (default: JUDGE_MODEL from .env)")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: auto-generated)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing partial output")
    return p.parse_args()


def _default_output_dir():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"outputs/runs/fairytale_evidence_audit_{ts}"


def _write_report(candidates, output_path, load_info):
    """Write the audit report."""
    total = len(candidates)
    if total == 0:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# FairytaleQA Evidence Audit Report\n\nNo candidates found.\n")
        return

    # --- Difficulty ---
    diff_counts = Counter(c.get("evidence_difficulty", "Easy") for c in candidates)
    easy_n = diff_counts.get("Easy", 0)
    med_n = diff_counts.get("Medium", 0)
    hard_n = diff_counts.get("Hard", 0)

    # --- Ablation fields ---
    ans_only_counts = Counter(c.get("answer_sentence_alone_sufficient", "N/A") for c in candidates)
    section_counts = Counter(c.get("section_evidence_sufficient", "N/A") for c in candidates)
    removal_counts = Counter(c.get("bridge_removal_effect", "N/A") for c in candidates)
    nec_type_counts = Counter(c.get("necessity_type", "N/A") for c in candidates)
    reasoning_op_counts = Counter(c.get("reasoning_operation", "N/A") for c in candidates)

    # --- Fairytale label cross-tabs ---
    local_or_sum_counts = Counter(c.get("local_or_sum", "N/A") for c in candidates)
    ex_or_im_counts = Counter(c.get("ex_or_im", "N/A") for c in candidates)
    attribute_counts = Counter(c.get("attribute", "N/A") for c in candidates)

    # Cross-tab: local_or_sum × difficulty
    los_x_diff = {}
    for c in candidates:
        los = c.get("local_or_sum", "N/A")
        diff = c.get("evidence_difficulty", "Easy")
        los_x_diff.setdefault(los, Counter())[diff] += 1

    # Cross-tab: ex_or_im × difficulty
    eoi_x_diff = {}
    for c in candidates:
        eoi = c.get("ex_or_im", "N/A")
        diff = c.get("evidence_difficulty", "Easy")
        eoi_x_diff.setdefault(eoi, Counter())[diff] += 1

    # Cross-tab: attribute × difficulty
    attr_x_diff = {}
    for c in candidates:
        attr = c.get("attribute", "N/A")
        diff = c.get("evidence_difficulty", "Easy")
        attr_x_diff.setdefault(attr, Counter())[diff] += 1

    # --- Hard-only ---
    hard_cands = [c for c in candidates if c.get("evidence_difficulty") == "Hard"]
    hard_reasoning = Counter(c.get("reasoning_operation", "N/A") for c in hard_cands)
    hard_nec_type = Counter(c.get("necessity_type", "N/A") for c in hard_cands)

    # --- Status ---
    status_counts = Counter(c.get("assessment_status", "N/A") for c in candidates)

    # --- Consistency diagnostics ---
    contradictions = sum(c.get("contradiction_count", 0) for c in candidates)
    parse_fail = sum(1 for c in candidates if not c.get("fairytale_evidence_parse_ok", True))
    missing_trace = sum(1 for c in candidates
                        if not c.get("fairytale_evidence_prompt")
                        or not c.get("fairytale_evidence_raw"))

    # Count invalid required_evidence_sentences (out of range)
    invalid_req_count = 0
    for c in candidates:
        num_sents = c.get("num_sentences_in_section", 0)
        for sid in c.get("required_evidence_sentences", []):
            if sid >= num_sents:
                invalid_req_count += 1

    # Count Hard candidates that violate Hard classification rules
    hard_violation = 0
    for c in candidates:
        if c.get("evidence_difficulty") != "Hard":
            continue
        suff = c.get("answer_sentence_alone_sufficient", "yes")
        num_req = c.get("num_required_sentences", 1)
        removal = c.get("bridge_removal_effect", "none")
        nec_type = c.get("necessity_type", "background_context")
        if not (suff == "no"
                and num_req >= 3
                and removal in ("ambiguous", "unanswerable")
                and nec_type in ("answer_identification", "disambiguation",
                                 "causal_bridge", "temporal_bridge",
                                 "motivation_bridge", "summary_synthesis")):
            hard_violation += 1

    lines = []
    lines.append("# FairytaleQA Evidence Audit Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Section 1: Dataset loading summary
    lines.append("## 1. Dataset Loading Summary")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Split | {load_info.get('split', 'N/A')} |")
    lines.append(f"| Total QA pairs loaded | {load_info.get('total_loaded', 0)} |")
    lines.append(f"| QA pairs assessed | {total} |")
    lines.append(f"| Fields available | {load_info.get('fields_available', 'N/A')} |")
    lines.append(f"| Source | {load_info.get('source', 'N/A')} |")
    lines.append("")

    # Section 2: Evidence difficulty distribution
    lines.append("## 2. Evidence Difficulty Distribution")
    lines.append("")
    lines.append("| Difficulty | Count | Pct |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Easy | {easy_n} | {100*easy_n/total:.1f}% |")
    lines.append(f"| Medium | {med_n} | {100*med_n/total:.1f}% |")
    lines.append(f"| Hard | {hard_n} | {100*hard_n/total:.1f}% |")
    lines.append("")

    # Section 3: Fairytale labels vs evidence difficulty
    lines.append("## 3. Fairytale Labels vs Evidence Difficulty")
    lines.append("")
    lines.append("### 3a. local-or-sum x difficulty")
    lines.append("")
    lines.append("| local-or-sum | Easy | Medium | Hard | Total |")
    lines.append("|---|---:|---:|---:|---:|")
    for los in sorted(los_x_diff.keys()):
        d = los_x_diff[los]
        t = sum(d.values())
        lines.append(f"| {los} | {d.get('Easy', 0)} | {d.get('Medium', 0)} | {d.get('Hard', 0)} | {t} |")
    lines.append("")

    lines.append("### 3b. ex-or-im x difficulty")
    lines.append("")
    lines.append("| ex-or-im | Easy | Medium | Hard | Total |")
    lines.append("|---|---:|---:|---:|---:|")
    for eoi in sorted(eoi_x_diff.keys()):
        d = eoi_x_diff[eoi]
        t = sum(d.values())
        lines.append(f"| {eoi} | {d.get('Easy', 0)} | {d.get('Medium', 0)} | {d.get('Hard', 0)} | {t} |")
    lines.append("")

    lines.append("### 3c. attribute x difficulty")
    lines.append("")
    lines.append("| attribute | Easy | Medium | Hard | Total |")
    lines.append("|---|---:|---:|---:|---:|")
    for attr in sorted(attr_x_diff.keys()):
        d = attr_x_diff[attr]
        t = sum(d.values())
        lines.append(f"| {attr} | {d.get('Easy', 0)} | {d.get('Medium', 0)} | {d.get('Hard', 0)} | {t} |")
    lines.append("")

    # Section 4: Verified Hard detail
    lines.append("## 4. Verified Hard Detail")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|---|---:|")
    lines.append(f"| Hard count | {hard_n} |")
    lines.append(f"| Hard rate | {100*hard_n/total:.1f}% |")
    lines.append("")
    lines.append("### Hard by reasoning_operation")
    lines.append("")
    lines.append("| Operation | Count |")
    lines.append("|---|---:|")
    for op in sorted(hard_reasoning.keys()):
        lines.append(f"| {op} | {hard_reasoning[op]} |")
    lines.append("")
    lines.append("### Hard by necessity_type")
    lines.append("")
    lines.append("| Type | Count |")
    lines.append("|---|---:|")
    for nt in sorted(hard_nec_type.keys()):
        lines.append(f"| {nt} | {hard_nec_type[nt]} |")
    lines.append("")

    # Section 5: Answer sufficiency diagnostics
    lines.append("## 5. Answer Sufficiency Diagnostics")
    lines.append("")
    lines.append("### answer_sentence_alone_sufficient")
    lines.append("")
    lines.append("| Value | Count | Pct |")
    lines.append("|---|---:|---:|")
    for val in ["yes", "partial", "no"]:
        cnt = ans_only_counts.get(val, 0)
        lines.append(f"| {val} | {cnt} | {100*cnt/total:.1f}% |")
    lines.append("")

    lines.append("### section_evidence_sufficient")
    lines.append("")
    lines.append("| Value | Count | Pct |")
    lines.append("|---|---:|---:|")
    for val in ["yes", "partial", "no"]:
        cnt = section_counts.get(val, 0)
        lines.append(f"| {val} | {cnt} | {100*cnt/total:.1f}% |")
    lines.append("")

    lines.append("### bridge_removal_effect")
    lines.append("")
    lines.append("| Value | Count | Pct |")
    lines.append("|---|---:|---:|")
    for val in ["none", "harder", "ambiguous", "unanswerable"]:
        cnt = removal_counts.get(val, 0)
        lines.append(f"| {val} | {cnt} | {100*cnt/total:.1f}% |")
    lines.append("")

    # Section 6: Consistency diagnostics
    lines.append("## 6. Consistency Diagnostics")
    lines.append("")
    lines.append("| Diagnostic | Count |")
    lines.append("|---|---:|")
    lines.append(f"| Parse failures | {parse_fail} |")
    lines.append(f"| Contradictions fixed | {contradictions} |")
    lines.append(f"| Missing trace fields | {missing_trace} |")
    lines.append(f"| Invalid required sentence IDs | {invalid_req_count} |")
    lines.append(f"| Hard validation violations | {hard_violation} |")
    lines.append(f"| Status = ok | {status_counts.get('ok', 0)} |")
    lines.append(f"| Status = llm_error | {status_counts.get('llm_error', 0)} |")
    lines.append("")

    # Section 7: Verified Hard trace examples
    lines.append("## 7. Verified Hard Examples (up to 10)")
    lines.append("")

    hard_examples = sorted(
        hard_cands,
        key=lambda c: (
            0 if c.get("bridge_removal_effect") == "unanswerable" else 1,
            0 if c.get("necessity_type") == "answer_identification" else 1,
        )
    )[:10]

    if not hard_examples:
        lines.append("No verified Hard candidates found.\n")
    else:
        for idx, c in enumerate(hard_examples, 1):
            lines.append(f"### Example {idx}")
            lines.append("")
            lines.append(f"**Story:** {c.get('story_name', 'N/A')}")
            lines.append(f"**Question:** {c.get('question', '')}")
            lines.append(f"**Answer:** {c.get('answer1', '')}")
            if c.get("answer2"):
                lines.append(f"**Answer2:** {c.get('answer2', '')}")
            lines.append(f"**Labels:** local-or-sum={c.get('local_or_sum', 'N/A')}, "
                         f"attribute={c.get('attribute', 'N/A')}, "
                         f"ex-or-im={c.get('ex_or_im', 'N/A')}")
            lines.append("")
            lines.append("**Evidence assessment:**")
            lines.append(f"- answer_sentence_alone_sufficient: {c.get('answer_sentence_alone_sufficient', 'N/A')}")
            lines.append(f"- section_evidence_sufficient: {c.get('section_evidence_sufficient', 'N/A')}")
            lines.append(f"- num_required_sentences: {c.get('num_required_sentences', 0)}")
            lines.append(f"- reasoning_operation: {c.get('reasoning_operation', 'N/A')}")
            lines.append(f"- bridge_removal_effect: {c.get('bridge_removal_effect', 'N/A')}")
            lines.append(f"- necessity_type: {c.get('necessity_type', 'N/A')}")
            lines.append(f"- evidence_necessity_reason: {c.get('evidence_necessity_reason', 'N/A')}")
            lines.append("")

            # Show evidence sentences (must use same splitter as auditor)
            section = c.get("story_section", "")
            sentences = _split_sentences(section)
            req_ids = c.get("required_evidence_sentences", [])
            bridge_ids = c.get("bridge_sentence_ids", [])
            if req_ids:
                lines.append("**Required evidence sentences:**")
                for sid in req_ids:
                    tag = " [BRIDGE]" if sid in bridge_ids else ""
                    if sid < len(sentences):
                        lines.append(f"- [S{sid}] {sentences[sid].strip()}{tag}")
                    else:
                        lines.append(f"- [S{sid}] (out of range){tag}")
                lines.append("")

            lines.append("---")
            lines.append("")

    # Section 8: Comparison note against MAVEN-ERE
    lines.append("## 8. Comparison Note Against MAVEN-ERE")
    lines.append("")
    lines.append("**MAVEN-ERE baseline (current):**")
    lines.append("- QG-eligible Hard rescue: 0/112 blind Hard among judged candidates")
    lines.append("- Quality-pass candidates: 0/21 blind Hard")
    lines.append("- Root cause: answer sentences are locally identifiable; event-path hop count does not translate to answering difficulty")
    lines.append("")
    lines.append(f"**FairytaleQA evidence audit ({total} QA pairs):**")
    lines.append(f"- Easy: {easy_n} ({100*easy_n/total:.1f}%)")
    lines.append(f"- Medium: {med_n} ({100*med_n/total:.1f}%)")
    lines.append(f"- Hard: {hard_n} ({100*hard_n/total:.1f}%)")
    lines.append("")
    if hard_n > 0:
        hard_rate = 100 * hard_n / total
        lines.append(f"**Assessment:** FairytaleQA produces {hard_n} verified Hard candidates "
                      f"({hard_rate:.1f}%) from {total} QA pairs. This is a meaningful "
                      f"improvement over MAVEN-ERE's 0% blind Hard rate. "
                      f"Narrative QA appears more promising for difficulty-controlled QG "
                      f"because answers often require understanding character motivation, "
                      f"causal chains, and multi-sentence inference rather than local "
                      f"event-phrase extraction.")
    else:
        lines.append("**Assessment:** No verified Hard candidates found in this sample. "
                      "A larger sample or different split may be needed.")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _load_done_keys(output_jsonl):
    """Load (story_name, question) pairs already processed."""
    done = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add((rec.get("story_name", ""), rec.get("question", "")))
                except json.JSONDecodeError:
                    pass
    return done


def main():
    args = parse_args()

    output_dir = args.output_dir or _default_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    output_jsonl = os.path.join(output_dir, "candidates.jsonl")
    report_path = os.path.join(output_dir, "audit_report.md")

    print(f"=== FairytaleQA Evidence Audit ===")
    print(f"Split:        {args.split}")
    print(f"Limit:        {args.limit} QA pairs")
    print(f"Model:        {args.model or '(default JUDGE_MODEL)'}")
    print(f"Output dir:   {output_dir}")
    print()

    # Load data
    print("Loading FairytaleQA dataset...")
    try:
        all_records = load_fairytaleqa(split=args.split, limit=args.limit)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    total_loaded = len(all_records)
    print(f"Loaded {total_loaded} QA pairs")

    # Determine available fields
    if all_records:
        fields = [k for k in all_records[0].keys() if all_records[0].get(k)]
        fields_str = ", ".join(fields[:10])
    else:
        fields_str = "none"

    load_info = {
        "split": args.split,
        "total_loaded": total_loaded,
        "fields_available": fields_str,
        "source": "HuggingFace" if total_loaded > 0 else "unknown",
    }

    # Resume support
    done_keys = set()
    existing_candidates = []
    if args.resume:
        done_keys = _load_done_keys(output_jsonl)
        if done_keys:
            with open(output_jsonl, encoding="utf-8") as f:
                for line in f:
                    try:
                        existing_candidates.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            print(f"Resuming: {len(done_keys)} QA pairs already processed")

    # Filter out already-processed
    remaining = [
        r for r in all_records
        if (r.get("story_name", ""), r.get("question", "")) not in done_keys
    ]
    print(f"Processing {len(remaining)} remaining QA pairs...")
    print()

    auditor = FairytaleEvidenceAuditor(
        batch_size=args.batch_size,
        model=args.model,
    )

    all_candidates = list(existing_candidates)
    stats = {"ok": 0, "error": 0, "total_candidates": 0}

    out_f = open(output_jsonl, "a" if args.resume else "w", encoding="utf-8")

    try:
        # Process in batches
        batch_size = args.batch_size
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(remaining) + batch_size - 1) // batch_size

            t0 = time.time()
            try:
                candidates = auditor.audit_batch(batch)
            except Exception as e:
                print(f"  [batch {batch_num}/{total_batches}] ERROR: {e}")
                stats["error"] += len(batch)
                continue
            elapsed = time.time() - t0

            for c in candidates:
                out_f.write(json.dumps(c, ensure_ascii=False) + "\n")
            out_f.flush()
            all_candidates.extend(candidates)

            n_cands = len(candidates)
            stats["total_candidates"] += n_cands
            stats["ok"] += len(batch)

            # Progress
            vhard = sum(1 for c in candidates if c.get("evidence_difficulty") == "Hard")
            vmed = sum(1 for c in candidates if c.get("evidence_difficulty") == "Medium")
            veasy = sum(1 for c in candidates if c.get("evidence_difficulty") == "Easy")

            print(f"  [batch {batch_num}/{total_batches}] "
                  f"q={n_cands:2d} (E={veasy} M={vmed} H={vhard}) "
                  f"{elapsed:.1f}s")

            time.sleep(0.2)

    finally:
        out_f.close()

    print()
    print(f"=== Audit Complete ===")
    print(f"QA pairs processed: {stats['ok']}")
    print(f"Errors: {stats['error']}")
    print(f"Total candidates: {len(all_candidates)}")

    diff_counts = Counter(c.get("evidence_difficulty", "Easy") for c in all_candidates)
    print(f"  Easy:   {diff_counts.get('Easy', 0)}")
    print(f"  Medium: {diff_counts.get('Medium', 0)}")
    print(f"  Hard:   {diff_counts.get('Hard', 0)}")

    _write_report(all_candidates, report_path, load_info)
    print(f"\nReport: {report_path}")
    print(f"Candidates: {output_jsonl}")


if __name__ == "__main__":
    main()
