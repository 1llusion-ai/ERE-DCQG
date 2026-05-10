"""Narrative Evidence Graph Audit.

Extracts and validates narrative evidence graphs from FairytaleQA Hard candidates.

Usage:
    python -m scripts.run_narrative_graph_audit \
        --input outputs/runs/fairytale_evidence_audit_train_implicit_500_20260510/candidates.jsonl \
        --limit 20 \
        --output_dir outputs/runs/narrative_graph_audit_20260510
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

from dcqg.graph.narrative_graph import NarrativeGraphExtractor, _split_sentences


def parse_args():
    p = argparse.ArgumentParser(description="Narrative Evidence Graph Audit")
    p.add_argument("--input", required=True,
                   help="Path to candidates.jsonl from evidence audit")
    p.add_argument("--limit", type=int, default=20,
                   help="Max Hard candidates to extract graphs for")
    p.add_argument("--model", default=None,
                   help="LLM model (default: JUDGE_MODEL from .env)")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: auto-generated)")
    return p.parse_args()


def _default_output_dir():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"outputs/runs/narrative_graph_audit_{ts}"


_TARGET_NEC_TYPES = {
    "motivation_bridge", "causal_bridge", "summary_synthesis", "disambiguation"
}


def _load_hard_candidates(input_path):
    """Load and filter Hard candidates from evidence audit output."""
    candidates = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            if c.get("evidence_difficulty") != "Hard":
                continue
            nec = c.get("necessity_type", "")
            if nec not in _TARGET_NEC_TYPES:
                continue
            if c.get("reasoning_operation") == "explicit_lookup":
                continue
            candidates.append(c)
    return candidates


def _stratified_sample(candidates, limit):
    """Sample candidates to maximize coverage across necessity types and attributes."""
    if len(candidates) <= limit:
        return candidates

    # Group by necessity_type
    by_nec = {}
    for c in candidates:
        by_nec.setdefault(c.get("necessity_type", "other"), []).append(c)

    # Round-robin across groups, taking 1 from each until limit
    selected = []
    groups = list(by_nec.values())
    idx = [0] * len(groups)
    while len(selected) < limit:
        added = False
        for g, group in enumerate(groups):
            if idx[g] < len(group):
                selected.append(group[idx[g]])
                idx[g] += 1
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break

    return selected


def _write_report(graphs, output_path, input_info):
    """Write the audit report."""
    total = len(graphs)
    if total == 0:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Narrative Evidence Graph Audit Report\n\nNo graphs extracted.\n")
        return

    # --- Section 2: Graph extraction success ---
    parse_ok_count = sum(1 for g in graphs if g.get("trace", {}).get("parse_ok", False))
    valid_count = sum(1 for g in graphs if g.get("graph_valid", False))
    fail_reasons = []
    for g in graphs:
        if not g.get("graph_valid", False):
            fail_reasons.append(g.get("graph_validation_reason", "unknown"))

    # --- Section 3: Graph structure statistics ---
    node_counts = [len(g.get("nodes", [])) for g in graphs]
    edge_counts = [len(g.get("edges", [])) for g in graphs]
    avg_nodes = sum(node_counts) / total if total else 0
    avg_edges = sum(edge_counts) / total if total else 0

    node_types = Counter()
    edge_relations = Counter()
    evidence_roles = Counter()
    edge_necessity = Counter()
    for g in graphs:
        for n in g.get("nodes", []):
            node_types[n.get("type", "unknown")] += 1
            evidence_roles[n.get("evidence_role", "unknown")] += 1
        for e in g.get("edges", []):
            edge_relations[e.get("relation", "unknown")] += 1
            edge_necessity[e.get("necessity", "unknown")] += 1

    # --- Section 4: Coverage diagnostics ---
    req_covered = 0
    req_total = 0
    bridge_covered = 0
    bridge_total = 0
    answer_covered = 0
    for g in graphs:
        req_ids = set(g.get("required_evidence_sentences", []))
        bridge_ids = set(g.get("bridge_sentence_ids", []))
        node_sids = {n.get("sentence_id") for n in g.get("nodes", [])}
        bridge_sids = {n.get("sentence_id") for n in g.get("nodes", [])
                       if n.get("evidence_role") == "bridge"}
        has_answer = any(n.get("evidence_role") == "answer" for n in g.get("nodes", []))

        req_total += len(req_ids)
        req_covered += len(req_ids & node_sids)
        bridge_total += len(bridge_ids)
        bridge_covered += len(bridge_ids & bridge_sids)
        if has_answer:
            answer_covered += 1

    req_cov_pct = 100 * req_covered / req_total if req_total else 0
    bridge_cov_pct = 100 * bridge_covered / bridge_total if bridge_total else 0
    answer_cov_pct = 100 * answer_covered / total if total else 0

    # --- Build report ---
    lines = []
    lines.append("# Narrative Evidence Graph Audit Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Section 1: Input summary
    lines.append("## 1. Input Summary")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Input file | {input_info.get('input_path', 'N/A')} |")
    lines.append(f"| Total Hard candidates available | {input_info.get('total_hard', 0)} |")
    lines.append(f"| Sampled for extraction | {total} |")
    lines.append(f"| Model | {input_info.get('model', 'N/A')} |")
    lines.append("")

    # Distribution by necessity_type
    nec_dist = Counter(g.get("necessity_type", "N/A") for g in graphs)
    lines.append("### Distribution by necessity_type")
    lines.append("")
    lines.append("| Type | Count |")
    lines.append("|---|---:|")
    for nt in sorted(nec_dist.keys()):
        lines.append(f"| {nt} | {nec_dist[nt]} |")
    lines.append("")

    # Distribution by reasoning_operation
    rop_dist = Counter(g.get("reasoning_operation", "N/A") for g in graphs)
    lines.append("### Distribution by reasoning_operation")
    lines.append("")
    lines.append("| Operation | Count |")
    lines.append("|---|---:|")
    for op in sorted(rop_dist.keys()):
        lines.append(f"| {op} | {rop_dist[op]} |")
    lines.append("")

    # Distribution by attribute
    attr_dist = Counter(g.get("attribute", "N/A") for g in graphs)
    lines.append("### Distribution by attribute")
    lines.append("")
    lines.append("| Attribute | Count |")
    lines.append("|---|---:|")
    for attr in sorted(attr_dist.keys()):
        lines.append(f"| {attr} | {attr_dist[attr]} |")
    lines.append("")

    # Section 2: Graph extraction success
    lines.append("## 2. Graph Extraction Success")
    lines.append("")
    lines.append("| Metric | Count | Pct |")
    lines.append("|---|---:|---:|")
    lines.append(f"| parse_ok | {parse_ok_count} | {100*parse_ok_count/total:.1f}% |")
    lines.append(f"| graph_valid | {valid_count} | {100*valid_count/total:.1f}% |")
    lines.append(f"| graph_invalid | {total - valid_count} | {100*(total-valid_count)/total:.1f}% |")
    lines.append("")

    if fail_reasons:
        lines.append("### Validation fail reasons")
        lines.append("")
        reason_counts = Counter(fail_reasons)
        lines.append("| Reason | Count |")
        lines.append("|---|---:|")
        for reason, cnt in reason_counts.most_common(10):
            # Truncate long reasons
            short = reason[:80] + "..." if len(reason) > 80 else reason
            lines.append(f"| {short} | {cnt} |")
        lines.append("")

    # Section 3: Graph structure statistics
    lines.append("## 3. Graph Structure Statistics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Avg nodes | {avg_nodes:.1f} |")
    lines.append(f"| Avg edges | {avg_edges:.1f} |")
    lines.append(f"| Min nodes | {min(node_counts) if node_counts else 0} |")
    lines.append(f"| Max nodes | {max(node_counts) if node_counts else 0} |")
    lines.append(f"| Min edges | {min(edge_counts) if edge_counts else 0} |")
    lines.append(f"| Max edges | {max(edge_counts) if edge_counts else 0} |")
    lines.append("")

    # Node type distribution
    lines.append("### Node type distribution")
    lines.append("")
    lines.append("| Type | Count | Pct |")
    lines.append("|---|---:|---:|")
    total_nodes = sum(node_types.values())
    for nt in sorted(node_types.keys(), key=lambda x: -node_types[x]):
        lines.append(f"| {nt} | {node_types[nt]} | {100*node_types[nt]/total_nodes:.1f}% |")
    lines.append("")

    # Edge relation distribution
    lines.append("### Edge relation distribution")
    lines.append("")
    lines.append("| Relation | Count | Pct |")
    lines.append("|---|---:|---:|")
    total_edges = sum(edge_relations.values())
    for rel in sorted(edge_relations.keys(), key=lambda x: -edge_relations[x]):
        lines.append(f"| {rel} | {edge_relations[rel]} | {100*edge_relations[rel]/total_edges:.1f}% |")
    lines.append("")

    # Evidence role distribution
    lines.append("### Evidence role distribution")
    lines.append("")
    lines.append("| Role | Count | Pct |")
    lines.append("|---|---:|---:|")
    for role in sorted(evidence_roles.keys(), key=lambda x: -evidence_roles[x]):
        lines.append(f"| {role} | {evidence_roles[role]} | {100*evidence_roles[role]/total_nodes:.1f}% |")
    lines.append("")

    # Edge necessity distribution
    lines.append("### Edge necessity distribution")
    lines.append("")
    lines.append("| Necessity | Count | Pct |")
    lines.append("|---|---:|---:|")
    for nec in sorted(edge_necessity.keys(), key=lambda x: -edge_necessity[x]):
        lines.append(f"| {nec} | {edge_necessity[nec]} | {100*edge_necessity[nec]/total_edges:.1f}% |")
    lines.append("")

    # Section 4: Coverage diagnostics
    lines.append("## 4. Coverage Diagnostics")
    lines.append("")
    lines.append("| Metric | Covered | Total | Pct |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Required evidence sentences | {req_covered} | {req_total} | {req_cov_pct:.1f}% |")
    lines.append(f"| Bridge sentences | {bridge_covered} | {bridge_total} | {bridge_cov_pct:.1f}% |")
    lines.append(f"| Answer node present | {answer_covered} | {total} | {answer_cov_pct:.1f}% |")
    lines.append("")

    # Section 5: Detailed graph examples (up to 5)
    lines.append("## 5. Detailed Graph Examples (up to 5)")
    lines.append("")

    # Pick examples: prefer valid graphs, diverse necessity types
    valid_graphs = [g for g in graphs if g.get("graph_valid")]
    if not valid_graphs:
        valid_graphs = graphs  # show invalid ones if no valid

    shown = set()
    examples = []
    # First pass: one per necessity type
    for g in valid_graphs:
        nt = g.get("necessity_type", "")
        if nt not in shown:
            examples.append(g)
            shown.add(nt)
        if len(examples) >= 5:
            break
    # Fill remaining
    for g in valid_graphs:
        if g not in examples:
            examples.append(g)
        if len(examples) >= 5:
            break

    for idx, g in enumerate(examples[:5], 1):
        lines.append(f"### Example {idx}")
        lines.append("")
        lines.append(f"**Story:** {g.get('story_name', 'N/A')}")
        lines.append(f"**Question:** {g.get('question', '')}")
        lines.append(f"**Answer:** {g.get('answer', '')}")
        lines.append(f"**Attribute:** {g.get('attribute', 'N/A')}")
        lines.append(f"**Reasoning:** {g.get('reasoning_operation', 'N/A')}")
        lines.append(f"**Necessity:** {g.get('necessity_type', 'N/A')}")
        lines.append(f"**Valid:** {g.get('graph_valid', False)}")
        lines.append(f"**Validation:** {g.get('graph_validation_reason', 'N/A')}")
        lines.append("")

        # Show required evidence sentences
        req_ids = g.get("required_evidence_sentences", [])
        section = ""
        # Try to get section from trace prompt (not stored directly)
        # We'll show sentence IDs only
        lines.append(f"**Required evidence sentences:** {req_ids}")
        lines.append(f"**Bridge sentences:** {g.get('bridge_sentence_ids', [])}")
        lines.append("")

        # Nodes table
        nodes = g.get("nodes", [])
        if nodes:
            lines.append("**Nodes:**")
            lines.append("")
            lines.append("| ID | Type | Sent | Role | Text | Participants |")
            lines.append("|---|---|---:|---|---|---|")
            for n in nodes:
                text = n.get("text", "")[:60]
                parts = ", ".join(n.get("participants", []))[:30]
                lines.append(
                    f"| {n.get('id', '?')} | {n.get('type', '?')} | "
                    f"{n.get('sentence_id', '?')} | {n.get('evidence_role', '?')} | "
                    f"{text} | {parts} |"
                )
            lines.append("")

        # Edges table
        edges = g.get("edges", [])
        if edges:
            lines.append("**Edges:**")
            lines.append("")
            lines.append("| Source | Target | Relation | Necessity | Reason |")
            lines.append("|---|---|---|---|---|")
            for e in edges:
                reason = e.get("reason", "")[:50]
                lines.append(
                    f"| {e.get('source', '?')} | {e.get('target', '?')} | "
                    f"{e.get('relation', '?')} | {e.get('necessity', '?')} | "
                    f"{reason} |"
                )
            lines.append("")

        lines.append("---")
        lines.append("")

    # Section 6: Recommendation
    lines.append("## 6. Recommendation")
    lines.append("")

    valid_pct = 100 * valid_count / total if total else 0
    parse_pct = 100 * parse_ok_count / total if total else 0

    if valid_pct >= 80 and parse_pct >= 95:
        lines.append("**Schema stability: PASS**")
        lines.append("")
        lines.append("The narrative evidence graph schema is stable enough for a QG pilot.")
        lines.append(f"- {valid_count}/{total} ({valid_pct:.0f}%) graphs are valid")
        lines.append(f"- {parse_ok_count}/{total} ({parse_pct:.0f}%) LLM responses parsed correctly")
        lines.append(f"- Average {avg_nodes:.1f} nodes and {avg_edges:.1f} edges per graph")
        lines.append("")

        # Most reliable types
        if node_types:
            top_node = max(node_types, key=node_types.get)
            lines.append(f"Most common node type: **{top_node}** ({node_types[top_node]})")
        if edge_relations:
            top_edge = max(edge_relations, key=edge_relations.get)
            lines.append(f"Most common edge relation: **{top_edge}** ({edge_relations[top_edge]})")
        lines.append("")

        lines.append("**Recommended next steps:**")
        lines.append("1. Use narrative graphs as QG scaffolding for Hard question generation")
        lines.append("2. Focus on motivation_bridge and causal_bridge (most reliable necessity types)")
        lines.append("3. Use causal_chain and motivation reasoning operations as primary targets")
    else:
        lines.append("**Schema stability: NEEDS WORK**")
        lines.append("")
        lines.append(f"- parse_ok: {parse_ok_count}/{total} ({parse_pct:.0f}%)")
        lines.append(f"- graph_valid: {valid_count}/{total} ({valid_pct:.0f}%)")
        lines.append("")
        if fail_reasons:
            top_fail = Counter(fail_reasons).most_common(3)
            lines.append("Top failure reasons:")
            for reason, cnt in top_fail:
                lines.append(f"- {reason}: {cnt}")
        lines.append("")
        lines.append("**Recommended fixes:**")
        lines.append("1. Address top failure reasons before proceeding to QG")
        lines.append("2. Consider relaxing validation constraints")
        lines.append("3. Improve prompt to reduce parse failures")

    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()

    output_dir = args.output_dir or _default_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    output_jsonl = os.path.join(output_dir, "graphs.jsonl")
    report_path = os.path.join(output_dir, "NARRATIVE_GRAPH_AUDIT.md")

    print(f"=== Narrative Evidence Graph Audit ===")
    print(f"Input:      {args.input}")
    print(f"Limit:      {args.limit}")
    print(f"Model:      {args.model or '(default JUDGE_MODEL)'}")
    print(f"Output dir: {output_dir}")
    print()

    # Load Hard candidates
    print("Loading Hard candidates...")
    all_hard = _load_hard_candidates(args.input)
    print(f"Found {len(all_hard)} Hard candidates "
          f"(necessity_type in {sorted(_TARGET_NEC_TYPES)}, no explicit_lookup)")

    if not all_hard:
        print("ERROR: No eligible Hard candidates found.")
        sys.exit(1)

    # Stratified sample
    sampled = _stratified_sample(all_hard, args.limit)
    print(f"Sampled {len(sampled)} for extraction")
    print()

    # Show sample distribution
    nec_dist = Counter(c.get("necessity_type", "N/A") for c in sampled)
    attr_dist = Counter(c.get("attribute", "N/A") for c in sampled)
    print("Sample distribution by necessity_type:")
    for nt, cnt in sorted(nec_dist.items()):
        print(f"  {nt}: {cnt}")
    print("Sample distribution by attribute:")
    for attr, cnt in sorted(attr_dist.items()):
        print(f"  {attr}: {cnt}")
    print()

    extractor = NarrativeGraphExtractor(model=args.model)

    graphs = []
    out_f = open(output_jsonl, "w", encoding="utf-8")

    try:
        for i, candidate in enumerate(sampled):
            t0 = time.time()
            print(f"  [{i+1}/{len(sampled)}] "
                  f"{candidate.get('story_name', '?')[:20]} "
                  f"nec={candidate.get('necessity_type', '?')[:15]} "
                  f"attr={candidate.get('attribute', '?')[:15]}... ",
                  end="", flush=True)

            try:
                record = extractor.extract(candidate)
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            elapsed = time.time() - t0
            valid = record.get("graph_valid", False)
            n_nodes = len(record.get("nodes", []))
            n_edges = len(record.get("edges", []))
            print(f"{'OK' if valid else 'INVALID'} "
                  f"nodes={n_nodes} edges={n_edges} {elapsed:.1f}s")

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            graphs.append(record)

            time.sleep(0.2)

    finally:
        out_f.close()

    print()
    print(f"=== Extraction Complete ===")
    print(f"Graphs extracted: {len(graphs)}")

    valid_count = sum(1 for g in graphs if g.get("graph_valid", False))
    parse_ok = sum(1 for g in graphs if g.get("trace", {}).get("parse_ok", False))
    print(f"parse_ok: {parse_ok}/{len(graphs)}")
    print(f"graph_valid: {valid_count}/{len(graphs)}")

    input_info = {
        "input_path": args.input,
        "total_hard": len(all_hard),
        "model": extractor.model,
    }
    _write_report(graphs, report_path, input_info)
    print(f"\nReport: {report_path}")
    print(f"Graphs: {output_jsonl}")


if __name__ == "__main__":
    main()
