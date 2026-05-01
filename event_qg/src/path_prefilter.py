"""
Stage 1: Rule-based Path Prefilter.

Cleans event paths AFTER sampling but BEFORE question generation.
Reduces downstream generation and judge failures by filtering paths
with weak triggers, missing answer phrases, or unfavorable compositions.

No LLM calls — pure rules.

Usage:
    python event_qg/src/path_prefilter.py \
        --input event_qg/outputs/sampled_paths_preview.jsonl \
        --output event_qg/outputs/prefiltered_paths.jsonl \
        --report_json event_qg/outputs/path_prefilter_report.json \
        --report_md event_qg/outputs/path_prefilter_report.md
"""
import json
import os
import sys
import argparse
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent))

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in open(env_path):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from answer_extraction import (
    extract_answer_phrase_local,
    enrich_path_item,
    GENERIC_TRIGGERS,
    WEAK_TRIGGERS,
    _simple_stem,
)
from quality_filter import (
    HARD_BLACKLIST_TRIGGERS,
    WEAK_TRIGGER_NEEDS_PHRASE,
    check_weak_trigger,
)


# ═══════════════════════════════════════════════════════════════
# COMBINED WEAK TRIGGER SETS
# ═══════════════════════════════════════════════════════════════

# Hard blacklist: always fail unless valid answer phrase
HARD_WEAK_TRIGGERS = HARD_BLACKLIST_TRIGGERS | GENERIC_TRIGGERS | {
    "become", "began", "ended", "continued", "included",
    "showed", "found", "called", "seen",
}

# Soft weak triggers: flag but allow if answer phrase is good
SOFT_WEAK_TRIGGERS = WEAK_TRIGGERS | WEAK_TRIGGER_NEEDS_PHRASE


# ═══════════════════════════════════════════════════════════════
# RELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════

def classify_relations(relation_subtypes):
    """Classify relation composition.
    Returns (non_temporal_count, relation_group).
    """
    if not relation_subtypes:
        return 0, "NONE"

    non_temporal = 0
    has_cause = False
    has_subevent = False

    for r in relation_subtypes:
        r_upper = r.upper()
        if not r_upper.startswith("TEMPORAL"):
            non_temporal += 1
        if "CAUSE" in r_upper:
            has_cause = True
        if "SUBEVENT" in r_upper:
            has_subevent = True

    if has_cause and has_subevent:
        group = "MIXED"
    elif has_cause:
        group = "CAUSE"
    elif has_subevent:
        group = "SUBEVENT"
    elif non_temporal > 0:
        group = "MIXED"
    else:
        group = "TEMPORAL"

    return non_temporal, group


# ═══════════════════════════════════════════════════════════════
# SUPPORT SPAN ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_support_span(events, supporting_sentences):
    """Compute support span and single-sentence risk.
    Returns (support_span, single_sentence_risk).
    """
    support_span = len(supporting_sentences) if supporting_sentences else 0

    # Check if all path events are in the same sentence
    sent_ids = set()
    for e in events:
        sid = e.get("sent_id", -1)
        if sid >= 0:
            sent_ids.add(sid)

    all_same_sentence = len(sent_ids) <= 1 and len(events) > 1

    # Determine risk
    if support_span <= 2:
        risk = "high"
    elif all_same_sentence:
        risk = "high"
    elif support_span <= 3:
        risk = "medium"
    else:
        risk = "low"

    return support_span, risk


# ═══════════════════════════════════════════════════════════════
# ANSWER PHRASE VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate_answer_phrase(phrase, trigger, answer_phrase_status=None):
    """Check if extracted answer phrase is good enough.
    Returns (pass: bool, reason: str).
    """
    if not phrase:
        return False, "empty phrase"

    # Reject partial extractions (used full sentence without clause boundaries)
    if answer_phrase_status == "partial":
        return False, "partial extraction (no clause boundary found)"

    phrase_lower = phrase.lower().strip()
    trigger_lower = trigger.lower().strip()

    # Phrase is just the trigger
    if phrase_lower == trigger_lower:
        return False, "phrase equals trigger"

    # Phrase too short (less than 2 words)
    if len(phrase.split()) < 2:
        return False, f"phrase too short: '{phrase}'"

    # Phrase doesn't contain trigger
    if trigger_lower not in phrase_lower:
        # Check stem match
        trigger_stem = _simple_stem(trigger_lower)
        phrase_stems = {_simple_stem(w) for w in phrase_lower.split()}
        if trigger_stem not in phrase_stems:
            return False, f"trigger '{trigger}' not in phrase '{phrase}'"

    return True, "valid phrase"


# ═══════════════════════════════════════════════════════════════
# MAIN PREFILTER
# ═══════════════════════════════════════════════════════════════

def prefilter_path(item):
    """Apply all prefilter rules to a single path item.
    Returns the item with prefilter fields added.
    """
    item = dict(item)  # shallow copy
    events = item.get("events", [])
    difficulty = item.get("difficulty", "Easy")
    supporting = item.get("supporting_sentences", [])
    relation_subtypes = item.get("relation_subtypes", [])

    # Enrich with answer phrase data
    item = enrich_path_item(item)

    final_event = events[-1] if events else {}
    trigger = final_event.get("trigger", "")
    answer_phrase = item.get("gold_answer_phrase", "")
    answer_sentence = item.get("gold_answer_sentence", "")

    # ── 1. Answer phrase extraction & validation ──
    answer_phrase_status = item.get("answer_phrase_status", "unknown")
    ap_pass, ap_reason = validate_answer_phrase(answer_phrase, trigger, answer_phrase_status)
    item["answer_phrase_pass"] = ap_pass
    item["answer_phrase_reason"] = ap_reason

    # ── 2. Weak trigger check ──
    trigger_lower = trigger.lower().strip()
    wt_result = check_weak_trigger(trigger, answer_phrase)
    item["weak_trigger_type"] = wt_result["weak_trigger_type"]
    item["weak_trigger_pass"] = wt_result["weak_trigger_pass"]
    item["weak_trigger_reason"] = wt_result["weak_trigger_reason"]

    # ── 3. Relation composition ──
    non_temporal_count, relation_group = classify_relations(relation_subtypes)
    item["non_temporal_count"] = non_temporal_count
    item["relation_group"] = relation_group

    # ── 4. Support span ──
    support_span, single_sentence_risk = analyze_support_span(events, supporting)
    item["support_span"] = support_span
    item["rule_single_sentence_risk"] = single_sentence_risk

    # ── 5. Pass / fail logic ──
    reasons = []

    # Must-fail: hard blacklisted trigger (always, regardless of phrase)
    if trigger_lower in HARD_WEAK_TRIGGERS:
        reasons.append(f"hard_weak_trigger='{trigger}'")

    # Must-fail: answer phrase not extractable
    if not ap_pass:
        reasons.append(f"answer_phrase_fail: {ap_reason}")

    # Must-fail: soft weak trigger without valid phrase
    if trigger_lower in SOFT_WEAK_TRIGGERS and not ap_pass:
        reasons.append(f"soft_weak_trigger='{trigger}' with no valid phrase")

    # High risk markers (not fail, just flag)
    risk_flags = []
    if difficulty == "Hard" and relation_group == "TEMPORAL":
        risk_flags.append("temporal_only_hard")
    if difficulty in ("Medium", "Hard") and single_sentence_risk == "high":
        risk_flags.append("single_sentence_risk_high")

    prefilter_pass = len(reasons) == 0
    prefilter_reason = "; ".join(reasons) if reasons else "pass"
    if risk_flags:
        prefilter_reason += f" [risk: {', '.join(risk_flags)}]"

    item["prefilter_pass"] = prefilter_pass
    item["prefilter_reason"] = prefilter_reason

    return item


# ═══════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════

def generate_prefilter_report(items, report_json_path, report_md_path):
    """Generate prefilter report in JSON and Markdown."""
    n_total = len(items)
    passed = [r for r in items if r.get("prefilter_pass", False)]
    n_passed = len(passed)

    by_level = defaultdict(list)
    for r in items:
        by_level[r.get("difficulty", "Easy")].append(r)

    # ── Per-level pass rate ──
    level_stats = {}
    for level in ["Easy", "Medium", "Hard"]:
        level_items = by_level[level]
        n = len(level_items)
        p = sum(1 for r in level_items if r.get("prefilter_pass", False))
        level_stats[level] = {
            "total": n,
            "passed": p,
            "pass_rate": round(p / n, 4) if n else 0,
        }

    # ── Weak trigger distribution ──
    wt_dist = Counter()
    wt_fail_examples = defaultdict(list)
    for r in items:
        wt_type = r.get("weak_trigger_type", "none")
        wt_dist[wt_type] += 1
        if wt_type != "none" and len(wt_fail_examples[wt_type]) < 5:
            wt_fail_examples[wt_type].append({
                "trigger": r.get("events", [{}])[-1].get("trigger", ""),
                "difficulty": r.get("difficulty", ""),
                "doc_id": r.get("doc_id", ""),
                "reason": r.get("weak_trigger_reason", ""),
            })

    # ── Answer phrase pass rate ──
    ap_pass = sum(1 for r in items if r.get("answer_phrase_pass", False))
    ap_rate = round(ap_pass / n_total, 4) if n_total else 0
    ap_fail_examples = []
    for r in items:
        if not r.get("answer_phrase_pass", False) and len(ap_fail_examples) < 5:
            ap_fail_examples.append({
                "trigger": r.get("events", [{}])[-1].get("trigger", ""),
                "difficulty": r.get("difficulty", ""),
                "doc_id": r.get("doc_id", ""),
                "phrase": r.get("gold_answer_phrase", ""),
                "reason": r.get("answer_phrase_reason", ""),
            })

    # ── Relation group distribution by difficulty ──
    rel_by_level = {}
    for level in ["Easy", "Medium", "Hard"]:
        level_items = by_level[level]
        rel_dist = Counter(r.get("relation_group", "NONE") for r in level_items)
        rel_by_level[level] = dict(rel_dist)

    # ── Temporal-only Hard ratio ──
    hard_items = by_level.get("Hard", [])
    temporal_hard = sum(1 for r in hard_items if r.get("relation_group") == "TEMPORAL")
    temporal_hard_ratio = round(temporal_hard / len(hard_items), 4) if hard_items else 0

    # ── Single-sentence risk distribution ──
    risk_dist = Counter(r.get("rule_single_sentence_risk", "low") for r in items)
    risk_by_level = {}
    for level in ["Easy", "Medium", "Hard"]:
        level_items = by_level[level]
        risk_by_level[level] = dict(Counter(r.get("rule_single_sentence_risk", "low") for r in level_items))

    # ── Failure reason distribution ──
    fail_reasons = Counter()
    fail_examples = defaultdict(list)
    for r in items:
        if not r.get("prefilter_pass", False):
            reason = r.get("prefilter_reason", "unknown")
            # Strip risk flags for categorization
            clean_reason = reason.split(" [risk:")[0].strip() if " [risk:" in reason else reason
            fail_reasons[clean_reason] += 1
            if len(fail_examples[clean_reason]) < 5:
                events = r.get("events", [])
                final = events[-1] if events else {}
                fail_examples[clean_reason].append({
                    "trigger": final.get("trigger", ""),
                    "difficulty": r.get("difficulty", ""),
                    "doc_id": r.get("doc_id", ""),
                    "reason": reason,
                })

    # ── Build report ──
    report = {
        "n_total": n_total,
        "n_passed": n_passed,
        "pass_rate": round(n_passed / n_total, 4) if n_total else 0,
        "per_level": level_stats,
        "weak_trigger_distribution": dict(wt_dist),
        "weak_trigger_fail_examples": {k: v for k, v in wt_fail_examples.items()},
        "answer_phrase_pass_rate": ap_rate,
        "answer_phrase_fail_examples": ap_fail_examples,
        "relation_group_by_level": rel_by_level,
        "temporal_only_hard_count": temporal_hard,
        "temporal_only_hard_ratio": temporal_hard_ratio,
        "single_sentence_risk_distribution": dict(risk_dist),
        "single_sentence_risk_by_level": risk_by_level,
        "fail_reason_distribution": dict(fail_reasons),
        "fail_examples": {k: v for k, v in fail_examples.items()},
    }

    # Save JSON
    report_json_path = Path(report_json_path)
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Save MD
    md = _format_report_md(report)
    report_md_path = Path(report_md_path)
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md)

    return report


def _format_report_md(report):
    """Format report as markdown."""
    lines = []
    lines.append("# Path Prefilter Report\n")
    lines.append(f"**Total paths:** {report['n_total']}")
    lines.append(f"**Passed:** {report['n_passed']} ({report['pass_rate']*100:.1f}%)\n")

    # Per-level
    lines.append("## Per-Level Pass Rate\n")
    lines.append("| Level | Total | Passed | Pass Rate |")
    lines.append("|-------|-------|--------|-----------|")
    for level in ["Easy", "Medium", "Hard"]:
        s = report["per_level"][level]
        lines.append(f"| {level} | {s['total']} | {s['passed']} | {s['pass_rate']*100:.1f}% |")

    # Weak trigger
    lines.append("\n## Weak Trigger Distribution\n")
    lines.append("| Type | Count |")
    lines.append("|------|-------|")
    for t, c in sorted(report["weak_trigger_distribution"].items(), key=lambda x: -x[1]):
        lines.append(f"| {t} | {c} |")
    if report["weak_trigger_fail_examples"]:
        lines.append("\nExamples:\n")
        for wt_type, examples in report["weak_trigger_fail_examples"].items():
            for ex in examples[:3]:
                lines.append(f'- [{ex["difficulty"]}] trigger="{ex["trigger"]}" → {ex["reason"]}')

    # Answer phrase
    lines.append(f"\n## Answer Phrase\n")
    lines.append(f"- Pass rate: {report['answer_phrase_pass_rate']*100:.1f}%")
    if report["answer_phrase_fail_examples"]:
        lines.append("\nFailures:\n")
        for ex in report["answer_phrase_fail_examples"]:
            lines.append(f'- [{ex["difficulty"]}] trigger="{ex["trigger"]}" phrase="{ex["phrase"]}" → {ex["reason"]}')

    # Relation group
    lines.append("\n## Relation Group by Difficulty\n")
    lines.append("| Level | TEMPORAL | CAUSE | SUBEVENT | MIXED | NONE |")
    lines.append("|-------|----------|-------|----------|-------|------|")
    for level in ["Easy", "Medium", "Hard"]:
        dist = report["relation_group_by_level"].get(level, {})
        lines.append(f"| {level} | {dist.get('TEMPORAL', 0)} | {dist.get('CAUSE', 0)} | {dist.get('SUBEVENT', 0)} | {dist.get('MIXED', 0)} | {dist.get('NONE', 0)} |")
    lines.append(f"\n- Temporal-only Hard: {report['temporal_only_hard_count']} ({report['temporal_only_hard_ratio']*100:.1f}%)")

    # Single-sentence risk
    lines.append("\n## Single-Sentence Risk\n")
    lines.append("| Level | Low | Medium | High |")
    lines.append("|-------|-----|--------|------|")
    for level in ["Easy", "Medium", "Hard"]:
        dist = report["single_sentence_risk_by_level"].get(level, {})
        lines.append(f"| {level} | {dist.get('low', 0)} | {dist.get('medium', 0)} | {dist.get('high', 0)} |")

    # Fail reasons
    lines.append("\n## Failure Reasons\n")
    lines.append("| Reason | Count |")
    lines.append("|--------|-------|")
    for reason, count in sorted(report["fail_reason_distribution"].items(), key=lambda x: -x[1]):
        lines.append(f"| {reason} | {count} |")
    if report["fail_examples"]:
        lines.append("\nExamples:\n")
        for reason, examples in report["fail_examples"].items():
            for ex in examples[:3]:
                lines.append(f'- [{ex["difficulty"]}] trigger="{ex["trigger"]}" → {ex["reason"]}')

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Rule-based path prefilter")
    parser.add_argument("--input", type=str, default="event_qg/outputs/sampled_paths_preview.jsonl",
                        help="Input JSONL with sampled paths")
    parser.add_argument("--output", type=str, default="event_qg/outputs/prefiltered_paths.jsonl",
                        help="Output JSONL with prefiltered paths")
    parser.add_argument("--report_json", type=str, default="event_qg/outputs/path_prefilter_report.json",
                        help="Output JSON report")
    parser.add_argument("--report_md", type=str, default="event_qg/outputs/path_prefilter_report.md",
                        help="Output Markdown report")
    args = parser.parse_args()

    # Load
    print(f"Loading paths from {args.input}...")
    with open(args.input, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    print(f"  Loaded {len(items)} paths")

    # Apply prefilter
    print("Applying prefilter rules...")
    results = []
    for i, item in enumerate(items):
        r = prefilter_path(item)
        results.append(r)

    n_passed = sum(1 for r in results if r.get("prefilter_pass", False))
    print(f"  Passed: {n_passed}/{len(results)} ({n_passed/len(results)*100:.1f}%)")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved to {output_path}")

    # Generate report
    print("Generating report...")
    report = generate_prefilter_report(results, args.report_json, args.report_md)
    print(f"  JSON report: {args.report_json}")
    print(f"  MD report: {args.report_md}")

    # Print summary
    print(f"\n{'='*60}")
    print("PREFILTER SUMMARY")
    print("=" * 60)
    print(f"  Total: {report['n_total']}")
    print(f"  Passed: {report['n_passed']} ({report['pass_rate']*100:.1f}%)")
    for level in ["Easy", "Medium", "Hard"]:
        s = report["per_level"][level]
        print(f"  {level}: {s['passed']}/{s['total']} ({s['pass_rate']*100:.1f}%)")
    print(f"  Temporal-only Hard: {report['temporal_only_hard_count']} ({report['temporal_only_hard_ratio']*100:.1f}%)")
    for level in ["Easy", "Medium", "Hard"]:
        dist = report["single_sentence_risk_by_level"].get(level, {})
        high = dist.get("high", 0)
        total = sum(dist.values())
        if total:
            print(f"  {level} single-sentence risk high: {high}/{total}")


if __name__ == "__main__":
    main()
