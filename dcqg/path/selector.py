"""Answer-phrase validation and prefilter report generation.

Moved from path_prefilter.py: validate_answer_phrase, generate_prefilter_report,
_format_report_md.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

from dcqg.path.answer_extraction import _simple_stem


# ================================================================
# ANSWER PHRASE VALIDATION
# ================================================================

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


# ================================================================
# REPORT
# ================================================================

def generate_prefilter_report(items, report_json_path, report_md_path):
    """Generate prefilter report in JSON and Markdown."""
    n_total = len(items)
    passed = [r for r in items if r.get("prefilter_pass", False)]
    n_passed = len(passed)

    by_level = defaultdict(list)
    for r in items:
        by_level[r.get("difficulty", "Easy")].append(r)

    # -- Per-level pass rate --
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

    # -- Weak trigger distribution --
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

    # -- Answer phrase pass rate --
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

    # -- Relation group distribution by difficulty --
    rel_by_level = {}
    for level in ["Easy", "Medium", "Hard"]:
        level_items = by_level[level]
        rel_dist = Counter(r.get("relation_group", "NONE") for r in level_items)
        rel_by_level[level] = dict(rel_dist)

    # -- Temporal-only Hard ratio --
    hard_items = by_level.get("Hard", [])
    temporal_hard = sum(1 for r in hard_items if r.get("relation_group") == "TEMPORAL")
    temporal_hard_ratio = round(temporal_hard / len(hard_items), 4) if hard_items else 0

    # -- Single-sentence risk distribution --
    risk_dist = Counter(r.get("rule_single_sentence_risk", "low") for r in items)
    risk_by_level = {}
    for level in ["Easy", "Medium", "Hard"]:
        level_items = by_level[level]
        risk_by_level[level] = dict(Counter(r.get("rule_single_sentence_risk", "low") for r in level_items))

    # -- Failure reason distribution --
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

    # -- Build report --
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
                lines.append(f'- [{ex["difficulty"]}] trigger="{ex["trigger"]}" -> {ex["reason"]}')

    # Answer phrase
    lines.append("\n## Answer Phrase\n")
    lines.append(f"- Pass rate: {report['answer_phrase_pass_rate']*100:.1f}%")
    if report["answer_phrase_fail_examples"]:
        lines.append("\nFailures:\n")
        for ex in report["answer_phrase_fail_examples"]:
            lines.append(f'- [{ex["difficulty"]}] trigger="{ex["trigger"]}" phrase="{ex["phrase"]}" -> {ex["reason"]}')

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
                lines.append(f'- [{ex["difficulty"]}] trigger="{ex["trigger"]}" -> {ex["reason"]}')

    return "\n".join(lines)
