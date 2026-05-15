"""Target-label calibration audit for FairytaleQA candidates.

Runs the blind difficulty judge on original FairytaleQA questions to check
whether evidence_difficulty labels match the judge's perception.

Produces:
  - candidates.calibrated.jsonl — all candidates with judge fields
  - CANDIDATE_TARGET_CALIBRATION_REPORT.md — agreement matrix and diagnostics
"""

import json
import os
import sys
import time
import argparse
from collections import Counter, defaultdict

from dcqg.generation.fairytale_qg import difficulty_evidence_judge
from dcqg.utils.config import get_api_config

# ── Answer property checks (no API needed) ───────────────────────

SHORT_EMOTION_LABELS = {
    "happy", "sad", "scared", "upset", "angry", "afraid", "surprised",
    "pleased", "frightened", "worried", "excited", "disappointed",
    "ashamed", "proud", "jealous", "grateful", "miserable", "sorry",
    "glad", "terrified", "horrified", "delighted", "annoyed",
    "satisfied", "content", "uneasy", "unhappy", "furious",
    "dissatisfied", "intrigued", "concerned", "confused",
}

COREFERENCE_RISK_PATTERNS = [
    " he ", " she ", " it ", " they ", " him ", " her ", " them ",
    " his ", " her ", " its ", " their ", " this ", " that ", " those ",
    " the man ", " the woman ", " the boy ", " the girl ", " the king ",
    " the queen ", " the prince ", " the princess ",
]


def check_answer_properties(answer_text):
    """Check answer for short emotion, coreference risk, fragment status."""
    ans = (answer_text or "").strip().lower().rstrip(" .")
    tokens = ans.split()
    token_count = len(tokens)

    # Short emotion label
    is_short_emotion = ans in SHORT_EMOTION_LABELS

    # Coreference risk: answer relies on pronouns or generic references
    has_coref = any(pat in f" {ans} " for pat in COREFERENCE_RISK_PATTERNS)

    # Fragment: very short or doesn't form a complete thought
    is_fragment = token_count <= 2 or (token_count <= 4 and not any(
        w in ans for w in ["because", "since", "when", "after", "before",
                            "wanted", "decided", "needed", "thought"]
    ))

    # Self-contained: answer makes sense without looking at the story
    easy_self_contained = not has_coref and not is_fragment and token_count >= 2

    return {
        "answer_is_short_emotion": is_short_emotion,
        "answer_has_coreference_risk": has_coref,
        "answer_is_fragment": is_fragment,
        "easy_self_contained": easy_self_contained,
        "answer_token_count": token_count,
    }


# ── Calibration rules ────────────────────────────────────────────

def check_calibrated(candidate, judge_result):
    """Apply calibration rules per difficulty.

    Returns (is_calibrated: bool, reject_reason: str).
    """
    evidence_diff = candidate.get("evidence_difficulty", "")
    pred_diff = judge_result.get("predicted_difficulty", "judge_error")
    asa = judge_result.get("answer_sentence_alone_sufficient", "?")
    num_needed = judge_result.get("num_sentences_needed", 0)
    bridge = judge_result.get("bridge_required", "?")
    ans_props = candidate.get("_answer_props", {})

    if evidence_diff == "Easy":
        if pred_diff != "Easy":
            return False, f"judge_predicted={pred_diff} (expected Easy)"
        if asa != "yes":
            return False, f"asa={asa} (expected yes)"
        if isinstance(num_needed, (int, float)) and num_needed != 1:
            return False, f"num_sentences={num_needed} (expected 1)"
        if not ans_props.get("easy_self_contained", False):
            if ans_props.get("answer_is_short_emotion"):
                return False, "answer=short_emotion (not self-contained for Easy)"
            if ans_props.get("answer_has_coreference_risk"):
                return False, "answer=coreference_risk"
            if ans_props.get("answer_is_fragment"):
                return False, "answer=fragment"
            return False, "answer=not_self_contained"
        return True, "ok"

    elif evidence_diff == "Medium":
        if pred_diff != "Medium":
            return False, f"judge_predicted={pred_diff} (expected Medium)"
        if isinstance(num_needed, (int, float)) and num_needed != 2:
            return False, f"num_sentences={num_needed} (expected 2)"
        return True, "ok"

    elif evidence_diff == "Hard":
        if pred_diff != "Hard":
            return False, f"judge_predicted={pred_diff} (expected Hard)"
        if asa != "no":
            return False, f"asa={asa} (expected no)"
        if bridge != "yes":
            return False, f"bridge_required={bridge} (expected yes)"
        if isinstance(num_needed, (int, float)) and num_needed < 3:
            return False, f"num_sentences={num_needed} (expected >=3)"
        if ans_props.get("answer_is_short_emotion"):
            # Still reject short emotion answers for Hard even if judge says Hard
            return False, "answer=short_emotion (inappropriate for Hard)"
        return True, "ok"

    return False, f"unknown evidence_difficulty={evidence_diff}"


# ── Main audit ───────────────────────────────────────────────────

def run_calibration_audit(candidates_path, output_dir, limit=None,
                           start_idx=0, resume=False):
    """Run calibration audit on all candidates."""
    os.makedirs(output_dir, exist_ok=True)
    calibrated_path = os.path.join(output_dir, "candidates.calibrated.jsonl")
    resume_path = os.path.join(output_dir, "calibration_resume.json")

    # Load candidates
    with open(candidates_path, encoding="utf-8") as f:
        all_candidates = [json.loads(l) for l in f]

    if limit:
        all_candidates = all_candidates[:limit]

    print(f"Total candidates: {len(all_candidates)}")

    # Resume support
    done_count = 0
    if resume and os.path.exists(calibrated_path):
        with open(calibrated_path, encoding="utf-8") as f:
            done_count = sum(1 for _ in f)
        print(f"Resuming from {done_count}/{len(all_candidates)}")
        all_candidates = all_candidates[done_count:]

    # Check rate limit config
    cfg = get_api_config()
    print(f"Model: {cfg.get('JUDGE_MODEL', cfg.get('MODEL', 'unknown'))}")

    out_f = open(calibrated_path, "a", encoding="utf-8")
    last_save = time.time()

    for i, c in enumerate(all_candidates):
        idx = done_count + i

        # Answer property checks (no API)
        answer_text = c.get("answer", "") or c.get("answer1", "")
        ans_props = check_answer_properties(answer_text)
        c["_answer_props"] = ans_props

        # Run blind difficulty judge
        question = c.get("question", "")
        story_section = c.get("story_section", "")
        evidence_diff = c.get("evidence_difficulty", "")

        judge_result = difficulty_evidence_judge(
            question=question,
            story_section=story_section,
            target_answer=answer_text,
            difficulty=evidence_diff or "Medium",
            required_evidence_sentences=c.get("required_evidence_sentences", []),
            bridge_sentence_ids=c.get("bridge_sentence_ids", []),
        )

        # Build calibrated record
        cal = {
            "story_name": c.get("story_name", "?"),
            "story_section": story_section,
            "question": question,
            "answer": answer_text,
            "evidence_difficulty": evidence_diff,
            "judge_predicted_difficulty": judge_result.get("predicted_difficulty", "?"),
            "judge_num_sentences_needed": judge_result.get("num_sentences_needed", "?"),
            "judge_answer_sentence_alone_sufficient": judge_result.get(
                "answer_sentence_alone_sufficient", "?"),
            "judge_bridge_required": judge_result.get("bridge_required", "?"),
            "judge_reason": judge_result.get("reason", "")[:200],
            "judge_status": judge_result.get("difficulty_judge_status", "ok"),
            "agreement": evidence_diff == judge_result.get("predicted_difficulty", "?"),
        }
        cal.update(ans_props)

        # Apply calibration rules
        is_cal, reject_reason = check_calibrated(c, judge_result)
        cal["calibrated"] = is_cal
        cal["reject_reason"] = reject_reason

        # Write
        out_f.write(json.dumps(cal, ensure_ascii=False) + "\n")

        # Progress
        if (idx + 1) % 50 == 0 or (idx + 1) == len(all_candidates):
            agree_count = sum(1 for _ in range(idx + 1))
            elapsed = time.time() - last_save if last_save else 0
            print(f"  [{idx + 1}/{len(all_candidates)}] "
                  f"last batch {elapsed:.0f}s")
            last_save = time.time()
            out_f.flush()

        # Rate limit: 0.5s between calls
        time.sleep(0.5)

    out_f.close()
    print(f"\nCalibrated output: {calibrated_path}")
    return calibrated_path


# ── Report building ──────────────────────────────────────────────

def build_report(calibrated_path, output_dir):
    """Build calibration report."""
    with open(calibrated_path, encoding="utf-8") as f:
        calibrated = [json.loads(l) for l in f]

    report_path = os.path.join(output_dir, "CANDIDATE_TARGET_CALIBRATION_REPORT.md")

    levels = ["Easy", "Medium", "Hard"]
    total = len(calibrated)

    # 1. Agreement matrix
    matrix = defaultdict(lambda: defaultdict(int))
    for c in calibrated:
        ev = c.get("evidence_difficulty", "?")
        jp = c.get("judge_predicted_difficulty", "?")
        matrix[ev][jp] += 1

    # 2. Calibrated pool by difficulty
    cal_by_diff = Counter()
    for c in calibrated:
        if c.get("calibrated"):
            cal_by_diff[c.get("evidence_difficulty", "?")] += 1

    # 3. Rejection reasons
    reject_reasons = Counter()
    for c in calibrated:
        if not c.get("calibrated"):
            reject_reasons[c.get("reject_reason", "?")] += 1

    # 4. Easy mismatch examples
    easy_mismatch = [
        c for c in calibrated
        if c.get("evidence_difficulty") == "Easy"
        and not c.get("calibrated")
    ][:15]

    # 5. Hard mismatch examples
    hard_mismatch = [
        c for c in calibrated
        if c.get("evidence_difficulty") == "Hard"
        and not c.get("calibrated")
    ][:15]

    # 6. Story-matched pool
    stories_cal = defaultdict(set)
    for c in calibrated:
        if c.get("calibrated"):
            stories_cal[c.get("story_name", "?")].add(c.get("evidence_difficulty"))

    full_stories = [s for s, diffs in stories_cal.items()
                    if set(levels).issubset(diffs)]

    # 7. Judge error rate
    judge_errors = sum(1 for c in calibrated
                       if c.get("judge_status") != "ok"
                       or c.get("judge_predicted_difficulty") in ("judge_error", "?"))

    # Write report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# FairytaleQA Target-Label Calibration Report\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total candidates audited: {total}\n")
        f.write(f"Judge errors: {judge_errors}/{total} "
                f"({100*judge_errors/total:.1f}%)\n\n")

        # Section 1: Agreement matrix
        f.write("## 1. Agreement Matrix: evidence_difficulty x judge_predicted_difficulty\n\n")
        all_diffs = sorted(set(
            d for d in list(matrix.keys()) + ["Easy", "Medium", "Hard"] if d in matrix
        ))
        f.write("| Evidence \\ Judge | " + " | ".join(all_diffs) + " | Total |\n")
        f.write("|---|" + "|".join(["---:"] * (len(all_diffs) + 1)) + "|\n")
        for ev in levels:
            row = matrix.get(ev, {})
            cells = [str(row.get(jp, 0)) for jp in all_diffs]
            row_total = sum(row.values())
            f.write(f"| {ev} | " + " | ".join(cells) + f" | {row_total} |\n")
        f.write("\n")

        # Section 2: Calibrated pool
        f.write("## 2. Calibrated Pool Size by Difficulty\n\n")
        f.write("| Difficulty | Calibrated | Total | Pct |\n")
        f.write("|---|---:|---:|---:|\n")
        for d in levels:
            d_total = sum(1 for c in calibrated if c.get("evidence_difficulty") == d)
            d_cal = cal_by_diff.get(d, 0)
            pct = f"{100*d_cal/d_total:.1f}%" if d_total else "N/A"
            f.write(f"| {d} | {d_cal} | {d_total} | {pct} |\n")
        f.write(f"| **Total** | **{sum(cal_by_diff.values())}** | **{total}** | "
                f"**{100*sum(cal_by_diff.values())/total:.1f}%** |\n\n")

        # Section 3: Rejection reasons
        f.write("## 3. Rejection Reasons by Difficulty\n\n")
        for d in levels:
            d_rejects = Counter()
            for c in calibrated:
                if c.get("evidence_difficulty") == d and not c.get("calibrated"):
                    d_rejects[c.get("reject_reason", "?")] += 1
            if d_rejects:
                f.write(f"### {d}\n\n")
                f.write("| Reason | Count |\n")
                f.write("|---|---:|\n")
                for reason, count in d_rejects.most_common(20):
                    f.write(f"| {reason} | {count} |\n")
                f.write("\n")

        # Section 4: Easy mismatch examples
        f.write("## 4. Easy Mismatch Examples (evidence=Easy, not calibrated)\n\n")
        if easy_mismatch:
            f.write("| # | Story | Question | Answer | Judge Pred | ASA | Reason |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for i, c in enumerate(easy_mismatch, 1):
                f.write(f"| {i} | {c['story_name'][:25]} | {c['question'][:60]} | "
                        f"{c['answer'][:30]} | {c['judge_predicted_difficulty']} | "
                        f"{c['judge_answer_sentence_alone_sufficient']} | "
                        f"{c['reject_reason'][:50]} |\n")
        else:
            f.write("No Easy mismatches found.\n")
        f.write("\n")

        # Section 5: Hard mismatch examples
        f.write("## 5. Hard Mismatch Examples (evidence=Hard, not calibrated)\n\n")
        if hard_mismatch:
            f.write("| # | Story | Question | Answer | Judge Pred | ASA | Reason |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for i, c in enumerate(hard_mismatch, 1):
                f.write(f"| {i} | {c['story_name'][:25]} | {c['question'][:60]} | "
                        f"{c['answer'][:30]} | {c['judge_predicted_difficulty']} | "
                        f"{c['judge_answer_sentence_alone_sufficient']} | "
                        f"{c['reject_reason'][:50]} |\n")
        else:
            f.write("No Hard mismatches found.\n")
        f.write("\n")

        # Section 6: Story-matched pool
        f.write("## 6. Story-Matched Calibrated Pool\n\n")
        f.write(f"Stories with calibrated Easy+Medium+Hard: **{len(full_stories)}**\n\n")
        f.write(f"Target: >= 70 stories\n")
        f.write(f"Status: {'**PASS**' if len(full_stories) >= 70 else '**FAIL (not enough stories)**'}\n\n")

        # Section 7: Bottleneck analysis
        f.write("## 7. Bottleneck Analysis\n\n")
        f.write("| Difficulty | Total Candidates | Calibrated | Missing per story (avg) |\n")
        f.write("|---|---:|---:|---:|\n")
        total_stories = len(set(c["story_name"] for c in calibrated))
        for d in levels:
            d_total = sum(1 for c in calibrated if c.get("evidence_difficulty") == d)
            d_cal = cal_by_diff.get(d, 0)
            avg_per_story = d_total / total_stories if total_stories else 0
            cal_per_story = d_cal / total_stories if total_stories else 0
            missing = avg_per_story - cal_per_story
            f.write(f"| {d} | {d_total} | {d_cal} | {missing:.1f} |\n")
        f.write("\n")

        # Top reject reasons overall
        f.write("### Top Rejection Reasons (all difficulties)\n\n")
        f.write("| Reason | Count |\n")
        f.write("|---|---:|\n")
        for reason, count in reject_reasons.most_common(10):
            f.write(f"| {reason} | {count} |\n")

    print(f"Report: {report_path}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Target-label calibration audit for FairytaleQA candidates")
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of candidates to audit")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing calibrated output")
    parser.add_argument("--report_only", action="store_true",
                        help="Only build report from existing calibrated output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    calibrated_path = os.path.join(args.output_dir, "candidates.calibrated.jsonl")

    if not args.report_only:
        print("=== FairytaleQA Target-Label Calibration Audit ===")
        run_calibration_audit(
            args.candidates, args.output_dir,
            limit=args.limit, resume=args.resume,
        )

    if os.path.exists(calibrated_path):
        build_report(calibrated_path, args.output_dir)
    else:
        print("No calibrated output found, skipping report.")


if __name__ == "__main__":
    main()
