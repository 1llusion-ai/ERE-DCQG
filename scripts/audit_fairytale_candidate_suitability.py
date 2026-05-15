"""FairytaleQA Candidate Suitability Audit (Stage 3).

Pre-generation candidate filtering based on evidence audit properties only.
Does NOT use generated questions, predicted difficulty, or judge results.

Outputs:
  - candidates.suitability.jsonl: all candidates with suitability flags
  - selected_story_matched_suitable.jsonl: story-matched selection from suitable pool
  - CANDIDATE_SUITABILITY_REPORT.md: full audit report
"""
import argparse
import json
import random
import time
from pathlib import Path
from collections import Counter


# ── Emotion/state labels that make Hard chain questions collapse ──
SHORT_EMOTION_LABELS = {
    "happy", "sad", "scared", "upset", "angry", "afraid", "surprised",
    "pleased", "frightened", "worried", "excited", "disappointed",
    "ashamed", "proud", "jealous", "grateful", "miserable", "sorry",
    "glad", "terrified", "horrified", "delighted", "annoyed",
    "satisfied", "content", "uneasy", "unhappy", "furious",
    "dissatisfied", "intrigued", "concerned", "confused",
}

# Weak necessity types that don't support true multi-step reasoning
WEAK_NECESSITY_TYPES = {
    "background_context",   # just provides setting, not causal chain
    "temporal_before",      # only ordering, not logical dependency
    "answer_identification", # often just picking the right entity
}

# Strong necessity types for Hard questions
STRONG_NECESSITY_TYPES = {
    "motivation_bridge",
    "causal_bridge",
    "summary_synthesis",
    "disambiguation",
}


def classify_answer_type(answer_text):
    """Heuristic classification of answer text."""
    if not answer_text:
        return "empty"
    ans = answer_text.strip().lower().rstrip(" .")
    tokens = ans.split()
    n_tokens = len(tokens)

    # Emotion/state label
    if ans in SHORT_EMOTION_LABELS:
        return "emotion_label"

    # Very short (1-2 tokens)
    if n_tokens <= 2:
        return "short_label"

    # Short phrase (3-5 tokens)
    if n_tokens <= 5:
        return "short_phrase"

    # Explanatory/longer
    return "explanatory"


def assess_candidate_suitability(candidate):
    """Apply pre-generation suitability rules to a single candidate.

    Returns dict with suitability flags. Does NOT use generation/judge outputs.
    """
    diff = candidate.get("evidence_difficulty", "")
    asa = candidate.get("answer_sentence_alone_sufficient", "")
    num_req = candidate.get("num_required_sentences", 0)
    nt = candidate.get("necessity_type", "")
    bridge_ids = candidate.get("bridge_sentence_ids", []) or []
    req_sents = candidate.get("required_evidence_sentences", []) or []
    answer = (candidate.get("answer", "") or "").strip()
    ans_type = classify_answer_type(answer)
    ans_tokens = len(answer.split()) if answer else 0
    story = candidate.get("story_name", "")

    result = {
        "story_name": story,
        "evidence_difficulty": diff,
        "answer": answer[:80],
        "answer_type": ans_type,
        "answer_token_count": ans_tokens,
        "asa": asa,
        "num_required_sentences": num_req,
        "necessity_type": nt,
        "bridge_sentence_count": len(bridge_ids),
        "required_sentence_count": len(req_sents),
        "suitable": False,
        "reject_reason": "",
    }

    if diff == "Easy":
        # Must be truly single-sentence answerable.
        # ASA=yes is the primary signal (answer sentence alone suffices).
        # Bridge sentences may be present in annotation but ASA=yes confirms
        # they are not strictly required.
        if asa != "yes":
            result["reject_reason"] = f"ASA={asa} (not yes)"
            return result
        if isinstance(num_req, (int, float)) and num_req > 1:
            result["reject_reason"] = f"num_required_sentences={num_req} (>1)"
            return result
        result["suitable"] = True
        return result

    elif diff == "Medium":
        # Must require exactly 2 sentences
        if isinstance(num_req, (int, float)):
            if num_req < 1:
                result["reject_reason"] = f"num_required_sentences={num_req} (<1)"
                return result
            if num_req > 2:
                result["reject_reason"] = f"num_required_sentences={num_req} (>2)"
                return result
        if asa == "yes":
            result["reject_reason"] = "ASA=yes (should require >1 sentence)"
            return result
        if not bridge_ids or len(bridge_ids) == 0:
            result["reject_reason"] = "no bridge sentences (Medium needs at least 1)"
            return result
        result["suitable"] = True
        return result

    elif diff == "Hard":
        # Must require 3+ sentences, be truly multi-step
        if isinstance(num_req, (int, float)) and num_req < 3:
            result["reject_reason"] = f"num_required_sentences={num_req} (<3)"
            return result
        if asa != "no":
            result["reject_reason"] = f"ASA={asa} (should be no for Hard)"
            return result
        # Exclude weak necessity types
        if nt in WEAK_NECESSITY_TYPES:
            result["reject_reason"] = f"necessity_type={nt} (weak)"
            return result
        # Exclude short emotion/state labels
        if ans_type == "emotion_label":
            result["reject_reason"] = f"answer_type=emotion_label ({answer[:30]})"
            return result
        if ans_type == "short_label":
            result["reject_reason"] = f"answer_type=short_label ({answer[:30]})"
            return result
        result["suitable"] = True
        return result

    else:
        result["reject_reason"] = f"unknown difficulty: {diff}"
        return result


def select_story_matched_suitable(suitable_candidates, per_level_per_story=1, max_stories=None, seed=42):
    """Select story-matched candidates from the suitable pool.

    Each story contributes exactly per_level_per_story Easy + Medium + Hard.
    """
    rng = random.Random(seed)

    # Group suitable candidates by (story, difficulty)
    story_levels = {}
    for c in suitable_candidates:
        diff = c.get("evidence_difficulty", "")
        if diff not in ("Easy", "Medium", "Hard"):
            continue
        sn = c.get("story_name", "")
        if sn not in story_levels:
            story_levels[sn] = {"Easy": [], "Medium": [], "Hard": []}
        story_levels[sn][diff].append(c)

    # Filter to stories with at least per_level_per_story suitable per level
    eligible = {}
    for sn, levels in story_levels.items():
        if (len(levels["Easy"]) >= per_level_per_story
                and len(levels["Medium"]) >= per_level_per_story
                and len(levels["Hard"]) >= per_level_per_story):
            eligible[sn] = levels

    eligible_names = sorted(eligible.keys())
    if max_stories and max_stories < len(eligible_names):
        eligible_names = rng.sample(eligible_names, max_stories)
        eligible_names.sort()

    def _easy_preference(c):
        score = 0
        if c.get("answer_sentence_alone_sufficient") == "yes":
            score += 10
        if c.get("num_required_sentences", 99) == 1:
            score += 5
        ans_type = classify_answer_type(c.get("answer", ""))
        if ans_type in ("explanatory", "short_phrase"):
            score += 3
        return score

    def _medium_preference(c):
        score = 0
        if c.get("num_required_sentences", 0) == 2:
            score += 10
        ans_len = len((c.get("answer", "") or "").strip().split())
        if ans_len >= 3:
            score += 3
        return score

    def _hard_preference(c):
        score = 0
        nt = c.get("necessity_type", "")
        if nt in ("motivation_bridge", "causal_bridge"):
            score += 10
        elif nt in ("disambiguation", "summary_synthesis"):
            score += 8
        ans_len = len((c.get("answer", "") or "").strip().split())
        if ans_len >= 5:
            score += 5
        elif ans_len >= 3:
            score += 3
        return score

    selected = []
    for idx, sn in enumerate(eligible_names):
        levels = eligible[sn]
        for diff, preference_fn in [
            ("Easy", _easy_preference),
            ("Medium", _medium_preference),
            ("Hard", _hard_preference),
        ]:
            pool = levels[diff]
            scored = [(preference_fn(c), i, c) for i, c in enumerate(pool)]
            scored.sort(key=lambda x: (-x[0], x[1]))
            chosen = [c for _, _, c in scored[:per_level_per_story]]
            for c in chosen:
                c_out = dict(c)
                c_out["target_difficulty"] = diff
                c_out["story_group_id"] = idx
                selected.append(c_out)

    return selected, len(eligible_names)


def build_report(all_candidates, suitable_pool, selected, output_dir):
    """Build the candidate suitability report."""
    report_path = output_dir / "CANDIDATE_SUITABILITY_REPORT.md"
    levels = ["Easy", "Medium", "Hard"]

    # Pool counts before/after by difficulty
    before = {}
    after = {}
    for d in levels:
        before[d] = [c for c in all_candidates if c.get("evidence_difficulty") == d]
        after[d] = [c for c in suitable_pool if c.get("evidence_difficulty") == d]

    # Rejection reasons by difficulty
    reject_reasons = {}
    for d in levels:
        reject_reasons[d] = Counter()
        for c in all_candidates:
            if c.get("evidence_difficulty") == d:
                rr = c.get("reject_reason", "")
                if rr:
                    reject_reasons[d][rr] += 1

    # Story coverage
    stories_before = set(c.get("story_name", "") for c in all_candidates if c.get("evidence_difficulty") in levels)
    stories_after = set(c.get("story_name", "") for c in suitable_pool if c.get("evidence_difficulty") in levels)
    # Story-matched eligibility: need >=1 per level
    sl_before = {}
    for c in all_candidates:
        sn = c.get("story_name", "")
        d = c.get("evidence_difficulty", "")
        if d in levels:
            sl_before.setdefault(sn, {"Easy": 0, "Medium": 0, "Hard": 0})[d] += 1
    eligible_before = sum(1 for sn, cnts in sl_before.items()
                          if cnts["Easy"] >= 1 and cnts["Medium"] >= 1 and cnts["Hard"] >= 1)

    sl_after = {}
    for c in suitable_pool:
        sn = c.get("story_name", "")
        d = c.get("evidence_difficulty", "")
        if d in levels:
            sl_after.setdefault(sn, {"Easy": 0, "Medium": 0, "Hard": 0})[d] += 1
    eligible_after = sum(1 for sn, cnts in sl_after.items()
                         if cnts["Easy"] >= 1 and cnts["Medium"] >= 1 and cnts["Hard"] >= 1)

    # Easy mismatch audit
    easy_mismatches = [c for c in all_candidates
                       if c.get("evidence_difficulty") == "Easy" and not c.get("suitable")]

    # Hard mismatch audit
    hard_mismatches = [c for c in all_candidates
                       if c.get("evidence_difficulty") == "Hard" and not c.get("suitable")]

    # Hard emotion-label mismatches
    hard_emotion = [c for c in hard_mismatches
                    if c.get("answer_type") == "emotion_label"]

    # Hard ASA != no
    hard_asa_not_no = [c for c in all_candidates
                       if c.get("evidence_difficulty") == "Hard"
                       and c.get("asa") != "no"]

    lines = [
        "# FairytaleQA Candidate Suitability Audit Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Pool Counts Before/After by Difficulty",
        "",
        "| Difficulty | Before | After | Removed | Retention |",
        "|---|---:|---:|---:|---:|",
    ]
    for d in levels:
        nb = len(before[d])
        na = len(after[d])
        removed = nb - na
        pct = f"{100*na/nb:.1f}%" if nb else "N/A"
        lines.append(f"| {d} | {nb} | {na} | {removed} | {pct} |")
    lines.append("")

    # 2. Rejection reasons
    lines.append("## 2. Rejection Reasons by Difficulty")
    lines.append("")
    for d in levels:
        lines.append(f"### {d}")
        lines.append("")
        rr = reject_reasons[d]
        if rr:
            lines.append("| Reason | Count |")
            lines.append("|---|---:|")
            for reason, count in rr.most_common(20):
                lines.append(f"| {reason} | {count} |")
        else:
            lines.append("No rejections.")
        lines.append("")

    # 3. Story coverage
    lines.append("## 3. Story Coverage Before/After")
    lines.append("")
    lines.append("| Metric | Before | After |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Unique stories | {len(stories_before)} | {len(stories_after)} |")
    lines.append(f"| Story-matched eligible (>=1 per level) | {eligible_before} | {eligible_after} |")
    lines.append("")

    # 4. Easy mismatch audit
    lines.append("## 4. Easy Mismatch Audit")
    lines.append("")
    lines.append(f"Easy candidates rejected: {len(easy_mismatches)}")
    lines.append("")
    if easy_mismatches:
        lines.append("| # | Story | Answer | ASA | num_req | Reject Reason |")
        lines.append("|---|---|---|---|---|---|")
        for i, c in enumerate(easy_mismatches[:20], 1):
            lines.append(f"| {i} | {c['story_name'][:35]} | {c['answer'][:40]} | "
                        f"{c['asa']} | {c['num_required_sentences']} | {c['reject_reason']} |")
    lines.append("")

    # 5. Hard mismatch audit
    lines.append("## 5. Hard Mismatch Audit")
    lines.append("")
    lines.append(f"Hard candidates rejected: {len(hard_mismatches)}")
    lines.append("")

    lines.append("### 5a. Hard: Short Emotion/State Answers")
    lines.append("")
    lines.append(f"Count: {len(hard_emotion)}")
    lines.append("")
    if hard_emotion:
        lines.append("| # | Story | Answer | Necessity Type |")
        lines.append("|---|---|---|---|")
        for i, c in enumerate(hard_emotion[:20], 1):
            lines.append(f"| {i} | {c['story_name'][:35]} | {c['answer'][:30]} | {c['necessity_type']} |")
    lines.append("")

    lines.append("### 5b. Hard: All Rejection Reasons")
    lines.append("")
    hrr = reject_reasons.get("Hard", Counter())
    if hrr:
        lines.append("| Reason | Count |")
        lines.append("|---|---:|")
        for reason, count in hrr.most_common(20):
            lines.append(f"| {reason} | {count} |")
    lines.append("")

    lines.append("### 5c. Hard: ASA != no (from original audit)")
    lines.append("")
    lines.append(f"Count: {len(hard_asa_not_no)}")
    lines.append("")

    # 6. Final story-matched suitable pool size
    lines.append("## 6. Final Story-Matched Suitable Pool")
    lines.append("")
    lines.append(f"| Suitable stories (>=1 per level) | {eligible_after} |")
    n_selected = len(selected)
    sel_per_level = Counter(c.get("target_difficulty", "?") for c in selected)
    lines.append(f"| Selected (1 per level per story) | {n_selected} ({sel_per_level.get('Easy', 0)}E/{sel_per_level.get('Medium', 0)}M/{sel_per_level.get('Hard', 0)}H) |")
    lines.append(f"| Selection target met (>=70) | {'YES' if eligible_after >= 70 else 'NO'} |")
    lines.append("")

    # 7. Selected examples
    lines.append("## 7. Selected Examples (10 per level)")
    lines.append("")
    for d in levels:
        examples = [c for c in selected if c.get("target_difficulty") == d][:10]
        lines.append(f"### {d} Examples")
        lines.append("")
        lines.append("| # | Story | Answer | Answer Type | ASA | num_req | NT |")
        lines.append("|---|---|---|---|---|---|---|")
        for i, c in enumerate(examples, 1):
            lines.append(f"| {i} | {c['story_name'][:35]} | {c['answer'][:40]} | "
                        f"{c.get('answer_type', '?')} | {c.get('asa', '?')} | "
                        f"{c.get('num_required_sentences', '?')} | {c.get('necessity_type', '?')} |")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Suitability report: {report_path}")
    return eligible_after


def main():
    parser = argparse.ArgumentParser(description="FairytaleQA Candidate Suitability Audit")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--max_stories", type=int, default=None,
                        help="Max stories for story-matched selection")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load candidates
    print(f"Loading candidates: {args.candidates}")
    all_candidates = []
    with open(args.candidates, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_candidates.append(json.loads(line))
    print(f"  Loaded {len(all_candidates)} candidates")

    # Assess suitability
    print("Assessing candidate suitability...")
    for c in all_candidates:
        result = assess_candidate_suitability(c)
        c.update(result)

    suitable_pool = [c for c in all_candidates if c.get("suitable")]
    print(f"  Suitable: {len(suitable_pool)}/{len(all_candidates)} "
          f"({100*len(suitable_pool)/len(all_candidates):.1f}%)")

    # Count by difficulty
    for d in ["Easy", "Medium", "Hard"]:
        before = sum(1 for c in all_candidates if c.get("evidence_difficulty") == d)
        after = sum(1 for c in suitable_pool if c.get("evidence_difficulty") == d)
        print(f"  {d}: {after}/{before} suitable")

    # Write suitability file
    suitability_path = output_dir / "candidates.suitability.jsonl"
    with open(suitability_path, "w", encoding="utf-8") as f:
        for c in all_candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"  Wrote: {suitability_path}")

    # Story-matched selection
    print("Selecting story-matched suitable candidates...")
    selected, n_eligible = select_story_matched_suitable(
        suitable_pool,
        per_level_per_story=1,
        max_stories=args.max_stories,
        seed=args.seed,
    )
    print(f"  Eligible stories: {n_eligible}")
    print(f"  Selected: {len(selected)} candidates from {len(set(c['story_name'] for c in selected))} stories")

    # Write selected
    selected_path = output_dir / "selected_story_matched_suitable.jsonl"
    with open(selected_path, "w", encoding="utf-8") as f:
        for c in selected:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"  Wrote: {selected_path}")

    # Build report
    eligible_after = build_report(all_candidates, suitable_pool, selected, output_dir)

    print()
    print("=== Summary ===")
    print(f"  Total candidates: {len(all_candidates)}")
    print(f"  Suitable: {len(suitable_pool)} ({100*len(suitable_pool)/len(all_candidates):.1f}%)")
    print(f"  Story-matched eligible (>=1 per level): {eligible_after}")
    if eligible_after >= 70:
        print(f"  Target >=70: PASS — ready for smoke/full run")
    else:
        print(f"  Target >=70: FAIL — pool too small, review rejection reasons")


if __name__ == "__main__":
    main()
