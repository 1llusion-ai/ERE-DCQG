"""Stage C self-consistency aggregation.

Takes multiple independent Stage A audit runs and optional Stage B
counterfactual verification, applies majority vote to produce final
training labels.
"""

import json
from collections import Counter
from pathlib import Path

from dcqg.utils.jsonl import read_jsonl, write_jsonl
from dcqg.path.fairytale_evidence_audit import classify_difficulty, _split_sentences


def _make_key(record):
    return (record["story_name"], record["question"])


def _load_runs(run_paths):
    """Load multiple audit JSONL files, index by (story_name, question)."""
    runs = []
    for p in run_paths:
        index = {}
        for rec in read_jsonl(p):
            key = _make_key(rec)
            index[key] = rec
        runs.append(index)
    return runs


def _collect_all_keys(runs):
    all_keys = set()
    for index in runs:
        all_keys.update(index.keys())
    return sorted(all_keys)


def _majority_vote(values, threshold):
    """Return the majority value if it meets threshold, else None."""
    if not values:
        return None
    counts = Counter(values)
    winner, count = counts.most_common(1)[0]
    if count >= threshold:
        return winner
    return None


def _normalized_difficulty(record):
    """Recompute difficulty from current evidence fields."""
    return classify_difficulty(record)


def _vote_sentence_ids(records_agreeing, field, threshold):
    """Keep sentence ids selected by at least threshold agreeing records."""
    if not records_agreeing:
        return []

    counts = Counter()
    for rec in records_agreeing:
        counts.update(set(rec.get(field, [])))
    return sorted(sid for sid, count in counts.items() if count >= threshold)


def _pick_field(records_agreeing, field, default=""):
    """Pick the most common value for a metadata field among agreeing records."""
    values = [rec.get(field, default) for rec in records_agreeing]
    if not values:
        return default
    counts = Counter(values)
    return counts.most_common(1)[0][0]


def _apply_stage_b(record, stage_b_lookup):
    """Filter evidence using Stage B verification, reclassify difficulty."""
    key = _make_key(record)
    if key not in stage_b_lookup:
        return record

    verified = stage_b_lookup[key]
    # Current Stage B writes verified_evidence_sentences / verification_details.
    # Older drafts used sentence_verdicts; keep a fallback for compatibility.
    if "verified_evidence_sentences" in verified:
        necessary_ids = set(verified.get("verified_evidence_sentences", []))
    else:
        necessary_ids = set()
        for item in verified.get("sentence_verdicts", []):
            if item.get("verdict") == "necessary":
                necessary_ids.add(item["sentence_id"])

    original_evidence = set(record.get("required_evidence_sentences", []))
    filtered_evidence = sorted(original_evidence & necessary_ids)

    original_bridge = set(record.get("bridge_sentence_ids", []))
    filtered_bridge = sorted(original_bridge & necessary_ids)

    record["required_evidence_sentences"] = filtered_evidence
    record["bridge_sentence_ids"] = filtered_bridge
    record["num_required_sentences"] = len(filtered_evidence)
    record["verified_evidence_sentences"] = sorted(necessary_ids)
    record["dropped_evidence_sentences"] = verified.get(
        "dropped_evidence_sentences", []
    )
    record["verification_details"] = verified.get("verification_details", [])
    record["verified_num_required"] = verified.get(
        "verified_num_required", len(filtered_evidence)
    )

    # Reclassify using the same logic as Stage A but with filtered counts
    pseudo_assessment = {
        "num_required_sentences": len(filtered_evidence),
        "answer_directly_found": record.get("answer_directly_found", "no"),
        "answer_sentence_alone_sufficient": (
            "yes"
            if record.get("answer_directly_found") == "yes"
            and len(filtered_evidence) == 1
            else "no"
        ),
        "bridge_removal_effect": record.get("bridge_removal_effect", "none"),
        "necessity_type": record.get("necessity_type", "background_context"),
    }
    record["difficulty_label"] = classify_difficulty(pseudo_assessment)
    record["verified_difficulty"] = verified.get(
        "verified_difficulty", record["difficulty_label"]
    )
    return record


def build_consensus_records(run_paths, agreement_threshold=2):
    """Build Stage-A consensus records before Stage-B verification.

    Invalid Stage-A votes are excluded. For valid majority votes, evidence is
    retained when at least agreement_threshold agreeing runs selected it.
    """
    runs = _load_runs(run_paths)
    all_keys = _collect_all_keys(runs)

    results = []
    agreement_counts = Counter()
    dropped = 0

    for key in all_keys:
        present = [run[key] for run in runs if key in run]
        if len(present) < agreement_threshold:
            dropped += 1
            agreement_counts["too_few_runs"] += 1
            continue

        valid_pairs = []
        raw_votes = []
        for rec in present:
            difficulty = _normalized_difficulty(rec)
            raw_votes.append(difficulty)
            if difficulty != "Invalid":
                valid_pairs.append((rec, difficulty))

        if len(valid_pairs) < agreement_threshold:
            dropped += 1
            agreement_counts["too_few_valid_votes"] += 1
            continue

        valid_difficulties = [difficulty for _, difficulty in valid_pairs]
        consensus_difficulty = _majority_vote(
            valid_difficulties, agreement_threshold
        )
        if consensus_difficulty is None:
            dropped += 1
            agreement_counts["no_consensus"] += 1
            continue

        agreeing = [
            rec for rec, difficulty in valid_pairs
            if difficulty == consensus_difficulty
        ]
        n_agree = len(agreeing)
        agreement_counts[n_agree] += 1

        evidence_ids = _vote_sentence_ids(
            agreeing, "required_evidence_sentences", agreement_threshold
        )
        bridge_ids = _vote_sentence_ids(
            agreeing, "bridge_sentence_ids", agreement_threshold
        )
        bridge_ids = [sid for sid in bridge_ids if sid in evidence_ids]

        if not evidence_ids:
            dropped += 1
            agreement_counts["empty_consensus_evidence"] += 1
            continue

        base = agreeing[0]
        answer_directly_found = _pick_field(
            agreeing,
            "answer_directly_found",
            default="yes" if len(evidence_ids) == 1 else "no",
        )
        out_record = {
            "story_name": base["story_name"],
            "story_section": base.get("story_section", ""),
            "question": base["question"],
            "answer": base.get("answer", base.get("answer1", "")),
            "answer1": base.get("answer1", ""),
            "answer2": base.get("answer2", ""),
            "attribute": base.get("attribute", ""),
            "difficulty_label": consensus_difficulty,
            "evidence_difficulty": consensus_difficulty,
            "required_evidence_sentences": evidence_ids,
            "bridge_sentence_ids": bridge_ids,
            "num_required_sentences": len(evidence_ids),
            "answer_directly_found": answer_directly_found,
            "answer_sentence_alone_sufficient": (
                "yes"
                if answer_directly_found == "yes" and len(evidence_ids) == 1
                else "no"
            ),
            "section_evidence_sufficient": "yes",
            "full_context_needed": "no",
            "reasoning_operation": _pick_field(agreeing, "reasoning_operation"),
            "necessity_type": _pick_field(agreeing, "necessity_type"),
            "bridge_removal_effect": _pick_field(agreeing, "bridge_removal_effect"),
            "evidence_necessity_reason": _pick_field(
                agreeing, "evidence_necessity_reason"
            ),
            "agreement_count": n_agree,
            "difficulty_votes": dict(Counter(raw_votes)),
            "difficulty_agreement": round(n_agree / len(present), 3),
            "n_audit_runs": len(runs),
            "source": "implicit",
        }

        recomputed = classify_difficulty(out_record)
        if recomputed == "Invalid":
            dropped += 1
            agreement_counts["invalid_consensus_record"] += 1
            continue
        out_record["difficulty_label"] = recomputed
        out_record["evidence_difficulty"] = recomputed

        results.append(out_record)

    summary = {
        "total": len(results),
        "coverage": len(results) / len(all_keys) if all_keys else 0.0,
        "difficulty_distribution": dict(
            Counter(r["difficulty_label"] for r in results)
        ),
        "agreement_stats": dict(agreement_counts),
        "dropped_count": dropped,
    }
    return results, summary


def aggregate_audit_runs(run_paths, stage_b_path=None, output_path="labels.jsonl",
                         agreement_threshold=2):
    """Aggregate multiple audit runs into consensus labels.

    Args:
        run_paths: list of JSONL file paths (each from FairytaleEvidenceAuditor)
        stage_b_path: optional JSONL from counterfactual verification
        output_path: where to write final labels
        agreement_threshold: minimum agreement (2 = 2/3 majority)

    Returns:
        summary dict with keys: total, coverage, difficulty_distribution,
        agreement_stats, dropped_count
    """
    stage_b_lookup = {}
    if stage_b_path:
        for rec in read_jsonl(stage_b_path):
            stage_b_lookup[_make_key(rec)] = rec

    results = []
    consensus_records, consensus_summary = build_consensus_records(
        run_paths, agreement_threshold=agreement_threshold
    )
    agreement_counts = Counter(consensus_summary.get("agreement_stats", {}))
    dropped = 0

    for out_record in consensus_records:
        if stage_b_lookup:
            out_record = _apply_stage_b(out_record, stage_b_lookup)

        if out_record.get("difficulty_label") == "Invalid":
            dropped += 1
            agreement_counts["invalid_after_stage_b"] += 1
            continue

        results.append(out_record)

    write_jsonl(output_path, results)

    diff_dist = Counter(r["difficulty_label"] for r in results)
    return {
        "total": len(results),
        "coverage": consensus_summary.get("coverage", 0.0),
        "difficulty_distribution": dict(diff_dist),
        "agreement_stats": dict(agreement_counts),
        "dropped_count": consensus_summary.get("dropped_count", 0) + dropped,
    }


def _fuzzy_find_answer_sentence(sentences, answer_text):
    """Find the sentence index that best matches the answer by token overlap."""
    if not sentences or not answer_text:
        return 0

    answer_tokens = set(answer_text.lower().split())
    best_idx = 0
    best_score = 0.0

    for i, sent in enumerate(sentences):
        sent_tokens = set(sent.lower().split())
        if not sent_tokens:
            continue
        overlap = len(answer_tokens & sent_tokens)
        score = overlap / max(len(answer_tokens), 1)
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def build_explicit_labels(records, output_path):
    """Build labels for explicit FairytaleQA items.

    Explicit items are overwhelmingly Easy (answer directly stated).

    Args:
        records: list of dicts with at least story_section, question, answer1, attribute
        output_path: where to write labels

    Returns:
        summary dict
    """
    results = []

    for rec in records:
        sentences = _split_sentences(rec.get("story_section", ""))
        answer_text = rec.get("answer1", "")
        answer_idx = _fuzzy_find_answer_sentence(sentences, answer_text)

        out_record = {
            "story_name": rec.get("story_name", ""),
            "story_section": rec.get("story_section", ""),
            "question": rec.get("question", ""),
            "answer": answer_text,
            "answer1": rec.get("answer1", ""),
            "answer2": rec.get("answer2", ""),
            "attribute": rec.get("attribute", ""),
            "local_or_sum": rec.get("local_or_sum", ""),
            "ex_or_im": rec.get("ex_or_im", "explicit"),
            "ex_or_im2": rec.get("ex_or_im2", ""),
            "difficulty_label": "Easy",
            "suggested_difficulty_label": "Easy",
            "selector_difficulty": "Easy",
            "selector_status": "ok",
            "annotation_priority": "high",
            "required_evidence_sentences": [answer_idx],
            "selected_evidence_sentences": [answer_idx],
            "bridge_sentence_ids": [],
            "num_required_sentences": 1,
            "answer_directly_found": "yes",
            "final_answer_directly_found": "yes",
            "suggested_answer_directly_found": "yes",
            "answer_sentence_alone_sufficient": "yes",
            "section_evidence_sufficient": "yes",
            "section_sufficient": "yes",
            "full_context_needed": "no",
            "evidence_set_sufficient": "yes",
            "sufficiency_reason": "Explicit FairytaleQA item; answer sentence selected by answer-text overlap.",
            "reasoning_operation": "explicit_lookup",
            "necessity_type": "direct_answer",
            "agreement_count": 1,
            "source": "explicit",
            "sample_original_index": rec.get("sample_original_index"),
            "sample_mode": rec.get("sample_mode"),
            "sample_seed": rec.get("sample_seed"),
        }
        results.append(out_record)

    write_jsonl(output_path, results)

    return {
        "total": len(results),
        "difficulty_distribution": {"Easy": len(results)},
    }


def merge_label_files(implicit_path, explicit_path, output_path):
    """Merge implicit + explicit labels into final train_dataset.jsonl.

    Returns summary dict with total count and difficulty distribution.
    """
    implicit = [
        rec for rec in read_jsonl(implicit_path)
        if rec.get("difficulty_label") != "Invalid"
    ]
    explicit = [
        rec for rec in read_jsonl(explicit_path)
        if rec.get("difficulty_label") != "Invalid"
    ]

    # Deduplicate: if the same (story_name, question) appears in both,
    # prefer the implicit label (it has auditor-verified evidence).
    seen = set()
    merged = []

    for rec in implicit:
        key = _make_key(rec)
        if key not in seen:
            seen.add(key)
            merged.append(rec)

    for rec in explicit:
        key = _make_key(rec)
        if key not in seen:
            seen.add(key)
            merged.append(rec)

    write_jsonl(output_path, merged)

    diff_dist = Counter(r["difficulty_label"] for r in merged)
    return {
        "total": len(merged),
        "difficulty_distribution": dict(diff_dist),
        "implicit_count": len(implicit),
        "explicit_count": len(explicit),
        "overlap_count": len(implicit) + len(explicit) - len(merged),
    }
