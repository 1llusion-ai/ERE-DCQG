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


def _intersect_evidence(records_agreeing):
    """Take intersection of required_evidence_sentences across agreeing records."""
    if not records_agreeing:
        return []
    sets = []
    for rec in records_agreeing:
        ids = rec.get("required_evidence_sentences", [])
        sets.append(set(ids))
    common = sets[0]
    for s in sets[1:]:
        common = common & s
    return sorted(common)


def _intersect_bridge(records_agreeing):
    if not records_agreeing:
        return []
    sets = []
    for rec in records_agreeing:
        ids = rec.get("bridge_sentence_ids", [])
        sets.append(set(ids))
    common = sets[0]
    for s in sets[1:]:
        common = common & s
    return sorted(common)


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

    # Reclassify using the same logic as Stage A but with filtered counts
    pseudo_assessment = {
        "num_required_sentences": len(filtered_evidence),
        "answer_sentence_alone_sufficient": "yes" if len(filtered_evidence) <= 1 else "no",
        "bridge_removal_effect": record.get("bridge_removal_effect", "none"),
        "necessity_type": record.get("necessity_type", "background_context"),
    }
    record["difficulty_label"] = classify_difficulty(pseudo_assessment)
    return record


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
    runs = _load_runs(run_paths)
    all_keys = _collect_all_keys(runs)

    stage_b_lookup = {}
    if stage_b_path:
        for rec in read_jsonl(stage_b_path):
            stage_b_lookup[_make_key(rec)] = rec

    results = []
    agreement_counts = Counter()
    dropped = 0

    for key in all_keys:
        present = [run[key] for run in runs if key in run]
        if len(present) < agreement_threshold:
            dropped += 1
            continue

        difficulties = [rec.get("evidence_difficulty", "Easy") for rec in present]
        consensus_difficulty = _majority_vote(difficulties, agreement_threshold)

        if consensus_difficulty is None:
            dropped += 1
            agreement_counts["no_consensus"] += 1
            continue

        agreeing = [rec for rec, d in zip(present, difficulties)
                    if d == consensus_difficulty]
        n_agree = len(agreeing)
        agreement_counts[n_agree] += 1

        evidence_ids = _intersect_evidence(agreeing)
        bridge_ids = _intersect_bridge(agreeing)

        # Use first agreeing record as base for metadata
        base = agreeing[0]
        out_record = {
            "story_name": base["story_name"],
            "story_section": base.get("story_section", ""),
            "question": base["question"],
            "answer1": base.get("answer1", ""),
            "answer2": base.get("answer2", ""),
            "attribute": base.get("attribute", ""),
            "difficulty_label": consensus_difficulty,
            "required_evidence_sentences": evidence_ids,
            "bridge_sentence_ids": bridge_ids,
            "num_required_sentences": len(evidence_ids),
            "reasoning_operation": _pick_field(agreeing, "reasoning_operation"),
            "necessity_type": _pick_field(agreeing, "necessity_type"),
            "bridge_removal_effect": _pick_field(agreeing, "bridge_removal_effect"),
            "agreement_count": n_agree,
            "source": "implicit",
        }

        if stage_b_lookup:
            out_record = _apply_stage_b(out_record, stage_b_lookup)

        results.append(out_record)

    write_jsonl(output_path, results)

    diff_dist = Counter(r["difficulty_label"] for r in results)
    return {
        "total": len(results),
        "coverage": len(results) / len(all_keys) if all_keys else 0.0,
        "difficulty_distribution": dict(diff_dist),
        "agreement_stats": dict(agreement_counts),
        "dropped_count": dropped,
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
            "answer1": rec.get("answer1", ""),
            "answer2": rec.get("answer2", ""),
            "attribute": rec.get("attribute", ""),
            "difficulty_label": "Easy",
            "required_evidence_sentences": [answer_idx],
            "bridge_sentence_ids": [],
            "num_required_sentences": 1,
            "reasoning_operation": "lookup",
            "necessity_type": "direct_answer",
            "agreement_count": 1,
            "source": "explicit",
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
    implicit = read_jsonl(implicit_path)
    explicit = read_jsonl(explicit_path)

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
