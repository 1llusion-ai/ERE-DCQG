"""Compute Table 3 metrics: human evaluation of reranked vs K=1.

Computes difficulty accuracy (human-judged), inter-annotator kappa,
classifier-human kappa, LLM-human kappa, and bootstrap 95% CIs.

Usage:
    python -m scripts.compute_table3 \
        --annotations_path outputs/eval/human_eval_annotations.csv \
        --sample_metadata_path outputs/eval/human_eval_metadata.json \
        --output_dir outputs/eval/table3/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

LABELS = ["Easy", "Medium", "Hard"]


# ═══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute Table 3 metrics (human evaluation).",
    )
    p.add_argument(
        "--annotations_path",
        type=str,
        default="outputs/eval/human_eval_annotations.csv",
        help="Path to human annotations CSV (columns: sample_id, annotator, difficulty_label).",
    )
    p.add_argument(
        "--sample_metadata_path",
        type=str,
        default="outputs/eval/human_eval_metadata.json",
        help="Path to sample metadata JSON (maps sample_id -> source, difficulty_target).",
    )
    p.add_argument(
        "--classifier_predictions_path",
        type=str,
        default=None,
        help="Optional: classifier predictions JSONL (for kappa(classifier, human)).",
    )
    p.add_argument(
        "--llm_judge_path",
        type=str,
        default=None,
        help="Optional: LLM judge results JSONL (for kappa(LLM, human)).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/eval/table3/",
        help="Output directory for table3.json and table3.txt.",
    )
    p.add_argument("--n_bootstrap", type=int, default=1000, help="Bootstrap resamples.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_annotations(csv_path: str) -> list[dict]:
    """Load human annotations from CSV.

    Expected columns: sample_id, annotator, difficulty_label
    (additional columns are preserved).
    """
    rows: list[dict] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def load_metadata(json_path: str) -> dict[str, dict]:
    """Load sample metadata. Returns {sample_id: {source, difficulty_target, ...}}."""
    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)
    return {r["sample_id"]: r for r in records}


# ═══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════════════

def cohens_kappa(y1: list[str], y2: list[str]) -> float:
    """Cohen's kappa between two raters."""
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(y1, y2, labels=LABELS))


def difficulty_accuracy(
    annotations_by_sample: dict[str, str],
    metadata: dict[str, dict],
) -> dict[str, float]:
    """Compute difficulty accuracy: fraction where human label == target.

    Returns per-source accuracy and overall.
    """
    by_source: dict[str, list[bool]] = defaultdict(list)
    all_correct: list[bool] = []

    for sid, human_label in annotations_by_sample.items():
        meta = metadata.get(sid, {})
        target = meta.get("difficulty_target", "")
        source = meta.get("source", "unknown")
        correct = human_label == target
        by_source[source].append(correct)
        all_correct.append(correct)

    result: dict[str, float] = {}
    for source, corrects in by_source.items():
        result[source] = sum(corrects) / len(corrects) if corrects else 0.0
    result["overall"] = sum(all_correct) / len(all_correct) if all_correct else 0.0
    return result


def bootstrap_accuracy_ci(
    labels: list[bool],
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap 95% CI for accuracy."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    if n == 0:
        return (0.0, 0.0)

    arr = np.array(labels, dtype=float)
    scores: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        scores.append(float(arr[idx].mean()))

    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return (lower, upper)


def bootstrap_kappa_ci(
    y1: list[str],
    y2: list[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap 95% CI for Cohen's kappa."""
    from sklearn.metrics import cohen_kappa_score

    rng = np.random.RandomState(seed)
    n = len(y1)
    if n == 0:
        return (0.0, 0.0)

    scores: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        a = [y1[i] for i in idx]
        b = [y2[i] for i in idx]
        if len(set(a)) < 2 or len(set(b)) < 2:
            continue
        scores.append(float(cohen_kappa_score(a, b, labels=LABELS)))

    if not scores:
        return (0.0, 0.0)

    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return (lower, upper)


# ═══════════════════════════════════════════════════════════════════════════════
# Table formatting
# ═══════════════════════════════════════════════════════════════════════════════

def format_table(data: dict) -> str:
    """Format Table 3 as a readable text table."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("Table 3: Human Evaluation")
    lines.append("=" * 80)

    # Difficulty accuracy
    lines.append("")
    lines.append("Difficulty Accuracy (human-judged difficulty == target):")
    lines.append(f"  {'Source':<15} {'Accuracy':>10} {'95% CI':>20} {'N':>5}")
    lines.append("  " + "-" * 55)

    acc_data = data.get("difficulty_accuracy", {})
    for source in ["reranked", "k1", "overall"]:
        info = acc_data.get(source, {})
        acc = info.get("accuracy", 0.0)
        ci = info.get("ci_95", [0.0, 0.0])
        n = info.get("n", 0)
        lines.append(
            f"  {source:<15} {acc:>10.3f} [{ci[0]:.3f}, {ci[1]:.3f}]{n:>9}"
        )

    # Kappa scores
    lines.append("")
    lines.append("Agreement (Cohen's kappa):")
    lines.append(f"  {'Comparison':<35} {'Kappa':>8} {'95% CI':>20}")
    lines.append("  " + "-" * 65)

    for kappa_entry in data.get("kappa_scores", []):
        name = kappa_entry["comparison"]
        k = kappa_entry["kappa"]
        ci = kappa_entry.get("ci_95", [0.0, 0.0])
        lines.append(f"  {name:<35} {k:>8.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

    lines.append("=" * 80)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    print(f"Loading annotations: {args.annotations_path}")
    annotations = load_annotations(args.annotations_path)
    print(f"  {len(annotations)} annotation rows")

    print(f"Loading metadata: {args.sample_metadata_path}")
    metadata = load_metadata(args.sample_metadata_path)
    print(f"  {len(metadata)} samples")

    # Group annotations by annotator
    by_annotator: dict[str, dict[str, str]] = defaultdict(dict)
    for row in annotations:
        annotator = row.get("annotator", "A1")
        sid = row.get("sample_id", "")
        label = row.get("difficulty_label", "")
        if sid and label:
            by_annotator[annotator][sid] = label

    annotators = sorted(by_annotator.keys())
    print(f"  Annotators: {annotators}")

    # Majority vote across annotators (for accuracy computation)
    from collections import Counter

    majority_labels: dict[str, str] = {}
    for sid in metadata:
        votes = [by_annotator[a].get(sid) for a in annotators if by_annotator[a].get(sid)]
        if votes:
            counts = Counter(votes)
            # Tie-break: Easy < Medium < Hard
            max_count = max(counts.values())
            for label in LABELS:
                if counts.get(label, 0) == max_count:
                    majority_labels[sid] = label
                    break

    print(f"  Majority labels computed for {len(majority_labels)} samples")

    # ── Difficulty Accuracy ──
    acc_results: dict[str, dict] = {}

    for source in ["reranked", "k1", "overall"]:
        if source == "overall":
            subset = majority_labels
        else:
            subset = {
                sid: lab
                for sid, lab in majority_labels.items()
                if metadata.get(sid, {}).get("source") == source
            }

        acc = difficulty_accuracy(subset, metadata)
        correct_list = [
            subset[sid] == metadata.get(sid, {}).get("difficulty_target", "")
            for sid in subset
        ]
        ci = bootstrap_accuracy_ci(correct_list, args.n_bootstrap, args.seed)

        acc_results[source] = {
            "accuracy": round(acc.get(source, acc.get("overall", 0.0)), 4),
            "ci_95": [round(ci[0], 4), round(ci[1], 4)],
            "n": len(subset),
        }

    # ── Kappa scores ──
    kappa_scores: list[dict] = []

    # kappa(human-human): pairwise between annotators
    if len(annotators) >= 2:
        for i in range(len(annotators)):
            for j in range(i + 1, len(annotators)):
                a1, a2 = annotators[i], annotators[j]
                shared = sorted(
                    set(by_annotator[a1].keys()) & set(by_annotator[a2].keys())
                )
                if len(shared) < 5:
                    continue
                y1 = [by_annotator[a1][s] for s in shared]
                y2 = [by_annotator[a2][s] for s in shared]
                k = cohens_kappa(y1, y2)
                ci = bootstrap_kappa_ci(y1, y2, args.n_bootstrap, args.seed)
                kappa_scores.append({
                    "comparison": f"human-human ({a1} vs {a2})",
                    "kappa": round(k, 4),
                    "ci_95": [round(ci[0], 4), round(ci[1], 4)],
                    "n": len(shared),
                })

    # kappa(classifier, human): if classifier predictions available
    if args.classifier_predictions_path:
        from dcqg.utils.jsonl import read_jsonl as _read

        cls_records = _read(args.classifier_predictions_path)
        cls_map = {
            r.get("sample_id") or r.get("question", ""): (
                r.get("predicted_difficulty") or r.get("prediction", "Medium")
            )
            for r in cls_records
        }
        shared = sorted(set(majority_labels.keys()) & set(cls_map.keys()))
        if len(shared) >= 5:
            y_human = [majority_labels[s] for s in shared]
            y_cls = [cls_map[s] for s in shared]
            k = cohens_kappa(y_human, y_cls)
            ci = bootstrap_kappa_ci(y_human, y_cls, args.n_bootstrap, args.seed)
            kappa_scores.append({
                "comparison": "classifier vs human",
                "kappa": round(k, 4),
                "ci_95": [round(ci[0], 4), round(ci[1], 4)],
                "n": len(shared),
            })

    # kappa(LLM, human): if LLM judge results available
    if args.llm_judge_path:
        from dcqg.utils.jsonl import read_jsonl as _read

        llm_records = _read(args.llm_judge_path)
        for judge_key, judge_name in [
            ("gpt4omini_majority", "GPT-4o-mini"),
            ("qwen32b_majority", "Qwen-32B"),
        ]:
            llm_map = {}
            for r in llm_records:
                sid = r.get("sample_id") or r.get("story_name", "")
                pred = r.get(judge_key, "")
                if sid and pred:
                    llm_map[sid] = pred

            shared = sorted(set(majority_labels.keys()) & set(llm_map.keys()))
            if len(shared) >= 5:
                y_human = [majority_labels[s] for s in shared]
                y_llm = [llm_map[s] for s in shared]
                k = cohens_kappa(y_human, y_llm)
                ci = bootstrap_kappa_ci(y_human, y_llm, args.n_bootstrap, args.seed)
                kappa_scores.append({
                    "comparison": f"{judge_name} vs human",
                    "kappa": round(k, 4),
                    "ci_95": [round(ci[0], 4), round(ci[1], 4)],
                    "n": len(shared),
                })

    # ── Assemble output ──
    output_data = {
        "difficulty_accuracy": acc_results,
        "kappa_scores": kappa_scores,
        "config": {
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "n_annotations": len(annotations),
            "n_annotators": len(annotators),
        },
    }

    json_path = output_dir / "table3.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved structured data: {json_path}")

    table_text = format_table(output_data)
    txt_path = output_dir / "table3.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table_text + "\n")
    print(f"Saved formatted table: {txt_path}")

    print()
    print(table_text)


if __name__ == "__main__":
    main()
