"""Compute Table 1 metrics: classifier vs LLM judges vs human ground truth.

Computes per-system: macro F1, per-class F1, Cohen's kappa, accuracy,
bootstrap 95% CI for macro F1, and McNemar's test (classifier vs each LLM judge).

Usage:
    python -m scripts.compute_table1 \
        --human_annotations_path outputs/eval/human_annotations.jsonl \
        --classifier_predictions_path outputs/eval/classifier_predictions.jsonl \
        --llm_judge_path outputs/eval/llm_judge_results.jsonl \
        --output_dir outputs/eval/table1/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dcqg.utils.jsonl import read_jsonl

LABELS = ["Easy", "Medium", "Hard"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}


# ═══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute Table 1 metrics (classifier vs LLM judges).",
    )
    p.add_argument(
        "--human_annotations_path",
        type=str,
        default="outputs/eval/human_annotations.jsonl",
        help="Path to human ground-truth annotations JSONL.",
    )
    p.add_argument(
        "--classifier_predictions_path",
        type=str,
        default="outputs/eval/classifier_predictions.jsonl",
        help="Path to classifier predictions JSONL.",
    )
    p.add_argument(
        "--llm_judge_path",
        type=str,
        default="outputs/eval/llm_judge_results.jsonl",
        help="Path to LLM judge results JSONL.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/eval/table1/",
        help="Output directory for table1.json and table1.txt.",
    )
    p.add_argument("--n_bootstrap", type=int, default=1000, help="Bootstrap resamples.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Simple accuracy."""
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def compute_f1_per_class(
    y_true: list[str], y_pred: list[str], labels: list[str],
) -> dict[str, float]:
    """Per-class F1 scores."""
    from sklearn.metrics import f1_score

    return {
        label: float(f1_score(y_true, y_pred, labels=labels, average=None)[i])
        for i, label in enumerate(labels)
    }


def compute_macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    """Macro F1 score."""
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, labels=LABELS, average="macro"))


def compute_cohens_kappa(y_true: list[str], y_pred: list[str]) -> float:
    """Cohen's kappa vs ground truth."""
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(y_true, y_pred, labels=LABELS))


def bootstrap_macro_f1_ci(
    y_true: list[str],
    y_pred: list[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap 95% CI for macro F1. Returns (lower, upper)."""
    from sklearn.metrics import f1_score

    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        # Guard against degenerate samples
        if len(set(yt)) < 2:
            continue
        scores.append(float(f1_score(yt, yp, labels=LABELS, average="macro")))

    if not scores:
        return (0.0, 0.0)

    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return (lower, upper)


def mcnemar_test(
    y_true: list[str], y_pred_a: list[str], y_pred_b: list[str],
) -> dict:
    """McNemar's test comparing two classifiers against the same ground truth.

    Returns chi2 statistic and p-value.
    Counts:
      b = items A got right but B got wrong
      c = items B got right but A got wrong
    """
    from scipy.stats import chi2 as chi2_dist

    b = sum(
        (pa == t) and (pb != t)
        for t, pa, pb in zip(y_true, y_pred_a, y_pred_b)
    )
    c = sum(
        (pb == t) and (pa != t)
        for t, pa, pb in zip(y_true, y_pred_a, y_pred_b)
    )

    # Continuity-corrected McNemar
    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": b, "c": c}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1.0 - chi2_dist.cdf(chi2, df=1))
    return {"chi2": round(chi2, 4), "p_value": round(p_value, 4), "b": b, "c": c}


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation for one system
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_system(
    name: str,
    y_true: list[str],
    y_pred: list[str],
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Compute all metrics for one system."""
    acc = compute_accuracy(y_true, y_pred)
    macro_f1 = compute_macro_f1(y_true, y_pred)
    per_class_f1 = compute_f1_per_class(y_true, y_pred, LABELS)
    kappa = compute_cohens_kappa(y_true, y_pred)
    ci_lower, ci_upper = bootstrap_macro_f1_ci(
        y_true, y_pred, n_bootstrap=n_bootstrap, seed=seed,
    )

    return {
        "system": name,
        "n": len(y_true),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": {k: round(v, 4) for k, v in per_class_f1.items()},
        "cohens_kappa": round(kappa, 4),
        "bootstrap_ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_gt(records: list[dict]) -> list[str]:
    """Extract ground-truth labels from human annotation records."""
    return [
        r.get("difficulty") or r.get("difficulty_label") or r.get("label", "Medium")
        for r in records
    ]


def _extract_classifier_preds(records: list[dict]) -> list[str]:
    """Extract classifier predictions."""
    return [
        r.get("predicted_difficulty") or r.get("prediction") or r.get("label", "Medium")
        for r in records
    ]


def _align_by_key(
    gt_records: list[dict],
    pred_records: list[dict],
    key: str = "question",
) -> tuple[list[str], list[str]]:
    """Align ground-truth and predictions by a shared key field."""
    pred_map = {}
    for r in pred_records:
        k = r.get(key, "")
        if k:
            pred_map[k] = r

    y_true, y_pred = [], []
    for r in gt_records:
        k = r.get(key, "")
        if k in pred_map:
            y_true.append(
                r.get("difficulty") or r.get("difficulty_label") or r.get("label", "Medium")
            )
            y_pred.append(
                pred_map[k].get("predicted_difficulty")
                or pred_map[k].get("prediction")
                or pred_map[k].get("label", "Medium")
            )

    return y_true, y_pred


# ═══════════════════════════════════════════════════════════════════════════════
# Table formatting
# ═══════════════════════════════════════════════════════════════════════════════

def format_table(results: list[dict], mcnemar_results: list[dict]) -> str:
    """Format results as a readable text table."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("Table 1: Classifier vs LLM Judges")
    lines.append("=" * 90)

    header = (
        f"{'System':<20} {'N':>4} {'Acc':>6} {'Macro F1':>9} "
        f"{'F1-E':>6} {'F1-M':>6} {'F1-H':>6} {'Kappa':>7} {'95% CI':>16}"
    )
    lines.append(header)
    lines.append("-" * 90)

    for r in results:
        ci = r["bootstrap_ci_95"]
        pf = r["per_class_f1"]
        lines.append(
            f"{r['system']:<20} {r['n']:>4} {r['accuracy']:>6.3f} {r['macro_f1']:>9.4f} "
            f"{pf.get('Easy', 0):>6.3f} {pf.get('Medium', 0):>6.3f} "
            f"{pf.get('Hard', 0):>6.3f} {r['cohens_kappa']:>7.3f} "
            f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        )

    lines.append("-" * 90)

    if mcnemar_results:
        lines.append("")
        lines.append("McNemar's Test (classifier vs each LLM judge):")
        lines.append(f"  {'Comparison':<40} {'chi2':>8} {'p-value':>8} {'b':>4} {'c':>4}")
        lines.append("  " + "-" * 70)
        for m in mcnemar_results:
            lines.append(
                f"  {m['comparison']:<40} {m['chi2']:>8.3f} {m['p_value']:>8.4f} "
                f"{m['b']:>4} {m['c']:>4}"
            )

    lines.append("=" * 90)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading human annotations: {args.human_annotations_path}")
    human_records = read_jsonl(args.human_annotations_path)
    print(f"  {len(human_records)} records")

    print(f"Loading classifier predictions: {args.classifier_predictions_path}")
    classifier_records = read_jsonl(args.classifier_predictions_path)
    print(f"  {len(classifier_records)} records")

    print(f"Loading LLM judge results: {args.llm_judge_path}")
    llm_records = read_jsonl(args.llm_judge_path)
    print(f"  {len(llm_records)} records")

    # Ground truth from human annotations
    gt_labels = _extract_gt(human_records)

    # Align classifier predictions
    cls_true, cls_pred = _align_by_key(human_records, classifier_records)
    print(f"\nAligned classifier: {len(cls_true)} items")

    # LLM judge predictions (aligned by index with human annotations)
    gpt_preds = [r.get("gpt4omini_majority", "Medium") for r in llm_records]
    qwen_preds = [r.get("qwen32b_majority", "Medium") for r in llm_records]

    # Use ground truth from LLM records if they include it, else from human
    llm_gt = [r.get("ground_truth", "Medium") for r in llm_records]

    # Evaluate each system
    results: list[dict] = []

    if cls_true:
        results.append(
            evaluate_system("Classifier", cls_true, cls_pred, args.n_bootstrap, args.seed)
        )

    if llm_gt and gpt_preds:
        results.append(
            evaluate_system("GPT-4o-mini", llm_gt, gpt_preds, args.n_bootstrap, args.seed)
        )

    if llm_gt and qwen_preds:
        results.append(
            evaluate_system("Qwen-32B", llm_gt, qwen_preds, args.n_bootstrap, args.seed)
        )

    # McNemar's test: classifier vs each LLM judge
    mcnemar_results: list[dict] = []

    # For McNemar, we need matched samples. Use the LLM ground truth as reference.
    if cls_true and llm_gt and len(cls_pred) == len(gpt_preds):
        m = mcnemar_test(llm_gt, cls_pred, gpt_preds)
        m["comparison"] = "Classifier vs GPT-4o-mini"
        mcnemar_results.append(m)

    if cls_true and llm_gt and len(cls_pred) == len(qwen_preds):
        m = mcnemar_test(llm_gt, cls_pred, qwen_preds)
        m["comparison"] = "Classifier vs Qwen-32B"
        mcnemar_results.append(m)

    # Output structured JSON
    output_data = {
        "systems": results,
        "mcnemar_tests": mcnemar_results,
        "config": {
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "n_samples": len(gt_labels),
        },
    }

    json_path = output_dir / "table1.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved structured data: {json_path}")

    # Output formatted table
    table_text = format_table(results, mcnemar_results)
    txt_path = output_dir / "table1.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table_text + "\n")
    print(f"Saved formatted table: {txt_path}")

    # Print to stdout
    print()
    print(table_text)


if __name__ == "__main__":
    main()
