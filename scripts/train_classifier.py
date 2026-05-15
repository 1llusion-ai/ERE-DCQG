"""Train the multi-task DeBERTa difficulty classifier with stratified k-fold CV.

Usage
-----
# Full 5-fold training:
python -m scripts.train_classifier \
    --data_path outputs/runs/evidence_audit_full/train_dataset.jsonl \
    --output_dir outputs/models/multitask_deberta_v1/ \
    --n_folds 5 --epochs 10 --batch_size 16 --lr 2e-5 \
    --lambda_evidence 0.3 --patience 3 --seed 42

# Ablation variants:
python -m scripts.train_classifier --lambda_evidence 0.0   # single-task (no evidence head)
python -m scripts.train_classifier --lambda_evidence 1.0   # equal weight

# Sanity check (overfit a tiny subset):
python -m scripts.train_classifier --overfit_check 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project root on sys.path so ``dcqg`` resolves when invoked as a module.
# ---------------------------------------------------------------------------
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dcqg.difficulty.data import (
    load_training_labels,
    create_stratified_folds,
    DifficultyEvidenceDataset,
    MARKER_TOKENS,
)
from dcqg.difficulty.classifier import MultiTaskDifficultyClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the multi-task DeBERTa difficulty classifier.",
    )
    p.add_argument(
        "--data_path",
        type=str,
        default="outputs/runs/evidence_audit_full/train_dataset.jsonl",
        help="Path to training JSONL.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/models/multitask_deberta_v1/",
        help="Directory for saved models and metrics.",
    )
    p.add_argument("--n_folds", type=int, default=5, help="Number of CV folds.")
    p.add_argument("--epochs", type=int, default=10, help="Max training epochs per fold.")
    p.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    p.add_argument("--lr", type=float, default=2e-5, help="Peak learning rate.")
    p.add_argument("--warmup_ratio", type=float, default=0.1, help="LR warmup ratio.")
    p.add_argument(
        "--lambda_evidence",
        type=float,
        default=0.3,
        help="Weight for the evidence detection loss (0=single-task, 1=equal).",
    )
    p.add_argument("--patience", type=int, default=3, help="Early stopping patience.")
    p.add_argument(
        "--model_name",
        type=str,
        default="microsoft/deberta-v3-base",
        help="HuggingFace model name.",
    )
    p.add_argument("--max_length", type=int, default=512, help="Max token length.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--overfit_check",
        type=int,
        default=None,
        help="If set to N, run a sanity overfit check on N items for 20 epochs.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for training (cuda / cpu).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_tokenizer(model_name: str):
    """Create a DeBERTa tokenizer with sentence marker special tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": MARKER_TOKENS}
    )
    if num_added:
        logger.info("Added %d marker tokens to tokenizer.", num_added)
    return tokenizer


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Overfit sanity check
# ---------------------------------------------------------------------------

def run_overfit_check(args: argparse.Namespace, records: list[dict]) -> None:
    """Train on a tiny subset for 20 epochs and assert the loss drops."""
    import torch

    n = min(args.overfit_check, len(records))
    subset = records[:n]
    print(f"\n=== OVERFIT CHECK: {n} items, 20 epochs ===\n")

    _set_seed(args.seed)
    tokenizer = _build_tokenizer(args.model_name)
    dataset = DifficultyEvidenceDataset(subset, tokenizer, max_length=args.max_length)

    classifier = MultiTaskDifficultyClassifier(
        model_name=args.model_name,
        tokenizer=tokenizer,
        lambda_evidence=args.lambda_evidence,
        device=args.device,
    )

    final_loss = classifier.train_fold(
        train_dataset=dataset,
        val_dataset=None,  # no validation in overfit mode
        epochs=20,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        patience=None,  # no early stopping
    )

    # ``train_fold`` returns a metrics dict; extract final training loss.
    if isinstance(final_loss, dict):
        loss_value = final_loss.get("final_train_loss", float("inf"))
    else:
        loss_value = float(final_loss)

    print(f"\nFinal training loss: {loss_value:.4f}")

    if loss_value < 0.1:
        print("SANITY CHECK PASSED")
    else:
        print("SANITY CHECK FAILED")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Single fold
# ---------------------------------------------------------------------------

def train_single_fold(
    fold_idx: int,
    train_records: list[dict],
    val_records: list[dict],
    args: argparse.Namespace,
) -> dict:
    """Train one fold and return its metrics dict."""
    import torch

    print(f"\n{'=' * 60}")
    print(f"  Fold {fold_idx}  |  train={len(train_records)}  val={len(val_records)}")
    print(f"{'=' * 60}\n")

    _set_seed(args.seed + fold_idx)

    tokenizer = _build_tokenizer(args.model_name)

    train_ds = DifficultyEvidenceDataset(
        train_records, tokenizer, max_length=args.max_length,
    )
    val_ds = DifficultyEvidenceDataset(
        val_records, tokenizer, max_length=args.max_length,
    )

    classifier = MultiTaskDifficultyClassifier(
        model_name=args.model_name,
        tokenizer=tokenizer,
        lambda_evidence=args.lambda_evidence,
        device=args.device,
    )

    metrics = classifier.train_fold(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        patience=args.patience,
    )

    # ── Persist fold artefacts ──────────────────────────────────────
    fold_dir = _ensure_dir(os.path.join(args.output_dir, f"fold_{fold_idx}"))

    # Save model weights
    model_dir = fold_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    classifier.save(str(model_dir))
    print(f"  Model saved to {model_dir}")

    # Save fold metrics
    metrics_path = fold_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  Metrics saved to {metrics_path}")

    return metrics


# ---------------------------------------------------------------------------
# Cross-validation summary
# ---------------------------------------------------------------------------

def compute_cv_summary(
    fold_metrics: list[dict],
    args: argparse.Namespace,
) -> dict:
    """Aggregate per-fold metrics into a CV summary."""
    macro_f1_scores: list[float] = []
    evidence_f1_scores: list[float] = []
    per_fold: list[dict] = []

    for i, m in enumerate(fold_metrics):
        macro_f1 = m.get("macro_f1", 0.0)
        macro_f1_scores.append(macro_f1)

        ev_f1 = m.get("evidence_f1", 0.0)
        evidence_f1_scores.append(ev_f1)

        per_class = m.get("per_class_f1", {})
        per_fold.append({
            "fold": i,
            "macro_f1": macro_f1,
            "evidence_f1": ev_f1,
            "per_class_f1": per_class,
        })

    macro_arr = np.array(macro_f1_scores)
    ev_arr = np.array(evidence_f1_scores)

    summary = {
        "n_folds": args.n_folds,
        "lambda_evidence": args.lambda_evidence,
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "patience": args.patience,
        "seed": args.seed,
        "mean_macro_f1": float(macro_arr.mean()),
        "std_macro_f1": float(macro_arr.std()),
        "mean_evidence_f1": float(ev_arr.mean()),
        "std_evidence_f1": float(ev_arr.std()),
        "per_fold": per_fold,
    }
    return summary


def print_summary(summary: dict) -> None:
    """Pretty-print the CV summary to stdout."""
    print(f"\n{'=' * 60}")
    print("  Cross-Validation Summary")
    print(f"{'=' * 60}")
    print(f"  Folds:            {summary['n_folds']}")
    print(f"  lambda_evidence:  {summary['lambda_evidence']}")
    print(f"  Model:            {summary['model_name']}")
    print(f"  Macro F1:         {summary['mean_macro_f1']:.4f} +/- {summary['std_macro_f1']:.4f}")
    print(f"  Evidence F1:      {summary['mean_evidence_f1']:.4f} +/- {summary['std_evidence_f1']:.4f}")
    print()

    for fold_info in summary["per_fold"]:
        per_class = fold_info.get("per_class_f1", {})
        cls_str = "  ".join(f"{k}={v:.3f}" for k, v in sorted(per_class.items()))
        print(
            f"  Fold {fold_info['fold']}:  macro_f1={fold_info['macro_f1']:.4f}  "
            f"evidence_f1={fold_info['evidence_f1']:.4f}  {cls_str}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    print(f"Loading data from: {args.data_path}")
    t0 = time.time()
    records = load_training_labels(args.data_path)
    t_load = time.time() - t0
    print(f"Loaded {len(records)} records in {t_load:.1f}s")

    if len(records) == 0:
        print("ERROR: no valid records found. Exiting.")
        sys.exit(1)

    # Print label distribution
    from collections import Counter
    label_counts = Counter(r["difficulty_label"] for r in records)
    print(f"Label distribution: {dict(sorted(label_counts.items()))}")

    # ── Overfit sanity check ────────────────────────────────────────
    if args.overfit_check is not None:
        run_overfit_check(args, records)
        return  # unreachable (overfit_check calls sys.exit)

    # ── Normal k-fold training ──────────────────────────────────────
    output_dir = _ensure_dir(args.output_dir)
    print(f"Output directory: {output_dir}")

    folds = create_stratified_folds(records, n_folds=args.n_folds, seed=args.seed)
    print(f"Created {len(folds)} stratified folds")

    fold_metrics: list[dict] = []

    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        train_records = [records[i] for i in train_indices]
        val_records = [records[i] for i in val_indices]
        metrics = train_single_fold(fold_idx, train_records, val_records, args)
        fold_metrics.append(metrics)

    # ── CV summary ──────────────────────────────────────────────────
    summary = compute_cv_summary(fold_metrics, args)
    print_summary(summary)

    summary_path = output_dir / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"CV summary saved to {summary_path}")


if __name__ == "__main__":
    main()
