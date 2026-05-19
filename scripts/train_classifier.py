"""Train the DeBERTa multi-task difficulty/evidence classifier.

Examples
--------
Local CPU smoke test with a tiny backbone:

    python -m scripts.train_classifier --smoke_test --device cpu

Full training after Label Studio export is converted/available:

    python -m scripts.train_classifier `
      --data_path outputs/runs/classifier_data/train.jsonl `
      --output_dir outputs/models/deberta_v3_base_multitask `
      --model_name microsoft/deberta-v3-base `
      --split_strategy story `
      --n_folds 5 --epochs 8 --batch_size 8 --lr 2e-5
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dcqg.difficulty.classifier import MultiTaskDifficultyClassifier
from dcqg.difficulty.data import (
    DEFAULT_MAX_MARKERS,
    DifficultyEvidenceDataset,
    create_stratified_folds,
    load_training_records,
    write_jsonl,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a two-head DeBERTa-v3-base classifier.",
    )
    parser.add_argument(
        "--data_path",
        action="append",
        default=[],
        help="Training data path. Can be Label Studio JSON, normalized JSONL, or CSV. Repeatable.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/models/deberta_v3_base_multitask",
        help="Output directory for checkpoints and metrics.",
    )
    parser.add_argument(
        "--model_name",
        default="microsoft/deberta-v3-base",
        help="HuggingFace encoder backbone.",
    )
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lambda_evidence", type=float, default=0.3)
    parser.add_argument("--evidence_pos_weight", type=float, default=3.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_markers", type=int, default=DEFAULT_MAX_MARKERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Use cpu locally if no CUDA GPU is available.",
    )
    parser.add_argument(
        "--split_strategy",
        choices=["stratified", "story"],
        default="story",
        help="Use story to keep the same story_name in one fold.",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Optional cap for local debugging.",
    )
    parser.add_argument(
        "--allow_model_labels",
        action="store_true",
        help="Use model-suggested labels when human annotations are absent. Not recommended for final training.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run a tiny synthetic end-to-end training pass.",
    )
    parser.add_argument(
        "--smoke_model_name",
        default="hf-internal-testing/tiny-random-bert",
        help="Tiny model used only by --smoke_test.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_smoke_records() -> list[dict[str, Any]]:
    contexts = [
        ("story-a", "Easy", [1], "what did the princess lose ?", "the ring ."),
        ("story-b", "Medium", [0], "why was the boy happy ?", "he felt lucky ."),
        ("story-c", "Hard", [0, 2], "why did the king trust the stranger ?", "the stranger had helped before ."),
        ("story-d", "Easy", [0], "where did the old woman live ?", "in the forest ."),
        ("story-e", "Medium", [1, 2], "who did the child follow ?", "the bird ."),
        ("story-f", "Hard", [0, 1, 3], "why did the queen forgive him ?", "he had saved her child ."),
    ]
    records: list[dict[str, Any]] = []
    for idx, (story, label, evidence, question, answer) in enumerate(contexts):
        records.append(
            {
                "sample_id": f"SMOKE{idx:03d}",
                "story_name": story,
                "source_index": str(idx),
                "question": question,
                "answer": answer,
                "difficulty_label": label,
                "difficulty_id": {"Easy": 0, "Medium": 1, "Hard": 2}[label],
                "answer_directly_found": "yes" if label == "Easy" else "no",
                "evidence_ids": evidence,
                "context_numbered": "\n".join(
                    [
                        "[S0] the stranger had once saved the child from danger .",
                        "[S1] the princess searched every room and found that the ring was gone .",
                        "[S2] the boy said that anyone with such a helper would be lucky .",
                        "[S3] later the queen remembered the rescue and forgave him .",
                    ]
                ),
                "sentences": [
                    {"id": 0, "text": "the stranger had once saved the child from danger ."},
                    {"id": 1, "text": "the princess searched every room and found that the ring was gone ."},
                    {"id": 2, "text": "the boy said that anyone with such a helper would be lucky ."},
                    {"id": 3, "text": "later the queen remembered the rescue and forgave him ."},
                ],
                "label_source": "smoke",
            }
        )
    return records


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.smoke_test:
        args.model_name = args.smoke_model_name
        args.n_folds = min(args.n_folds, 2)
        args.epochs = min(args.epochs, 1)
        args.batch_size = min(args.batch_size, 2)
        return build_smoke_records()

    if not args.data_path:
        raise SystemExit("ERROR: --data_path is required unless --smoke_test is set.")

    records = load_training_records(
        args.data_path,
        allow_model_labels=args.allow_model_labels,
    )
    if args.max_records is not None:
        records = records[: args.max_records]
    return records


def maybe_reduce_folds(records: list[dict[str, Any]], requested_folds: int) -> int:
    label_counts = Counter(record["difficulty_label"] for record in records)
    min_count = min(label_counts.values()) if label_counts else 0
    n_folds = min(requested_folds, min_count)
    if n_folds < 2:
        raise SystemExit(
            "ERROR: at least two examples per difficulty label are needed for CV. "
            f"Current label distribution: {dict(label_counts)}"
        )
    if n_folds != requested_folds:
        logger.warning("Reducing n_folds from %d to %d due to label counts.", requested_folds, n_folds)
    return n_folds


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_records": len(records),
        "difficulty_distribution": dict(Counter(record["difficulty_label"] for record in records)),
        "label_source_distribution": dict(Counter(record.get("label_source", "") for record in records)),
        "num_stories": len({record.get("story_name", "") for record in records}),
    }


def train_one_fold(
    fold_idx: int,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    print(f"\nFold {fold_idx}: train={len(train_records)} val={len(val_records)}")
    set_seed(args.seed + fold_idx)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [f"[S{i}]" for i in range(args.max_markers)]}
    )

    train_dataset = DifficultyEvidenceDataset(
        train_records,
        tokenizer,
        max_length=args.max_length,
        max_markers=args.max_markers,
    )
    val_dataset = DifficultyEvidenceDataset(
        val_records,
        tokenizer,
        max_length=args.max_length,
        max_markers=args.max_markers,
    )

    classifier = MultiTaskDifficultyClassifier(
        model_name=args.model_name,
        tokenizer=tokenizer,
        lambda_evidence=args.lambda_evidence,
        evidence_pos_weight=args.evidence_pos_weight,
        max_markers=args.max_markers,
        dropout=args.dropout,
        device=args.device,
    )
    print(f"Model parameters: {classifier.num_parameters / 1e6:.2f}M")

    metrics = classifier.train_fold(
        train_dataset,
        val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed + fold_idx,
    )

    fold_dir = Path(args.output_dir) / f"fold_{fold_idx}"
    model_dir = fold_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    classifier.save(model_dir)
    with (fold_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    write_jsonl(train_records, fold_dir / "train_records.jsonl")
    write_jsonl(val_records, fold_dir / "val_records.jsonl")
    return metrics


def aggregate_metrics(fold_metrics: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    diff_f1 = [float(metrics.get("difficulty_macro_f1", 0.0)) for metrics in fold_metrics]
    evidence_f1 = [float(metrics.get("evidence_f1", 0.0)) for metrics in fold_metrics]
    exact = [float(metrics.get("evidence_exact_match", 0.0)) for metrics in fold_metrics]
    return {
        "model_name": args.model_name,
        "n_folds": len(fold_metrics),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_evidence": args.lambda_evidence,
        "evidence_pos_weight": args.evidence_pos_weight,
        "split_strategy": args.split_strategy,
        "mean_difficulty_macro_f1": float(np.mean(diff_f1)),
        "std_difficulty_macro_f1": float(np.std(diff_f1)),
        "mean_evidence_f1": float(np.mean(evidence_f1)),
        "std_evidence_f1": float(np.std(evidence_f1)),
        "mean_evidence_exact_match": float(np.mean(exact)),
        "fold_metrics": fold_metrics,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    set_seed(args.seed)

    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args)
    if not records:
        raise SystemExit("ERROR: no usable training records were loaded.")

    n_folds = maybe_reduce_folds(records, args.n_folds)
    args.n_folds = n_folds

    run_summary = summarize_records(records)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    write_jsonl(records, output_dir / "normalized_records.jsonl")
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    folds = create_stratified_folds(
        records,
        n_folds=args.n_folds,
        seed=args.seed,
        group_by_story=args.split_strategy == "story",
    )
    fold_metrics: list[dict[str, Any]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]
        fold_metrics.append(train_one_fold(fold_idx, train_records, val_records, args))

    summary = aggregate_metrics(fold_metrics, args)
    summary["record_summary"] = run_summary
    summary["elapsed_seconds"] = time.time() - started
    with (output_dir / "cv_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nTraining complete")
    print(f"Output: {output_dir}")
    print(f"Mean difficulty macro-F1: {summary['mean_difficulty_macro_f1']:.4f}")
    print(f"Mean evidence F1: {summary['mean_evidence_f1']:.4f}")


if __name__ == "__main__":
    main()
