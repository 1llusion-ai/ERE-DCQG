"""Multi-task DeBERTa classifier for difficulty and evidence sentences."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .data import (
    DEFAULT_MAX_MARKERS,
    DIFFICULTY_LABELS,
    ID2LABEL,
    LABEL2ID,
    MARKER_TOKENS,
    build_classifier_input,
    build_marked_context,
    collate_difficulty_evidence,
)

logger = logging.getLogger(__name__)

_CONFIG_FILE = "config.json"
_WEIGHTS_FILE = "model.pt"
_TOKENIZER_DIR = "tokenizer"


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for MultiTaskDifficultyClassifier. "
            "Install the training environment first."
        ) from exc


def _build_model_class():
    import torch.nn as nn

    class MultiTaskEncoder(nn.Module):
        def __init__(self, backbone: nn.Module, num_labels: int, dropout: float) -> None:
            super().__init__()
            self.backbone = backbone
            hidden_size = int(backbone.config.hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.difficulty_head = nn.Linear(hidden_size, num_labels)
            self.evidence_head = nn.Linear(hidden_size, 1)

        def forward(
            self,
            input_ids,
            attention_mask,
            marker_positions,
        ):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state

            cls_hidden = self.dropout(hidden[:, 0])
            difficulty_logits = self.difficulty_head(cls_hidden)

            batch_size, marker_count = marker_positions.shape
            hidden_size = hidden.size(-1)
            safe_positions = marker_positions.clamp(0, hidden.size(1) - 1)
            gather_index = safe_positions.unsqueeze(-1).expand(
                batch_size,
                marker_count,
                hidden_size,
            )
            marker_hidden = hidden.gather(1, gather_index)
            marker_hidden = self.dropout(marker_hidden)
            evidence_logits = self.evidence_head(marker_hidden).squeeze(-1)
            return difficulty_logits, evidence_logits

    return MultiTaskEncoder


class MultiTaskDifficultyClassifier:
    """High-level training and inference wrapper.

    The model uses one encoder backbone and two task heads:
    - difficulty head over the ``[CLS]`` representation;
    - evidence head over each ``[Sx]`` marker representation.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        *,
        tokenizer: Any | None = None,
        lambda_evidence: float = 0.3,
        evidence_pos_weight: float = 1.0,
        max_markers: int = DEFAULT_MAX_MARKERS,
        dropout: float = 0.1,
        device: str = "cuda",
    ) -> None:
        torch = _torch()
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.lambda_evidence = float(lambda_evidence)
        self.evidence_pos_weight = float(evidence_pos_weight)
        self.max_markers = int(max_markers)
        self.dropout = float(dropout)
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        if device == "cuda" and self.device.type == "cpu":
            logger.warning("CUDA requested but unavailable; falling back to CPU")

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        marker_tokens = [f"[S{i}]" for i in range(self.max_markers)]
        self.tokenizer.add_special_tokens({"additional_special_tokens": marker_tokens})
        self.marker_token_ids = {
            token: self.tokenizer.convert_tokens_to_ids(token) for token in marker_tokens
        }

        backbone = AutoModel.from_pretrained(model_name)
        backbone.resize_token_embeddings(len(self.tokenizer))
        model_cls = _build_model_class()
        self.model = model_cls(backbone, len(DIFFICULTY_LABELS), self.dropout)
        self.model.to(self.device)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def train_fold(
        self,
        train_dataset,
        val_dataset=None,
        *,
        epochs: int = 3,
        batch_size: int = 8,
        lr: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        patience: int | None = 2,
        seed: int = 42,
        grad_clip: float = 1.0,
    ) -> dict[str, Any]:
        import copy

        import torch
        import torch.nn as nn
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup

        torch.manual_seed(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_difficulty_evidence,
        )
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_difficulty_evidence,
            )

        total_steps = max(1, len(train_loader) * epochs)
        warmup_steps = int(total_steps * warmup_ratio)
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        difficulty_loss_fn = nn.CrossEntropyLoss()
        evidence_loss_fn = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=torch.tensor(self.evidence_pos_weight, device=self.device),
        )

        best_score = -1.0
        best_epoch = 0
        best_state = None
        best_metrics: dict[str, Any] = {}
        epochs_without_improvement = 0
        train_loss_history: list[float] = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            batch_count = 0
            for batch in train_loader:
                batch = self._move_batch(batch)
                optimizer.zero_grad(set_to_none=True)
                difficulty_logits, evidence_logits = self.model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["marker_positions"],
                )
                loss = self._combined_loss(
                    difficulty_logits,
                    evidence_logits,
                    batch,
                    difficulty_loss_fn,
                    evidence_loss_fn,
                )
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                running_loss += float(loss.item())
                batch_count += 1

            train_loss = running_loss / max(batch_count, 1)
            train_loss_history.append(train_loss)

            if val_loader is None:
                logger.info("epoch=%d train_loss=%.4f", epoch, train_loss)
                best_epoch = epoch
                best_metrics = {
                    "final_train_loss": train_loss,
                    "train_loss_history": train_loss_history,
                }
                continue

            val_metrics = self.evaluate(val_loader, difficulty_loss_fn, evidence_loss_fn)
            score = float(val_metrics.get("difficulty_macro_f1", 0.0))
            logger.info(
                "epoch=%d train_loss=%.4f val_loss=%.4f diff_macro_f1=%.4f evidence_f1=%.4f",
                epoch,
                train_loss,
                val_metrics.get("val_loss", 0.0),
                score,
                val_metrics.get("evidence_f1", 0.0),
            )

            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                best_metrics = dict(val_metrics)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if patience is not None and epochs_without_improvement >= patience:
                    logger.info("early stopping at epoch %d", epoch)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        best_metrics.update(
            {
                "best_epoch": best_epoch,
                "train_loss_history": train_loss_history,
            }
        )
        return best_metrics

    def evaluate(self, dataloader, difficulty_loss_fn=None, evidence_loss_fn=None) -> dict[str, Any]:
        import torch
        import torch.nn as nn
        from sklearn.metrics import accuracy_score, classification_report, f1_score

        if difficulty_loss_fn is None:
            difficulty_loss_fn = nn.CrossEntropyLoss()
        if evidence_loss_fn is None:
            evidence_loss_fn = nn.BCEWithLogitsLoss(
                reduction="none",
                pos_weight=torch.tensor(self.evidence_pos_weight, device=self.device),
            )

        self.model.eval()
        losses: list[float] = []
        difficulty_true: list[int] = []
        difficulty_pred: list[int] = []
        evidence_true: list[int] = []
        evidence_pred: list[int] = []
        evidence_exact: list[int] = []

        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch(batch)
                difficulty_logits, evidence_logits = self.model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["marker_positions"],
                )
                loss = self._combined_loss(
                    difficulty_logits,
                    evidence_logits,
                    batch,
                    difficulty_loss_fn,
                    evidence_loss_fn,
                )
                losses.append(float(loss.item()))

                difficulty_true.extend(batch["difficulty_label"].cpu().tolist())
                difficulty_pred.extend(difficulty_logits.argmax(dim=-1).cpu().tolist())

                probs = torch.sigmoid(evidence_logits)
                preds = (probs > 0.5).long()
                labels = batch["evidence_labels"].long()
                mask = batch["marker_mask"].long()
                for row_idx in range(mask.size(0)):
                    row_true: list[int] = []
                    row_pred: list[int] = []
                    for marker_idx in range(mask.size(1)):
                        if int(mask[row_idx, marker_idx]) == 0:
                            continue
                        true_value = int(labels[row_idx, marker_idx])
                        pred_value = int(preds[row_idx, marker_idx])
                        evidence_true.append(true_value)
                        evidence_pred.append(pred_value)
                        if true_value:
                            row_true.append(marker_idx)
                        if pred_value:
                            row_pred.append(marker_idx)
                    evidence_exact.append(1 if row_true == row_pred else 0)

        report = classification_report(
            difficulty_true,
            difficulty_pred,
            labels=list(range(len(DIFFICULTY_LABELS))),
            target_names=DIFFICULTY_LABELS,
            output_dict=True,
            zero_division=0,
        )
        metrics = {
            "val_loss": sum(losses) / max(len(losses), 1),
            "difficulty_accuracy": accuracy_score(difficulty_true, difficulty_pred),
            "difficulty_macro_f1": f1_score(
                difficulty_true,
                difficulty_pred,
                labels=list(range(len(DIFFICULTY_LABELS))),
                average="macro",
                zero_division=0,
            ),
            "difficulty_per_class": {
                label: report[label]["f1-score"] for label in DIFFICULTY_LABELS
            },
            "evidence_f1": f1_score(
                evidence_true,
                evidence_pred,
                average="binary",
                zero_division=0,
            ) if evidence_true else 0.0,
            "evidence_exact_match": sum(evidence_exact) / max(len(evidence_exact), 1),
        }
        return metrics

    def predict(self, texts: list[str], batch_size: int = 16) -> list[dict[str, Any]]:
        import torch
        from torch.utils.data import DataLoader

        dataset = _InferenceDataset(
            texts,
            self.tokenizer,
            self.marker_token_ids,
            max_markers=self.max_markers,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_inference)
        self.model.eval()
        results: list[dict[str, Any]] = []
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch(batch)
                difficulty_logits, evidence_logits = self.model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["marker_positions"],
                )
                difficulty_probs = torch.softmax(difficulty_logits, dim=-1).cpu()
                evidence_probs = torch.sigmoid(evidence_logits).cpu()
                marker_mask = batch["marker_mask"].cpu()
                for row_idx in range(difficulty_probs.size(0)):
                    probs = difficulty_probs[row_idx].tolist()
                    pred_id = int(difficulty_probs[row_idx].argmax())
                    evidence: list[int] = []
                    evidence_prob_map: dict[int, float] = {}
                    for marker_idx in range(marker_mask.size(1)):
                        if int(marker_mask[row_idx, marker_idx]) == 0:
                            continue
                        prob = float(evidence_probs[row_idx, marker_idx])
                        evidence_prob_map[marker_idx] = prob
                        if prob > 0.5:
                            evidence.append(marker_idx)
                    results.append(
                        {
                            "predicted_difficulty": ID2LABEL[pred_id],
                            "difficulty_probs": probs,
                            "evidence_sentences": evidence,
                            "evidence_probs": evidence_prob_map,
                        }
                    )
        return results

    def predict_records(self, records: list[dict[str, Any]], batch_size: int = 16) -> list[dict[str, Any]]:
        texts = []
        for record in records:
            sentence_rows = [(int(item["id"]), str(item["text"])) for item in record["sentences"]]
            texts.append(
                build_classifier_input(
                    record["question"],
                    record["answer"],
                    build_marked_context(sentence_rows),
                )
            )
        return self.predict(texts, batch_size=batch_size)

    def predict_difficulty_probs(self, text: str) -> list[float]:
        return self.predict([text], batch_size=1)[0]["difficulty_probs"]

    def save(self, path: str | Path) -> None:
        torch = _torch()
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), output_dir / _WEIGHTS_FILE)
        tokenizer_dir = output_dir / _TOKENIZER_DIR
        self.tokenizer.save_pretrained(tokenizer_dir)
        config = {
            "model_name": self.model_name,
            "lambda_evidence": self.lambda_evidence,
            "evidence_pos_weight": self.evidence_pos_weight,
            "max_markers": self.max_markers,
            "dropout": self.dropout,
        }
        with (output_dir / _CONFIG_FILE).open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda") -> "MultiTaskDifficultyClassifier":
        torch = _torch()
        model_dir = Path(path)
        with (model_dir / _CONFIG_FILE).open(encoding="utf-8") as f:
            config = json.load(f)
        instance = cls(
            model_name=config["model_name"],
            lambda_evidence=config.get("lambda_evidence", 0.3),
            evidence_pos_weight=config.get("evidence_pos_weight", 1.0),
            max_markers=config.get("max_markers", DEFAULT_MAX_MARKERS),
            dropout=config.get("dropout", 0.1),
            device=device,
        )
        tokenizer_path = model_dir / _TOKENIZER_DIR
        from transformers import AutoTokenizer

        instance.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        instance.marker_token_ids = {
            f"[S{i}]": instance.tokenizer.convert_tokens_to_ids(f"[S{i}]")
            for i in range(instance.max_markers)
        }
        state = torch.load(model_dir / _WEIGHTS_FILE, map_location=instance.device)
        instance.model.load_state_dict(state)
        instance.model.eval()
        return instance

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved = {}
        for key, value in batch.items():
            if hasattr(value, "to"):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _combined_loss(
        self,
        difficulty_logits,
        evidence_logits,
        batch: dict[str, Any],
        difficulty_loss_fn,
        evidence_loss_fn,
    ):
        difficulty_loss = difficulty_loss_fn(difficulty_logits, batch["difficulty_label"])
        evidence_loss_raw = evidence_loss_fn(evidence_logits, batch["evidence_labels"])
        evidence_mask = batch["marker_mask"]
        evidence_loss = (evidence_loss_raw * evidence_mask).sum() / evidence_mask.sum().clamp(min=1.0)
        return difficulty_loss + self.lambda_evidence * evidence_loss


class _InferenceDataset:
    def __init__(
        self,
        texts: list[str],
        tokenizer: Any,
        marker_token_ids: dict[str, int],
        *,
        max_length: int = 512,
        max_markers: int = DEFAULT_MAX_MARKERS,
    ) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.marker_token_ids = marker_token_ids
        self.max_length = max_length
        self.max_markers = max_markers

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        torch = _torch()
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        input_id_list = input_ids.tolist()
        marker_positions = torch.zeros(self.max_markers, dtype=torch.long)
        marker_mask = torch.zeros(self.max_markers, dtype=torch.float32)
        for marker_idx in range(self.max_markers):
            marker_id = self.marker_token_ids.get(f"[S{marker_idx}]")
            if marker_id is None:
                continue
            try:
                position = input_id_list.index(marker_id)
            except ValueError:
                continue
            marker_positions[marker_idx] = position
            marker_mask[marker_idx] = 1.0
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "marker_positions": marker_positions,
            "marker_mask": marker_mask,
        }


def _collate_inference(batch: list[dict[str, Any]]) -> dict[str, Any]:
    torch = _torch()
    return {
        key: torch.stack([item[key] for item in batch])
        for key in ["input_ids", "attention_mask", "marker_positions", "marker_mask"]
    }
