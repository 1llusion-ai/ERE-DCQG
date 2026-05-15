"""Multi-task DeBERTa-v3-base classifier for difficulty + evidence identification.

Architecture
------------
DeBERTa-v3-base backbone (184M params)
  -> [CLS] hidden state -> Linear(768, 3) -> difficulty logits  (Easy / Medium / Hard)
  -> [Sn] hidden states  -> Linear(768, 1) -> evidence logits   (binary per marker)

Loss: L_diff + lambda * L_evidence

All heavy imports (torch, transformers, sklearn) are deferred to method-call time
so the module can be imported in non-GPU environments for type checking.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants (importable without torch)
# ---------------------------------------------------------------------------
MAX_MARKERS_DEFAULT = 31
MARKER_TOKENS: list[str] = [f"[S{i}]" for i in range(MAX_MARKERS_DEFAULT)]
DIFFICULTY_LABELS: list[str] = ["Easy", "Medium", "Hard"]
LABEL2ID: dict[str, int] = {lbl: i for i, lbl in enumerate(DIFFICULTY_LABELS)}
ID2LABEL: dict[int, str] = {i: lbl for i, lbl in enumerate(DIFFICULTY_LABELS)}

_CONFIG_FILE = "config.json"
_WEIGHTS_FILE = "model.pt"
_TOKENIZER_DIR = "tokenizer"


# ── internal nn.Module (built lazily) ──────────────────────────────────────

def _build_model_class():
    """Return the ``_MultiTaskModel`` class.

    Defers ``import torch.nn`` so the enclosing module stays importable
    without torch installed.
    """
    import torch.nn as nn

    class _MultiTaskModel(nn.Module):
        """Two-head model: difficulty classification + evidence tagging."""

        def __init__(self, backbone: nn.Module, num_classes: int, max_markers: int):
            super().__init__()
            self.backbone = backbone
            hidden_size: int = backbone.config.hidden_size
            self.difficulty_head = nn.Linear(hidden_size, num_classes)
            self.evidence_head = nn.Linear(hidden_size, 1)
            self.max_markers = max_markers

        def forward(
            self,
            input_ids,
            attention_mask,
            marker_positions=None,
            marker_mask=None,
        ):
            """Forward pass returning difficulty logits and optional evidence logits.

            Parameters
            ----------
            input_ids : Tensor[B, T]
            attention_mask : Tensor[B, T]
            marker_positions : Tensor[B, M] | None
                Token-level indices of ``[Sn]`` markers.  Padded slots use 0.
            marker_mask : Tensor[B, M] | None
                1.0 for real markers, 0.0 for padding.

            Returns
            -------
            diff_logits : Tensor[B, num_classes]
            evidence_logits : Tensor[B, M] | None
            """
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state  # [B, T, H]

            # ── difficulty from [CLS] token ──
            cls_hidden = hidden[:, 0]  # [B, H]
            diff_logits = self.difficulty_head(cls_hidden)  # [B, num_classes]

            # ── evidence from [Sn] marker positions ──
            evidence_logits = None
            if marker_positions is not None:
                B, M = marker_positions.shape
                H = hidden.size(-1)
                safe_pos = marker_positions.clamp(0, hidden.size(1) - 1)
                idx = safe_pos.unsqueeze(-1).expand(B, M, H)
                marker_hidden = hidden.gather(1, idx)  # [B, M, H]
                evidence_logits = self.evidence_head(marker_hidden).squeeze(-1)  # [B, M]

            return diff_logits, evidence_logits

    return _MultiTaskModel


# ── public wrapper class ───────────────────────────────────────────────────

class MultiTaskDifficultyClassifier:
    """High-level training and prediction API for the multi-task DeBERTa model.

    Usage
    -----
    >>> clf = MultiTaskDifficultyClassifier()
    >>> results = clf.train_fold(train_ds, val_ds, epochs=10)
    >>> preds = clf.predict(["[S0] Once upon a time ... [S1] The end."])
    >>> probs = clf.predict_difficulty_probs("[S0] Once upon a time ...")
    """

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        lambda_evidence: float = 0.3,
        num_classes: int = 3,
        max_markers: int = MAX_MARKERS_DEFAULT,
        device: str = "cuda",
    ):
        """Initialize model, tokenizer, and classification heads.

        Adds ``[S0]`` through ``[S{max_markers-1}]`` as special tokens and
        resizes the backbone embeddings accordingly.
        """
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.lambda_evidence = lambda_evidence
        self.num_classes = num_classes
        self.max_markers = max_markers

        # Resolve device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")

        # ── tokenizer with [Sn] marker tokens ──
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        marker_tokens = [f"[S{i}]" for i in range(max_markers)]
        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": marker_tokens}
        )
        logger.info("Added %d special marker tokens to tokenizer", num_added)

        self._marker_token_ids: dict[str, int] = {
            tok: self.tokenizer.convert_tokens_to_ids(tok) for tok in marker_tokens
        }

        # ── backbone + two heads ──
        backbone = AutoModel.from_pretrained(model_name)
        backbone.resize_token_embeddings(len(self.tokenizer))

        _MultiTaskModel = _build_model_class()
        self.model = _MultiTaskModel(backbone, num_classes, max_markers)
        self.model.to(self.device)

        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(
            "MultiTaskDifficultyClassifier ready  device=%s  backbone=%s  params=%.1fM",
            self.device,
            model_name,
            param_count,
        )

    # ------------------------------------------------------------- training

    def train_fold(
        self,
        train_dataset,
        val_dataset,
        epochs: int = 10,
        batch_size: int = 16,
        lr: float = 2e-5,
        warmup_ratio: float = 0.1,
        patience: int = 3,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Train one fold with early stopping on validation macro-F1.

        Parameters
        ----------
        train_dataset, val_dataset
            Sequence-like of dicts, each with keys:

            - ``text``             : str  -- input with ``[Sn]`` sentence markers
            - ``label``            : int  -- difficulty label (0 / 1 / 2)
            - ``evidence_labels``  : list[int]  -- binary labels aligned to markers

        Returns
        -------
        dict
            ``best_epoch``, ``best_val_macro_f1``, ``val_per_class_f1``,
            ``val_evidence_f1``, ``train_loss_history``, ``val_loss_history``.
        """
        import copy

        import torch
        import torch.nn as nn
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup

        torch.manual_seed(seed)

        # ── data loaders ──
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

        # ── optimizer + linear-warmup scheduler ──
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        diff_criterion = nn.CrossEntropyLoss()
        ev_criterion = nn.BCEWithLogitsLoss(reduction="none")

        # ── bookkeeping ──
        best_val_f1 = -1.0
        best_epoch = -1
        best_state: dict | None = None
        epochs_no_improve = 0
        train_loss_history: list[float] = []
        val_loss_history: list[float] = []
        best_report: dict[str, Any] = {}
        best_evidence_f1: float = 0.0

        for epoch in range(epochs):
            # ── train one epoch ──
            self.model.train()
            running_loss = 0.0
            n_batches = 0

            for batch_data in train_loader:
                optimizer.zero_grad()
                loss = self._forward_loss(batch_data, diff_criterion, ev_criterion)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()
                n_batches += 1

            avg_train_loss = running_loss / max(n_batches, 1)
            train_loss_history.append(avg_train_loss)

            # ── validate ──
            val_loss, val_macro_f1, per_class, ev_f1 = self._evaluate(
                val_loader, diff_criterion, ev_criterion
            )
            val_loss_history.append(val_loss)

            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  "
                "val_macro_f1=%.4f  val_evidence_f1=%.4f",
                epoch + 1,
                epochs,
                avg_train_loss,
                val_loss,
                val_macro_f1,
                ev_f1,
            )

            # ── early stopping ──
            if val_macro_f1 > best_val_f1:
                best_val_f1 = val_macro_f1
                best_epoch = epoch + 1
                best_state = copy.deepcopy(self.model.state_dict())
                best_report = per_class
                best_evidence_f1 = ev_f1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d)",
                        epoch + 1,
                        patience,
                    )
                    break

        # ── restore best checkpoint ──
        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info(
                "Restored best model from epoch %d (macro-F1=%.4f)",
                best_epoch,
                best_val_f1,
            )

        return {
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "val_per_class_f1": best_report,
            "val_evidence_f1": best_evidence_f1,
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
        }

    # ----------------------------------------------------------- prediction

    def predict(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """Predict difficulty and evidence for texts containing ``[Sn]`` markers.

        Parameters
        ----------
        texts : list[str]
        batch_size : int

        Returns
        -------
        list[dict]
            Each dict has:

            - ``predicted_difficulty`` : str  (``"Easy"`` / ``"Medium"`` / ``"Hard"``)
            - ``difficulty_probs``     : list[float]  ``[P(Easy), P(Medium), P(Hard)]``
            - ``evidence_sentences``   : list[int]   indices where sigmoid > 0.5
            - ``evidence_probs``       : dict[int, float]  ``{sent_idx: prob}``
        """
        import torch

        self.model.eval()
        all_results: list[dict] = []

        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            batch_data = self._prepare_inference_batch(chunk)

            with torch.no_grad():
                diff_logits, ev_logits = self.model(
                    input_ids=batch_data["input_ids"],
                    attention_mask=batch_data["attention_mask"],
                    marker_positions=batch_data["marker_positions"],
                    marker_mask=batch_data["marker_mask"],
                )

            diff_probs = torch.softmax(diff_logits, dim=-1).cpu()  # [B, 3]
            ev_probs_all = None
            if ev_logits is not None:
                ev_probs_all = torch.sigmoid(ev_logits).cpu()  # [B, M]

            marker_mask_cpu = batch_data["marker_mask"].cpu()
            marker_indices = batch_data["marker_indices"]  # list[list[int]]

            for i in range(len(chunk)):
                probs = diff_probs[i].tolist()
                pred_id = int(diff_probs[i].argmax())
                result: dict[str, Any] = {
                    "predicted_difficulty": ID2LABEL[pred_id],
                    "difficulty_probs": probs,
                    "evidence_sentences": [],
                    "evidence_probs": {},
                }

                if ev_probs_all is not None:
                    indices = marker_indices[i]
                    for j, sent_idx in enumerate(indices):
                        if marker_mask_cpu[i, j] > 0:
                            prob = float(ev_probs_all[i, j])
                            result["evidence_probs"][sent_idx] = prob
                            if prob > 0.5:
                                result["evidence_sentences"].append(sent_idx)

                all_results.append(result)

        return all_results

    def predict_difficulty_probs(self, text: str) -> list[float]:
        """Return ``[P(Easy), P(Medium), P(Hard)]`` for a single text.

        This is the primary interface for the reranker and other downstream
        consumers that need a quick difficulty distribution.
        """
        results = self.predict([text], batch_size=1)
        return results[0]["difficulty_probs"]

    # ----------------------------------------------------------- save/load

    def save(self, path: str) -> None:
        """Persist model weights, tokenizer, and config to *path* directory."""
        import torch

        os.makedirs(path, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(path, _WEIGHTS_FILE))

        tok_dir = os.path.join(path, _TOKENIZER_DIR)
        self.tokenizer.save_pretrained(tok_dir)

        cfg = {
            "model_name": self.model_name,
            "lambda_evidence": self.lambda_evidence,
            "num_classes": self.num_classes,
            "max_markers": self.max_markers,
        }
        with open(os.path.join(path, _CONFIG_FILE), "w") as f:
            json.dump(cfg, f, indent=2)

        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "MultiTaskDifficultyClassifier":
        """Load a previously saved model from *path* directory."""
        import torch
        from transformers import AutoTokenizer

        with open(os.path.join(path, _CONFIG_FILE)) as f:
            cfg = json.load(f)

        # Instantiate with the original backbone name (downloads config only;
        # random head weights are overwritten by the checkpoint below).
        instance = cls(
            model_name=cfg["model_name"],
            lambda_evidence=cfg["lambda_evidence"],
            num_classes=cfg["num_classes"],
            max_markers=cfg["max_markers"],
            device=device,
        )

        # Overwrite tokenizer with the saved copy (preserves marker tokens
        # even if the upstream HuggingFace tokenizer changes).
        tok_dir = os.path.join(path, _TOKENIZER_DIR)
        instance.tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        instance._marker_token_ids = {
            f"[S{i}]": instance.tokenizer.convert_tokens_to_ids(f"[S{i}]")
            for i in range(instance.max_markers)
        }

        # Load weights
        weights_path = os.path.join(path, _WEIGHTS_FILE)
        state = torch.load(weights_path, map_location=instance.device, weights_only=True)
        instance.model.load_state_dict(state)
        instance.model.eval()

        logger.info("Model loaded from %s onto %s", path, instance.device)
        return instance

    # ====================================================================
    #  Private helpers
    # ====================================================================

    # ── collation (training) ──────────────────────────────────────────────

    def _collate_fn(self, batch: list[dict]) -> dict:
        """Tokenize and pad a list of dataset items into a model-ready batch.

        Expected item keys: ``text``, ``label``, ``evidence_labels``.
        """
        import torch

        texts = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"]  # [B, T]
        attention_mask = encoding["attention_mask"]
        B = input_ids.size(0)
        M = self.max_markers

        marker_positions = torch.zeros(B, M, dtype=torch.long)
        marker_mask = torch.zeros(B, M, dtype=torch.float)
        evidence_labels_t = torch.zeros(B, M, dtype=torch.float)

        for i in range(B):
            ids_i = input_ids[i].tolist()
            ev_labels = batch[i].get("evidence_labels", [])
            slot = 0
            for s_idx in range(M):
                tok_id = self._marker_token_ids.get(f"[S{s_idx}]")
                if tok_id is None:
                    continue
                try:
                    pos = ids_i.index(tok_id)
                except ValueError:
                    continue  # marker not present in this sample
                marker_positions[i, slot] = pos
                marker_mask[i, slot] = 1.0
                if slot < len(ev_labels):
                    evidence_labels_t[i, slot] = float(ev_labels[slot])
                slot += 1

        dev = self.device
        return {
            "input_ids": input_ids.to(dev),
            "attention_mask": attention_mask.to(dev),
            "labels": labels.to(dev),
            "marker_positions": marker_positions.to(dev),
            "marker_mask": marker_mask.to(dev),
            "evidence_labels": evidence_labels_t.to(dev),
        }

    # ── collation (inference) ────────────────────────────────────────────

    def _prepare_inference_batch(self, texts: list[str]) -> dict:
        """Tokenize texts for inference and locate marker positions.

        Returns a model-ready dict plus ``marker_indices``
        (``list[list[int]]``) mapping each slot back to its sentence index.
        """
        import torch

        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"]  # [B, T]
        attention_mask = encoding["attention_mask"]
        B = input_ids.size(0)
        M = self.max_markers

        marker_positions = torch.zeros(B, M, dtype=torch.long)
        marker_mask = torch.zeros(B, M, dtype=torch.float)
        marker_indices: list[list[int]] = []

        for i in range(B):
            ids_i = input_ids[i].tolist()
            indices_i: list[int] = []
            slot = 0
            for s_idx in range(M):
                tok_id = self._marker_token_ids.get(f"[S{s_idx}]")
                if tok_id is None:
                    continue
                try:
                    pos = ids_i.index(tok_id)
                except ValueError:
                    continue
                marker_positions[i, slot] = pos
                marker_mask[i, slot] = 1.0
                indices_i.append(s_idx)
                slot += 1
            marker_indices.append(indices_i)

        dev = self.device
        return {
            "input_ids": input_ids.to(dev),
            "attention_mask": attention_mask.to(dev),
            "marker_positions": marker_positions.to(dev),
            "marker_mask": marker_mask.to(dev),
            "marker_indices": marker_indices,
        }

    # ── loss helpers ─────────────────────────────────────────────────────

    def _forward_loss(self, batch_data: dict, diff_criterion, ev_criterion):
        """Run a forward pass and return the combined multi-task loss."""
        diff_logits, ev_logits = self.model(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            marker_positions=batch_data["marker_positions"],
            marker_mask=batch_data["marker_mask"],
        )
        return self._combined_loss(
            diff_logits, ev_logits, batch_data, diff_criterion, ev_criterion
        )

    def _combined_loss(
        self, diff_logits, ev_logits, batch_data, diff_criterion, ev_criterion
    ):
        """Compute ``L_diff + lambda * L_evidence`` from pre-computed logits."""
        diff_loss = diff_criterion(diff_logits, batch_data["labels"])
        total_loss = diff_loss

        if ev_logits is not None and batch_data.get("marker_mask") is not None:
            mask = batch_data["marker_mask"]           # [B, M]
            ev_labels = batch_data["evidence_labels"]  # [B, M]
            ev_loss_raw = ev_criterion(ev_logits, ev_labels)  # [B, M]
            ev_loss = (ev_loss_raw * mask).sum() / mask.sum().clamp(min=1)
            total_loss = diff_loss + self.lambda_evidence * ev_loss

        return total_loss

    # ── evaluation ───────────────────────────────────────────────────────

    def _evaluate(self, val_loader, diff_criterion, ev_criterion):
        """Evaluate on *val_loader* under ``torch.no_grad()``.

        Returns ``(val_loss, macro_f1, per_class_report, evidence_f1)``.
        """
        import torch
        from sklearn.metrics import classification_report, f1_score

        self.model.eval()

        all_diff_true: list[int] = []
        all_diff_pred: list[int] = []
        all_ev_true: list[int] = []
        all_ev_pred: list[int] = []
        running_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                diff_logits, ev_logits = self.model(
                    input_ids=batch_data["input_ids"],
                    attention_mask=batch_data["attention_mask"],
                    marker_positions=batch_data["marker_positions"],
                    marker_mask=batch_data["marker_mask"],
                )

                loss = self._combined_loss(
                    diff_logits, ev_logits, batch_data, diff_criterion, ev_criterion
                )
                running_loss += loss.item()
                n_batches += 1

                # ── difficulty predictions ──
                preds = diff_logits.argmax(dim=-1).cpu().tolist()
                trues = batch_data["labels"].cpu().tolist()
                all_diff_true.extend(trues)
                all_diff_pred.extend(preds)

                # ── evidence predictions ──
                if ev_logits is not None:
                    mask = batch_data["marker_mask"].cpu()
                    ev_preds = (torch.sigmoid(ev_logits).cpu() > 0.5).long()
                    ev_trues = batch_data["evidence_labels"].cpu().long()
                    for b in range(mask.size(0)):
                        for m in range(mask.size(1)):
                            if mask[b, m] > 0:
                                all_ev_true.append(int(ev_trues[b, m]))
                                all_ev_pred.append(int(ev_preds[b, m]))

        val_loss = running_loss / max(n_batches, 1)

        macro_f1 = f1_score(
            all_diff_true, all_diff_pred, average="macro", zero_division=0
        )
        report = classification_report(
            all_diff_true,
            all_diff_pred,
            target_names=DIFFICULTY_LABELS[: self.num_classes],
            output_dict=True,
            zero_division=0,
        )

        evidence_f1 = 0.0
        if all_ev_true:
            evidence_f1 = f1_score(
                all_ev_true, all_ev_pred, average="binary", zero_division=0
            )

        return val_loss, macro_f1, report, evidence_f1
