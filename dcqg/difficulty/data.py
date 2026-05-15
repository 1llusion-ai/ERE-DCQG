"""Training data construction for multi-task DeBERTa classifier.

Prepares FairytaleQA training data with [S0]-[S30] sentence markers,
evidence-focused truncation, and a PyTorch Dataset class for joint
difficulty classification + evidence sentence detection.

Design note: torch is imported lazily so that pure-data helpers
(add_sentence_markers, load_training_labels, etc.) work in environments
where PyTorch is not installed.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from dcqg.path.fairytale_evidence_audit import _split_sentences
from dcqg.utils.jsonl import read_jsonl

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MARKER_TOKENS: list[str] = [f"[S{i}]" for i in range(31)]
"""Special tokens added to the DeBERTa tokenizer (max 31 sentences)."""

_MAX_MARKERS = len(MARKER_TOKENS)  # 31


# ---------------------------------------------------------------------------
# Deferred torch import
# ---------------------------------------------------------------------------

def _get_torch():
    """Import and return the ``torch`` module.

    Raises ``ImportError`` with a helpful message when torch is absent.
    """
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for DifficultyEvidenceDataset. "
            "Install it with: pip install torch"
        ) from exc


# ---------------------------------------------------------------------------
# Sentence markers
# ---------------------------------------------------------------------------

def add_sentence_markers(text: str) -> tuple[str, list[str]]:
    """Split *text* into sentences and prepend ``[Sn]`` markers.

    Returns
    -------
    marked_text : str
        ``"[S0] First sentence. [S1] Second sentence. ..."``
    sentences : list[str]
        ``["First sentence.", "Second sentence.", ...]``

    At most :pydata:`_MAX_MARKERS` (31) sentences are kept; any beyond that
    are silently discarded so marker indices stay in range.
    """
    sentences = _split_sentences(text)
    sentences = sentences[:_MAX_MARKERS]

    parts: list[str] = []
    for idx, sent in enumerate(sentences):
        parts.append(f"{MARKER_TOKENS[idx]} {sent}")

    marked_text = " ".join(parts)
    return marked_text, sentences


# ---------------------------------------------------------------------------
# Evidence-focused truncation
# ---------------------------------------------------------------------------

def evidence_focused_truncation(
    sentences: list[str],
    answer_sentence_ids: list[int],
    evidence_sentence_ids: list[int],
    bridge_sentence_ids: list[int],
    max_sentences: int = _MAX_MARKERS,
) -> list[int]:
    """Select which sentence indices to keep when *len(sentences)* exceeds *max_sentences*.

    Priority order (higher priority sentences are always included first):

    1. **Answer sentences** -- indices in *answer_sentence_ids*.
    2. **Required evidence** -- indices in *evidence_sentence_ids* that are not
       already in the answer set.
    3. **Bridge sentences** -- indices in *bridge_sentence_ids* not already
       selected.
    4. **Neighbours** -- indices within +/-2 of any already-selected index.
    5. **Rest** -- remaining indices in document order.

    Returns a **sorted** list of at most *max_sentences* indices.  If the
    input already has <= *max_sentences* sentences, all indices are returned.
    """
    n = len(sentences)

    if n <= max_sentences:
        return list(range(n))

    valid = set(range(n))

    # Deduplicate and clamp to valid range.
    answer_set = sorted(set(answer_sentence_ids) & valid)
    evidence_set = sorted(set(evidence_sentence_ids) & valid)
    bridge_set = sorted(set(bridge_sentence_ids) & valid)

    selected: list[int] = []
    selected_set: set[int] = set()

    def _add(indices: list[int]) -> None:
        for idx in indices:
            if idx not in selected_set and len(selected) < max_sentences:
                selected.append(idx)
                selected_set.add(idx)

    # Priority 1: answer sentences
    _add(answer_set)

    # Priority 2: required evidence (excluding already-added answer ids)
    _add(evidence_set)

    # Priority 3: bridge sentences
    _add(bridge_set)

    # Priority 4: neighbours (+/-2) of everything selected so far
    if len(selected) < max_sentences:
        neighbour_candidates: list[int] = []
        for idx in list(selected):  # snapshot
            for delta in (-2, -1, 1, 2):
                nb = idx + delta
                if nb in valid and nb not in selected_set:
                    neighbour_candidates.append(nb)
        # Deduplicate while preserving order
        seen: set[int] = set()
        unique_neighbours: list[int] = []
        for nb in neighbour_candidates:
            if nb not in seen:
                seen.add(nb)
                unique_neighbours.append(nb)
        _add(sorted(unique_neighbours))

    # Priority 5: rest in document order
    if len(selected) < max_sentences:
        rest = [i for i in range(n) if i not in selected_set]
        _add(rest)

    return sorted(selected)


# ---------------------------------------------------------------------------
# JSONL loading and normalisation
# ---------------------------------------------------------------------------

def _normalise_record(rec: dict) -> dict | None:
    """Normalise a single training record.

    Returns ``None`` when essential fields are missing.
    """
    story_section = rec.get("story_section", "")
    if not story_section:
        return None

    # Difficulty label -- accept both field names used by different stages.
    difficulty = rec.get("difficulty_label") or rec.get("final_difficulty") or ""
    if difficulty not in ("Easy", "Medium", "Hard"):
        return None

    question = rec.get("question", "")
    answer1 = rec.get("answer1", "")
    if not question or not answer1:
        return None

    # Evidence / bridge lists -- may be absent in some data variants.
    required = rec.get("required_evidence_sentences", [])
    if not isinstance(required, list):
        required = []
    required = [int(x) for x in required if isinstance(x, (int, float))]

    bridge = rec.get("bridge_sentence_ids", [])
    if not isinstance(bridge, list):
        bridge = []
    bridge = [int(x) for x in bridge if isinstance(x, (int, float))]

    return {
        "story_section": story_section,
        "difficulty_label": difficulty,
        "required_evidence_sentences": required,
        "bridge_sentence_ids": bridge,
        "question": question,
        "answer1": answer1,
        # Preserve optional metadata that callers may use.
        "story_name": rec.get("story_name", ""),
        "answer2": rec.get("answer2", ""),
        "local_or_sum": rec.get("local_or_sum", ""),
        "attribute": rec.get("attribute", ""),
        "ex_or_im": rec.get("ex_or_im", ""),
    }


def load_training_labels(
    implicit_path: str,
    explicit_path: str | None = None,
) -> list[dict]:
    """Load and merge training labels from one or two JSONL files.

    Parameters
    ----------
    implicit_path : str
        Path to the primary (implicit) training labels JSONL.
    explicit_path : str, optional
        Path to an additional (explicit) labels JSONL to merge.

    Returns
    -------
    list[dict]
        Normalised records.  Records with missing essential fields
        (story_section, difficulty_label, question, answer1) are filtered out.
    """
    raw_records: list[dict] = read_jsonl(implicit_path)

    if explicit_path and Path(explicit_path).exists():
        raw_records.extend(read_jsonl(explicit_path))

    results: list[dict] = []
    skipped = 0
    for rec in raw_records:
        normalised = _normalise_record(rec)
        if normalised is not None:
            results.append(normalised)
        else:
            skipped += 1

    if skipped:
        logger.info(
            "load_training_labels: kept %d records, skipped %d incomplete",
            len(results),
            skipped,
        )

    return results


# ---------------------------------------------------------------------------
# Stratified k-fold
# ---------------------------------------------------------------------------

def create_stratified_folds(
    records: list[dict],
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Create stratified k-fold splits by ``difficulty_label``.

    Uses :class:`sklearn.model_selection.StratifiedKFold`.

    Parameters
    ----------
    records : list[dict]
        Each record must contain a ``difficulty_label`` key
        (one of ``"Easy"``, ``"Medium"``, ``"Hard"``).
    n_folds : int
        Number of folds.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[tuple[list[int], list[int]]]
        ``(train_indices, val_indices)`` per fold.
    """
    from sklearn.model_selection import StratifiedKFold

    labels = [r["difficulty_label"] for r in records]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds: list[tuple[list[int], list[int]]] = []
    # StratifiedKFold.split needs an array-like X; indices are fine.
    dummy_x = list(range(len(records)))
    for train_idx, val_idx in skf.split(dummy_x, labels):
        folds.append((train_idx.tolist(), val_idx.tolist()))

    return folds


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class DifficultyEvidenceDataset:
    """PyTorch-compatible Dataset for multi-task difficulty + evidence classification.

    Each item is a dict with the following tensors:

    * ``input_ids``        -- ``LongTensor [max_length]``
    * ``attention_mask``   -- ``LongTensor [max_length]``
    * ``difficulty_label`` -- ``LongTensor []``  (scalar, 0=Easy 1=Medium 2=Hard)
    * ``evidence_labels``  -- ``FloatTensor [max_markers]``
      (1.0 at evidence sentence positions, 0.0 elsewhere)
    * ``marker_mask``      -- ``FloatTensor [max_markers]``
      (1.0 where a sentence marker exists in the input, 0.0 for padding)

    The tokenizer **must** already have ``[S0]``..``[S30]`` as added special
    tokens (e.g. via ``tokenizer.add_special_tokens({"additional_special_tokens": MARKER_TOKENS})``).
    """

    DIFFICULTY_MAP: dict[str, int] = {"Easy": 0, "Medium": 1, "Hard": 2}

    def __init__(
        self,
        records: list[dict],
        tokenizer: Any,
        max_length: int = 512,
        max_markers: int = _MAX_MARKERS,
    ) -> None:
        torch = _get_torch()  # noqa: F841 -- validates availability

        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_markers = max_markers

        # Pre-compute marker token IDs for fast lookup.
        self._marker_ids: list[int] = [
            tokenizer.convert_tokens_to_ids(tok)
            for tok in MARKER_TOKENS[:max_markers]
        ]

    # -- sequence protocol --------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        torch = _get_torch()
        rec = self.records[idx]

        # 1. Split sentences
        sentences = _split_sentences(rec["story_section"])

        # 2. Truncate to max_markers if necessary
        evidence_ids = rec.get("required_evidence_sentences", [])
        bridge_ids = rec.get("bridge_sentence_ids", [])

        # Determine answer sentence ids -- use the first evidence sentence as
        # the answer sentence when no explicit field is present.
        answer_ids = rec.get("answer_sentence_ids", evidence_ids[:1])

        if len(sentences) > self.max_markers:
            keep = evidence_focused_truncation(
                sentences,
                answer_sentence_ids=answer_ids,
                evidence_sentence_ids=evidence_ids,
                bridge_sentence_ids=bridge_ids,
                max_sentences=self.max_markers,
            )
            # Remap sentence lists after truncation.  ``keep`` is sorted.
            old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep)}
            sentences = [sentences[i] for i in keep]
            evidence_ids = sorted(
                old_to_new[i] for i in evidence_ids if i in old_to_new
            )
            bridge_ids = sorted(
                old_to_new[i] for i in bridge_ids if i in old_to_new
            )

        num_sents = len(sentences)

        # 3. Build marked text and prepend the question
        marked_parts: list[str] = []
        for s_idx, sent in enumerate(sentences):
            marked_parts.append(f"{MARKER_TOKENS[s_idx]} {sent}")
        marked_text = " ".join(marked_parts)

        # Prepend question so the model sees the query context.
        input_text = f"{rec['question']} {self.tokenizer.sep_token} {marked_text}"

        # 4. Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)        # [max_length]
        attention_mask = encoding["attention_mask"].squeeze(0)  # [max_length]

        # 5. Locate marker positions in input_ids
        #    For each marker [Sn] that should appear (n < num_sents), find its
        #    token position in input_ids.
        marker_positions: list[int] = []  # token index for each sentence marker
        input_ids_list = input_ids.tolist()

        for m_idx in range(num_sents):
            m_id = self._marker_ids[m_idx]
            # Find the token position -- should exist unless truncated away.
            try:
                pos = input_ids_list.index(m_id)
            except ValueError:
                pos = -1  # marker was truncated
            marker_positions.append(pos)

        # 6. Build evidence_labels: 1.0 at evidence sentence positions
        evidence_set = set(evidence_ids)
        evidence_labels = torch.zeros(self.max_markers, dtype=torch.float32)
        for s_idx in range(num_sents):
            if s_idx in evidence_set:
                evidence_labels[s_idx] = 1.0

        # 7. Build marker_mask: 1.0 where a valid marker exists in the tokenized input
        marker_mask = torch.zeros(self.max_markers, dtype=torch.float32)
        for s_idx in range(num_sents):
            if marker_positions[s_idx] >= 0:
                marker_mask[s_idx] = 1.0

        # 8. Difficulty label
        diff_str = rec.get("difficulty_label", "Easy")
        diff_label = self.DIFFICULTY_MAP.get(diff_str, 0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "difficulty_label": torch.tensor(diff_label, dtype=torch.long),
            "evidence_labels": evidence_labels,
            "marker_mask": marker_mask,
        }
