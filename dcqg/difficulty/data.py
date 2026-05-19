"""Data utilities for the difficulty/evidence multi-task classifier.

The classifier is trained on QA examples with numbered story-section
sentences.  It predicts:

1. the human difficulty label: Easy / Medium / Hard;
2. the minimal evidence sentence set as binary labels over ``[S0]`` markers.

This module accepts both normalized JSONL records and Label Studio JSON
exports, so the same code can be used during annotation and later on the
server.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

DIFFICULTY_LABELS: list[str] = ["Easy", "Medium", "Hard"]
LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(DIFFICULTY_LABELS)}
ID2LABEL: dict[int, str] = {idx: label for label, idx in LABEL2ID.items()}

DEFAULT_MAX_MARKERS = 64
MARKER_TOKENS: list[str] = [f"[S{i}]" for i in range(DEFAULT_MAX_MARKERS)]

_SENTENCE_LINE_RE = re.compile(r"^\s*\[S(?P<idx>\d+)\]\s*(?P<text>.*?)(?:\s*)$")
_FULL_QA_RE = re.compile(
    r"Full context:\s*(?P<context>.*?)(?:\n\s*\n\s*QA:\s*(?P<qa>.*))?\s*$",
    re.DOTALL | re.IGNORECASE,
)
_QA_RE = re.compile(
    r"Q:\s*(?P<question>.*?)(?:\n|$)\s*A:\s*(?P<answer>.*)\s*$",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class LabelStudioAnnotation:
    """Flat view of one Label Studio annotation result."""

    values: dict[str, Any]
    annotation_id: str | int | None = None


def _get_torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for DifficultyEvidenceDataset. "
            "Install the training environment first."
        ) from exc


def parse_sentence_ids(value: Any) -> list[int]:
    """Parse sentence ids from ``[3, 5]``, ``S3,S5``, lists, or scalars."""
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, list):
        ids: list[int] = []
        for item in value:
            ids.extend(parse_sentence_ids(item))
        return sorted(set(ids))
    text = str(value).strip()
    if not text:
        return []
    ids = [int(match) for match in re.findall(r"\d+", text)]
    return sorted(set(ids))


def parse_numbered_context(text: str) -> list[tuple[int, str]]:
    """Return ``[(sentence_id, sentence_text), ...]`` from ``[Sx]`` lines."""
    rows: list[tuple[int, str]] = []
    for line in str(text or "").splitlines():
        match = _SENTENCE_LINE_RE.match(line)
        if not match:
            continue
        rows.append((int(match.group("idx")), match.group("text").strip()))
    return rows


def strip_context_from_full_context_qa(text: str) -> tuple[str, str]:
    """Split a Label Studio ``full_context_qa`` field into context and QA text."""
    raw = str(text or "").strip()
    match = _FULL_QA_RE.match(raw)
    if not match:
        return raw, ""
    context = (match.group("context") or "").strip()
    qa = (match.group("qa") or "").strip()
    return context, qa


def parse_qa_text(text: str) -> tuple[str, str]:
    """Parse ``Q: ...`` / ``A: ...`` text."""
    match = _QA_RE.search(str(text or "").strip())
    if not match:
        return "", ""
    return match.group("question").strip(), match.group("answer").strip()


def build_marked_context(sentences: list[tuple[int, str]]) -> str:
    return "\n".join(f"[S{idx}] {sent}" for idx, sent in sentences)


def add_sentence_markers(text: str, max_markers: int = DEFAULT_MAX_MARKERS) -> tuple[str, list[str]]:
    """Backward-compatible helper returning marked text and sentence strings."""
    numbered = parse_numbered_context(text)
    if numbered:
        kept = numbered[:max_markers]
        return build_marked_context([(idx, sent) for idx, sent in kept]), [sent for _idx, sent in kept]

    try:
        from dcqg.path.fairytale_evidence_audit import _split_sentences

        sentences = _split_sentences(str(text or ""))
    except Exception:
        sentences = [line.strip() for line in str(text or "").splitlines() if line.strip()]

    sentences = sentences[:max_markers]
    rows = [(idx, sentence) for idx, sentence in enumerate(sentences)]
    return build_marked_context(rows), sentences


def build_classifier_input(question: str, answer: str, marked_context: str) -> str:
    """Build the text consumed by DeBERTa."""
    return (
        f"Question: {question.strip()}\n"
        f"Answer: {answer.strip()}\n"
        "Context:\n"
        f"{marked_context.strip()}"
    ).strip()


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON, JSONL, or CSV records."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".csv":
        return _read_csv(p)
    return _read_json_or_jsonl(p)


def flatten_label_studio_annotation(task: dict[str, Any]) -> LabelStudioAnnotation | None:
    """Return the latest non-cancelled annotation as ``from_name -> value``."""
    annotations = task.get("annotations") or []
    valid_annotations = [
        ann for ann in annotations
        if isinstance(ann, dict) and not ann.get("was_cancelled")
    ]
    if not valid_annotations:
        return None

    annotation = valid_annotations[-1]
    values: dict[str, Any] = {}
    for result in annotation.get("result", []):
        if not isinstance(result, dict):
            continue
        name = result.get("from_name")
        value = result.get("value", {})
        if not name:
            continue
        if "choices" in value:
            choices = value.get("choices") or []
            values[name] = choices[0] if choices else ""
        elif "text" in value:
            text = value.get("text") or []
            values[name] = text[0] if text else ""
        else:
            values[name] = value
    return LabelStudioAnnotation(values=values, annotation_id=annotation.get("id"))


def _fallback_model_label(data: dict[str, Any]) -> tuple[str, list[int], str]:
    difficulty = data.get("suggested_difficulty_label") or data.get("selector_difficulty") or ""
    evidence_ids = parse_sentence_ids(
        data.get("required_evidence_ids") or data.get("selector_evidence_ids")
    )
    direct = (
        data.get("suggested_answer_directly_found")
        or data.get("selector_answer_directly_found")
        or ""
    )
    return str(difficulty), evidence_ids, str(direct)


def normalize_training_record(
    raw: dict[str, Any],
    *,
    allow_model_labels: bool = False,
) -> dict[str, Any] | None:
    """Normalize one raw record into the classifier training schema.

    Label Studio records are included only when:
    - sentence split is marked ``yes``;
    - QA validity is marked ``yes``;
    - difficulty is Easy/Medium/Hard;
    - at least one evidence id is provided.

    ``allow_model_labels`` is only for smoke tests or weakly supervised pilots.
    Human-labeled training should keep it false.
    """
    data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
    annotation = flatten_label_studio_annotation(raw) if "annotations" in raw else None

    if annotation is not None:
        ann = annotation.values
        split_ok = str(ann.get("human_sentence_split_ok", "")).lower()
        human_valid = str(ann.get("human_valid", "")).lower()
        if split_ok and split_ok != "yes":
            return None
        if human_valid and human_valid != "yes":
            return None
        difficulty = str(ann.get("human_difficulty_label", ""))
        evidence_ids = parse_sentence_ids(ann.get("human_evidence_ids", ""))
        direct = str(ann.get("human_answer_directly_found", ""))
        source = "label_studio_human"
    elif allow_model_labels:
        difficulty, evidence_ids, direct = _fallback_model_label(data)
        source = "model_label"
    else:
        difficulty = str(
            data.get("difficulty_label")
            or data.get("final_difficulty")
            or data.get("human_difficulty_label")
            or ""
        )
        evidence_ids = parse_sentence_ids(
            data.get("evidence_ids")
            or data.get("required_evidence_sentences")
            or data.get("human_evidence_ids")
        )
        direct = str(data.get("answer_directly_found") or "")
        source = str(data.get("label_source") or "normalized")

    if difficulty not in LABEL2ID:
        return None
    if not evidence_ids:
        return None

    context_numbered = str(
        data.get("full_context_numbered") or data.get("context_numbered") or ""
    ).strip()
    full_context_qa = str(data.get("full_context_qa") or "").strip()
    qa_text = str(data.get("qa") or "").strip()

    if not context_numbered and full_context_qa:
        context_numbered, qa_from_full = strip_context_from_full_context_qa(full_context_qa)
        if not qa_text:
            qa_text = qa_from_full

    if not context_numbered:
        story_section = str(data.get("story_section") or "").strip()
        if story_section:
            sentence_rows = parse_numbered_context(story_section)
            if not sentence_rows:
                sentence_rows = [(idx, sent.strip()) for idx, sent in enumerate(story_section.splitlines()) if sent.strip()]
            context_numbered = build_marked_context(sentence_rows)

    sentence_rows = parse_numbered_context(context_numbered)
    if not sentence_rows and isinstance(data.get("sentences"), list):
        sentence_rows = [
            (int(item["id"]), str(item["text"]).strip())
            for item in data["sentences"]
            if isinstance(item, dict) and "id" in item and str(item.get("text", "")).strip()
        ]
    if not sentence_rows:
        return None

    question = str(data.get("question") or "").strip()
    answer = str(data.get("answer") or data.get("answer1") or "").strip()
    if (not question or not answer) and qa_text:
        question_from_qa, answer_from_qa = parse_qa_text(qa_text)
        question = question or question_from_qa
        answer = answer or answer_from_qa

    if not question or not answer:
        return None

    available_ids = {idx for idx, _ in sentence_rows}
    evidence_ids = sorted(idx for idx in evidence_ids if idx in available_ids)
    if not evidence_ids:
        return None

    return {
        "sample_id": str(data.get("sample_id") or data.get("id") or ""),
        "story_name": str(data.get("story_name") or ""),
        "source_index": str(data.get("source_index") or data.get("sample_original_index") or ""),
        "attribute": str(data.get("attribute") or ""),
        "fairytale_scope": str(data.get("fairytale_scope") or data.get("local_or_sum") or ""),
        "fairytale_explicitness": str(data.get("fairytale_explicitness") or data.get("ex_or_im") or ""),
        "question": question,
        "answer": answer,
        "difficulty_label": difficulty,
        "difficulty_id": LABEL2ID[difficulty],
        "answer_directly_found": direct,
        "evidence_ids": evidence_ids,
        "context_numbered": build_marked_context(sentence_rows),
        "sentences": [{"id": idx, "text": sent} for idx, sent in sentence_rows],
        "label_source": source,
        "annotation_id": annotation.annotation_id if annotation else None,
    }


def load_training_labels(
    implicit_path: str,
    explicit_path: str | None = None,
    *,
    allow_model_labels: bool = False,
) -> list[dict[str, Any]]:
    """Backward-compatible loader used by training scripts."""
    paths = [implicit_path]
    if explicit_path:
        paths.append(explicit_path)
    return load_training_records(paths, allow_model_labels=allow_model_labels)


def load_training_records(
    paths: str | Path | Iterable[str | Path],
    *,
    allow_model_labels: bool = False,
) -> list[dict[str, Any]]:
    """Load and normalize one or more training data files."""
    if isinstance(paths, (str, Path)):
        path_list = [paths]
    else:
        path_list = list(paths)

    records: list[dict[str, Any]] = []
    skipped = 0
    for path in path_list:
        for raw in read_records(path):
            item = normalize_training_record(raw, allow_model_labels=allow_model_labels)
            if item is None:
                skipped += 1
            else:
                records.append(item)

    logger.info("Loaded %d classifier records; skipped %d", len(records), skipped)
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_stratified_folds(
    records: list[dict[str, Any]],
    n_folds: int = 5,
    seed: int = 42,
    *,
    group_by_story: bool = False,
) -> list[tuple[list[int], list[int]]]:
    """Create stratified folds, optionally keeping stories in one fold."""
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if len(records) < n_folds:
        raise ValueError(f"Need at least {n_folds} records, got {len(records)}")

    labels = [record["difficulty_label"] for record in records]
    indices = list(range(len(records)))

    if group_by_story:
        from sklearn.model_selection import StratifiedGroupKFold

        groups = [
            record.get("story_name") or f"row-{idx}"
            for idx, record in enumerate(records)
        ]
        splitter = StratifiedGroupKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed,
        )
        return [
            (train_idx.tolist(), val_idx.tolist())
            for train_idx, val_idx in splitter.split(indices, labels, groups)
        ]

    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [
        (train_idx.tolist(), val_idx.tolist())
        for train_idx, val_idx in splitter.split(indices, labels)
    ]


def truncate_sentences_for_markers(
    sentence_rows: list[tuple[int, str]],
    evidence_ids: list[int],
    max_markers: int,
) -> tuple[list[tuple[int, str]], list[int], dict[int, int]]:
    """Keep evidence and nearby context when a section exceeds max markers.

    Returns remapped sentence rows with marker ids starting at 0, remapped
    evidence ids, and ``old_id -> new_id``.
    """
    if len(sentence_rows) <= max_markers:
        mapping = {old_id: pos for pos, (old_id, _) in enumerate(sentence_rows)}
        remapped_rows = [(mapping[old_id], text) for old_id, text in sentence_rows]
        remapped_evidence = sorted(mapping[idx] for idx in evidence_ids if idx in mapping)
        return remapped_rows, remapped_evidence, mapping

    available = [idx for idx, _ in sentence_rows]
    available_set = set(available)
    selected: list[int] = []
    selected_set: set[int] = set()

    def add(ids: Iterable[int]) -> None:
        for idx in ids:
            if idx in available_set and idx not in selected_set and len(selected) < max_markers:
                selected.append(idx)
                selected_set.add(idx)

    add(evidence_ids)
    for ev_id in evidence_ids:
        add([ev_id - 2, ev_id - 1, ev_id + 1, ev_id + 2])
    add(available)

    selected_order = [idx for idx in available if idx in selected_set][:max_markers]
    mapping = {old_id: new_id for new_id, old_id in enumerate(selected_order)}
    text_by_id = dict(sentence_rows)
    remapped_rows = [(mapping[old_id], text_by_id[old_id]) for old_id in selected_order]
    remapped_evidence = sorted(mapping[idx] for idx in evidence_ids if idx in mapping)
    return remapped_rows, remapped_evidence, mapping


class DifficultyEvidenceDataset:
    """PyTorch Dataset for joint difficulty and evidence training."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        *,
        max_length: int = 512,
        max_markers: int = DEFAULT_MAX_MARKERS,
    ) -> None:
        _get_torch()
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_markers = max_markers
        self.marker_tokens = [f"[S{i}]" for i in range(max_markers)]
        self.marker_token_ids = [
            tokenizer.convert_tokens_to_ids(token) for token in self.marker_tokens
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        torch = _get_torch()
        record = self.records[idx]
        sentence_rows = [
            (int(item["id"]), str(item["text"])) for item in record["sentences"]
        ]
        sentence_rows, evidence_ids, old_to_new = truncate_sentences_for_markers(
            sentence_rows,
            parse_sentence_ids(record["evidence_ids"]),
            self.max_markers,
        )

        marked_context = build_marked_context(sentence_rows)
        input_text = build_classifier_input(
            record["question"],
            record["answer"],
            marked_context,
        )
        encoding = self.tokenizer(
            input_text,
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
        evidence_labels = torch.zeros(self.max_markers, dtype=torch.float32)
        evidence_set = set(evidence_ids)

        for marker_idx, _sentence_text in sentence_rows:
            if marker_idx >= self.max_markers:
                continue
            marker_token_id = self.marker_token_ids[marker_idx]
            try:
                token_pos = input_id_list.index(marker_token_id)
            except ValueError:
                continue
            marker_positions[marker_idx] = token_pos
            marker_mask[marker_idx] = 1.0
            if marker_idx in evidence_set:
                evidence_labels[marker_idx] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "marker_positions": marker_positions,
            "marker_mask": marker_mask,
            "difficulty_label": torch.tensor(record["difficulty_id"], dtype=torch.long),
            "evidence_labels": evidence_labels,
            "metadata": {
                "sample_id": record.get("sample_id", ""),
                "story_name": record.get("story_name", ""),
                "old_to_new_sentence_ids": old_to_new,
            },
        }


def collate_difficulty_evidence(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack already-tokenized dataset items into a batch."""
    torch = _get_torch()
    tensor_keys = [
        "input_ids",
        "attention_mask",
        "marker_positions",
        "marker_mask",
        "difficulty_label",
        "evidence_labels",
    ]
    output = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    output["metadata"] = [item.get("metadata", {}) for item in batch]
    return output
