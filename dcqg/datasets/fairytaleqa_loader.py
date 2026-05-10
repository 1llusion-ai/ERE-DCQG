"""FairytaleQA dataset loader.

Supports two sources (tried in order):
  1. HuggingFace: WorkInTheDark/FairytaleQA (requires `datasets` package)
  2. Local CSV: uci-soe/FairytaleQAData repo structure

Returns list of dicts with normalized field names.
"""
import csv
import os
from pathlib import Path


# Canonical field names we want
CANONICAL_FIELDS = [
    "story_name",
    "story_section",
    "question",
    "answer1",
    "answer2",
    "local_or_sum",   # "local" or "summary"
    "attribute",       # e.g. "action", "feeling", "setting", etc.
    "ex_or_im",        # "explicit" or "implicit"
    "ex_or_im2",
    "split",           # "train" / "validation" / "test"
]


def _try_huggingface(split, limit):
    """Try loading from HuggingFace by downloading CSV directly."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    # Map split name to CSV filename
    csv_name_map = {"train": "train.csv", "validation": "valid.csv", "test": "test.csv"}
    csv_name = csv_name_map.get(split, f"{split}.csv")

    try:
        csv_path = hf_hub_download(
            repo_id="WorkInTheDark/FairytaleQA",
            filename=csv_name,
            repo_type="dataset",
        )
    except Exception:
        return None

    records = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            rec = _normalize_csv_row(row, split)
            if rec:
                records.append(rec)
    return records if records else None


def _try_local_csv(split, limit):
    """Try loading from local CSV repo structure (uci-soe/FairytaleQAData).

    Expected structure:
      FairytaleQAData/
        train.csv
        validation.csv
        test.csv
    """
    # Search for CSV files
    search_dirs = [
        Path("data/fairytaleqa"),
        Path("data/raw/fairytaleqa"),
        Path("FairytaleQAData"),
        Path("data/FairytaleQAData"),
    ]

    # Also check env var
    env_dir = os.environ.get("FAIRYTALEQA_DIR")
    if env_dir:
        search_dirs.insert(0, Path(env_dir))

    csv_path = None
    for d in search_dirs:
        candidate = d / f"{split}.csv"
        if candidate.exists():
            csv_path = candidate
            break
        # Try without split prefix
        for f in d.glob("*.csv"):
            if split.lower() in f.stem.lower():
                csv_path = f
                break
        if csv_path:
            break

    if not csv_path:
        return None

    records = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            rec = _normalize_csv_row(row, split)
            if rec:
                records.append(rec)
    return records if records else None


def _normalize_csv_row(row, split):
    """Normalize a CSV row to canonical fields."""
    def _get(key, default=""):
        # Try exact key, then lowercase, then with hyphens/underscores swapped
        val = row.get(key)
        if val is None:
            val = row.get(key.lower())
        if val is None:
            val = row.get(key.replace("_", "-"))
        if val is None:
            val = row.get(key.replace("-", "_"))
        if val is None:
            return default
        return str(val).strip()

    question = _get("question")
    if not question:
        return None

    return {
        "story_name": _get("story_name") or _get("story") or _get("book"),
        "story_section": _get("story_section") or _get("section") or _get("context"),
        "question": question,
        "answer1": _get("answer1") or _get("answer"),
        "answer2": _get("answer2", ""),
        "local_or_sum": _get("local_or_sum") or _get("local-or-sum"),
        "attribute": _get("attribute"),
        "ex_or_im": _get("ex_or_im") or _get("ex-or-im"),
        "ex_or_im2": _get("ex_or_im2") or _get("ex-or-im2", ""),
        "split": split,
    }


def load_fairytaleqa(split="validation", limit=None):
    """Load FairytaleQA dataset. Returns list of normalized dicts.

    Tries HuggingFace first, falls back to local CSV.

    Args:
        split: "train", "validation", or "test"
        limit: max records to load (None = all)

    Returns:
        list of dicts with canonical fields

    Raises:
        RuntimeError: if neither source is available
    """
    records = _try_huggingface(split, limit)
    if records:
        return records

    records = _try_local_csv(split, limit)
    if records:
        return records

    raise RuntimeError(
        "Could not load FairytaleQA. Install `datasets` package "
        "(pip install datasets) for HuggingFace access, or place CSV files "
        "in data/fairytaleqa/{split}.csv or set FAIRYTALEQA_DIR env var."
    )
