"""Consolidated JSONL I/O utilities."""
import json
from pathlib import Path


def read_jsonl(path, n=None):
    """Read JSONL file. If n is given, read at most n lines."""
    with open(path, encoding="utf-8") as f:
        lines = []
        for i, line in enumerate(f):
            if n and i >= n:
                break
            stripped = line.strip()
            if stripped:
                lines.append(json.loads(stripped))
    return lines


def write_jsonl(path, rows):
    """Write rows as JSONL. Creates parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
