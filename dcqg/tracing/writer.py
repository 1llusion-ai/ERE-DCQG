import json
from pathlib import Path


def write_full_trace(traces, output_dir):
    """Write full_trace.jsonl -- one JSON line per item, no truncation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "full_trace.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for trace in traces:
            f.write(trace.to_json() + "\n")
    return path
