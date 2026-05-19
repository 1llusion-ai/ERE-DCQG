"""Convert annotation exports into normalized classifier JSONL.

Input can be:
- Label Studio JSON export;
- Label Studio task JSON with annotations;
- already-normalized JSONL;
- review CSV.

Only human-valid examples are kept by default. Use ``--allow_model_labels`` only
for weak-label debugging before final human annotations are ready.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dcqg.difficulty.data import load_training_records, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare normalized data for the DeBERTa multi-task classifier.",
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input file. Repeat for implicit + explicit files.",
    )
    parser.add_argument("--output", required=True, help="Output normalized JSONL.")
    parser.add_argument(
        "--allow_model_labels",
        action="store_true",
        help="Use model-suggested labels when no human annotation exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_training_records(
        args.input,
        allow_model_labels=args.allow_model_labels,
    )
    output_path = Path(args.output)
    write_jsonl(records, output_path)

    summary = {
        "num_records": len(records),
        "difficulty_distribution": dict(Counter(r["difficulty_label"] for r in records)),
        "source_distribution": dict(Counter(r.get("label_source", "") for r in records)),
        "explicitness_distribution": dict(Counter(r.get("fairytale_explicitness", "") for r in records)),
        "num_stories": len({r.get("story_name", "") for r in records}),
    }
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote records: {output_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
