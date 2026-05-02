"""Stage 4: Generate questions using PathQG-HardAware.

Usage:
    python -m scripts.04_generate_questions --input outputs/runs/latest/paths.filtered.jsonl --output outputs/runs/latest/questions.raw.jsonl
"""
import argparse
import json
import time
from pathlib import Path

from dcqg.generation.generator import generate_with_retry_hardaware
from dcqg.utils.jsonl import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description="Generate questions using PathQG-HardAware.")
    parser.add_argument("--input", default="outputs/runs/latest/paths.filtered.jsonl", help="Filtered paths JSONL")
    parser.add_argument("--output", default="outputs/runs/latest/questions.raw.jsonl", help="Output questions JSONL")
    parser.add_argument("--limit", type=int, default=0, help="Limit items (0=all)")
    parser.add_argument("--max_attempts", type=int, default=3, help="Max retry attempts per item")
    parser.add_argument("--include_failed_judge", action="store_true", help="Include path-judge-rejected items")
    args = parser.parse_args()

    items = read_jsonl(args.input, n=args.limit or None)
    print(f"Loaded {len(items)} paths from {args.input}")

    if not args.include_failed_judge:
        items = [i for i in items if i.get("llm_path_keep", True)]
        print(f"  After filtering path-judge rejects: {len(items)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(items):
            item["_item_id"] = i
            r, attempts = generate_with_retry_hardaware(item, max_attempts=args.max_attempts)
            results.append(r)
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out_f.flush()

            if (i + 1) % 20 == 0:
                n_pass = sum(1 for r in results if r.get("grammar_pass", False))
                print(f"  [{i+1}/{len(items)}] grammar_pass={n_pass}", flush=True)

            time.sleep(0.1)

    n_pass = sum(1 for r in results if r.get("grammar_pass", False))
    print(f"\nGenerated {len(results)} questions, {n_pass} grammar-passed ({n_pass/len(results)*100:.0f}%)")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
