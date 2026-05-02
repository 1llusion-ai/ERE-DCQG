"""Stage 2: Sample Easy/Medium/Hard paths from event graphs.

Usage:
    python -m scripts.02_sample_paths --input data/raw/maven_ere/valid.jsonl --output outputs/runs/latest/paths.raw.jsonl --limit 100
"""
import argparse
from pathlib import Path

from dcqg.graph import EventGraph
from dcqg.path.sampler import sample_from_doc
from dcqg.utils.jsonl import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description="Sample difficulty-labeled paths from event graphs.")
    parser.add_argument("--input", default="data/raw/maven_ere/valid.jsonl", help="Raw MAVEN-ERE JSONL file")
    parser.add_argument("--output", default="outputs/runs/latest/paths.raw.jsonl", help="Output JSONL with sampled paths")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of documents (0=all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_per_doc", type=int, default=5, help="Max paths per document per difficulty")
    args = parser.parse_args()

    docs = read_jsonl(args.input, n=args.limit or None)
    print(f"Loaded {len(docs)} documents from {args.input}")

    all_paths = []
    target_counts = {"Easy": args.max_per_doc, "Medium": args.max_per_doc, "Hard": args.max_per_doc}
    import random
    rng = random.Random(args.seed)
    for doc in docs:
        paths = sample_from_doc(EventGraph(doc), target_counts=target_counts, rng=rng)
        all_paths.extend(paths)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, all_paths)

    by_diff = {}
    for p in all_paths:
        d = p.get("difficulty", "unknown")
        by_diff[d] = by_diff.get(d, 0) + 1
    print(f"Sampled {len(all_paths)} paths: {by_diff}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
