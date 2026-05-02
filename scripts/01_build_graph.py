"""Stage 1: Build document-level event graphs from raw MAVEN-ERE data.

Usage:
    python -m scripts.01_build_graph --input data/raw/maven_ere/valid.jsonl --output outputs/runs/latest/graphs.jsonl --limit 10
"""
import argparse
import json
from pathlib import Path

from dcqg.graph import EventGraph
from dcqg.utils.jsonl import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description="Build event graphs from raw MAVEN-ERE data.")
    parser.add_argument("--input", default="data/raw/maven_ere/valid.jsonl", help="Raw MAVEN-ERE JSONL file")
    parser.add_argument("--output", default="outputs/runs/latest/graphs.jsonl", help="Output JSONL with graph stats")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of documents (0=all)")
    args = parser.parse_args()

    docs = read_jsonl(args.input, n=args.limit or None)
    print(f"Loaded {len(docs)} documents from {args.input}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = []
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            g = EventGraph(doc)
            stat = {
                "doc_id": g.doc_id,
                "title": g.title,
                "num_events": g.num_events,
                "num_edges": g.num_edges,
                "relation_distribution": g.relation_type_distribution(),
                "num_sentences": len(g.sentences),
            }
            stats.append(stat)
            f.write(json.dumps(stat, ensure_ascii=False) + "\n")

    total_events = sum(s["num_events"] for s in stats)
    total_edges = sum(s["num_edges"] for s in stats)
    print(f"Built {len(stats)} graphs: {total_events} events, {total_edges} edges")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
