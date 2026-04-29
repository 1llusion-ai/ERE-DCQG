"""
Inspect MAVEN-ERE data: read raw files, print keys, parse events and relations.
Output: 5-10 readable samples to outputs/inspect_samples.md
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_jsonl(path, n=None):
    """Load jsonl file, optionally limit to n lines."""
    with open(path, encoding="utf-8") as f:
        lines = []
        for i, line in enumerate(f):
            if n and i >= n:
                break
            lines.append(json.loads(line))
    return lines


def print_readable_sample(doc, sample_id):
    """Convert a document into a readable text representation."""
    lines = []
    lines.append(f"## Sample {sample_id}: {doc.get('title', 'No title')}")
    lines.append(f"**Doc ID**: {doc.get('id', 'N/A')}")

    # Sentences
    sentences = doc.get("sentences", [])
    if sentences:
        lines.append(f"\n### Sentences ({len(sentences)} total)")
        for i, sent in enumerate(sentences):
            sent_text = sent if isinstance(sent, str) else str(sent)
            lines.append(f"  [{i}] {sent_text[:200]}")

    # Events
    events = doc.get("events", [])
    if events:
        lines.append(f"\n### Events ({len(events)} total)")
        for e in events[:30]:  # cap at 30 for readability
            eid = e.get("id", "N/A")
            etype = e.get("type", "N/A")
            mentions = e.get("mention", [])
            for m in mentions:
                trigger = m.get("trigger_word", "N/A")
                sent_id = m.get("sent_id", "N/A")
                offset = m.get("offset", [])
                lines.append(f"  - {eid} | type={etype} | trigger='{trigger}' | sent_id={sent_id} | offset={offset}")

    # Relations
    for rel_type in ["causal_relations", "temporal_relations", "subevent_relations"]:
        rel_data = doc.get(rel_type)
        if rel_data:
            lines.append(f"\n### {rel_type} ({len(rel_data)} entries)")
            if isinstance(rel_data, dict):
                for sub_type, pairs in rel_data.items():
                    lines.append(f"  ** {sub_type}: {len(pairs)} pairs")
                    for pair in pairs[:10]:
                        lines.append(f"    -> {pair}")
            elif isinstance(rel_data, list):
                for pair in rel_data[:10]:
                    lines.append(f"  -> {pair}")
            if rel_data and isinstance(rel_data, (dict, list)):
                total = len(rel_data) if isinstance(rel_data, list) else sum(len(v) for v in rel_data.values())
                if total > 10:
                    lines.append(f"  ... (showing 10 of {total})")

    lines.append("\n---")
    return "\n".join(lines)


def print_raw_keys(docs):
    """Print all top-level keys found across documents."""
    all_keys = set()
    for doc in docs:
        all_keys.update(doc.keys())
    return sorted(all_keys)


def inspect_nested_fields(doc, prefix=""):
    """Recursively find all nested field names and types."""
    fields = {}
    if not isinstance(doc, dict):
        return fields
    for k, v in doc.items():
        full_k = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            fields[full_k] = "dict"
            fields.update(inspect_nested_fields(v, full_k))
        elif isinstance(v, list):
            if not v:
                fields[full_k] = "list (empty)"
            elif isinstance(v[0], dict):
                fields[full_k] = f"list[dict] (len={len(v)})"
                fields.update(inspect_nested_fields(v[0], f"{full_k}[0]"))
            else:
                fields[full_k] = f"list[{type(v[0]).__name__}] (len={len(v)})"
        else:
            fields[full_k] = type(v).__name__
    return fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="event_qg/data/raw")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_docs", type=int, default=20)
    parser.add_argument("--output", default="event_qg/outputs/inspect_samples.md")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    split_file = data_dir / f"{args.split}.jsonl"

    print(f"Loading {args.num_docs} docs from {split_file}...")
    docs = load_jsonl(split_file, n=args.num_docs)
    print(f"Loaded {len(docs)} documents")

    # 1. Raw keys
    all_keys = print_raw_keys(docs)
    print(f"\nTop-level keys across documents: {all_keys}")

    # 2. Nested field inspection on first doc
    first_fields = inspect_nested_fields(docs[0])
    print(f"\nNested fields in first doc ({docs[0].get('id', 'N/A')}):")
    for k, v in sorted(first_fields.items()):
        print(f"  {k}: {v}")

    # 3. Generate readable samples
    samples = []
    for i, doc in enumerate(docs[:10]):
        samples.append(print_readable_sample(doc, i + 1))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# MAVEN-ERE Data Inspection\n\n")
        f.write(f"**Split**: {args.split} | **Docs inspected**: {len(docs[:10])}\n\n")
        f.write(f"**Top-level keys**: {all_keys}\n\n")
        f.write("---\n\n")
        f.write("\n\n".join(samples))

    print(f"\nWritable samples saved to {output_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()