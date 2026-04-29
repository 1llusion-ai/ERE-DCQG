"""
Build document-level event graphs from MAVEN-ERE documents.
Node: event mention
Edge: typed directed relation (using relation subtype, e.g. TEMPORAL/BEFORE, CAUSE/PRECONDITION)
Output: per-document event count, edge count, relation type distribution.
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_jsonl(path, n=None):
    with open(path, encoding="utf-8") as f:
        lines = []
        for i, line in enumerate(f):
            if n and i >= n:
                break
            lines.append(json.loads(line))
    return lines


class EventGraph:
    """Document-level event graph with directed edges."""

    def __init__(self, doc):
        self.doc = doc
        self.doc_id = doc.get("id", "unknown")
        self.title = doc.get("title", "")
        self.sentences = doc.get("sentences", [])
        self.events = {}  # event_id -> event dict
        # Directed edges: (src_id, tgt_id, rel_type, rel_subtype)
        self.edges = []
        # Outgoing neighbors only (direction-aware)
        self.out_neighbors = defaultdict(list)  # src -> [(tgt, rel_type, rel_subtype), ...]

        self._build()

    def _build(self):
        """Build nodes and directed edges from the document."""
        for e in self.doc.get("events", []):
            eid = e.get("id")
            if eid:
                self.events[eid] = e

        type_map = {
            "causal_relations": "CAUSE",
            "temporal_relations": "TEMPORAL",
            "subevent_relations": "SUBEVENT"
        }

        for rel_type_key in ["causal_relations", "temporal_relations", "subevent_relations"]:
            rel_data = self.doc.get(rel_type_key)
            if not rel_data:
                continue
            edge_type = type_map.get(rel_type_key, rel_type_key.upper())

            if isinstance(rel_data, dict):
                for sub_type, pairs in rel_data.items():
                    for pair in pairs:
                        if len(pair) >= 2:
                            src, tgt = pair[0], pair[1]
                            if src in self.events and tgt in self.events:
                                self.edges.append((src, tgt, edge_type, sub_type))
                                self.out_neighbors[src].append((tgt, edge_type, sub_type))
            elif isinstance(rel_data, list):
                for pair in rel_data:
                    if len(pair) >= 2:
                        src, tgt = pair[0], pair[1]
                        if src in self.events and tgt in self.events:
                            self.edges.append((src, tgt, edge_type, ""))
                            self.out_neighbors[src].append((tgt, edge_type, ""))

    @property
    def num_events(self):
        return len(self.events)

    @property
    def num_edges(self):
        return len(self.edges)

    def get_event_info(self, eid):
        e = self.events.get(eid, {})
        mentions = e.get("mention", [])
        if mentions:
            m = mentions[0]
            return {
                "id": eid,
                "type": e.get("type", "N/A"),
                "trigger": m.get("trigger_word", "N/A"),
                "sent_id": m.get("sent_id", 0),
                "offset": m.get("offset", [])
            }
        return {"id": eid, "type": e.get("type", "N/A"), "trigger": "N/A", "sent_id": 0, "offset": []}

    def get_sentence(self, sent_id):
        if 0 <= sent_id < len(self.sentences):
            sent = self.sentences[sent_id]
            return sent if isinstance(sent, str) else str(sent)
        return ""

    def relation_type_distribution(self):
        dist = defaultdict(int)
        for src, tgt, edge_type, sub_type in self.edges:
            key = f"{edge_type}/{sub_type}" if sub_type else edge_type
            dist[key] += 1
        return dict(dist)

    def get_out_neighbors(self, event_id):
        """Return all outgoing (tgt, rel_type, rel_subtype) from this event."""
        return self.out_neighbors.get(event_id, [])

    def __repr__(self):
        return (f"EventGraph(doc_id={self.doc_id}, "
                f"events={self.num_events}, edges={self.num_edges})")


def build_graphs_from_file(filepath, limit=None):
    docs = load_jsonl(filepath, n=limit)
    graphs = []
    for doc in docs:
        g = EventGraph(doc)
        graphs.append(g)
    return graphs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="event_qg/data/raw")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_docs", type=int, default=50)
    parser.add_argument("--output_dir", default="event_qg/outputs")
    args = parser.parse_args()

    filepath = Path(args.data_dir) / f"{args.split}.jsonl"
    print(f"Building graphs from {filepath}...")
    graphs = build_graphs_from_file(filepath, limit=args.num_docs)

    total_events = sum(g.num_events for g in graphs)
    total_edges = sum(g.num_edges for g in graphs)

    rel_dist = defaultdict(int)
    for g in graphs:
        for k, v in g.relation_type_distribution().items():
            rel_dist[k] += v

    print(f"\n=== Graph Building Report ===")
    print(f"Documents processed: {len(graphs)}")
    print(f"Total events: {total_events}")
    print(f"Total edges: {total_edges}")
    print(f"\nRelation type distribution:")
    for k, v in sorted(rel_dist.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "graph_building_report.json"
    report = {
        "split": args.split,
        "num_docs": len(graphs),
        "total_events": total_events,
        "total_edges": total_edges,
        "relation_type_distribution": dict(rel_dist)
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()