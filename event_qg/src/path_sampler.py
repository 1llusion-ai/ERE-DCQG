"""
Path-level instance sampler with hop-based difficulty.
Difficulty is determined solely by hop count:
  Easy = 1 hop, Medium = 2 hops, Hard = 3 hops.
Directed paths: only follow outgoing relation direction.
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
import random

from graph_builder import EventGraph, load_jsonl


def _get_supporting_sentences(g, path):
    """Get supporting sentence IDs: path event sentences ± 1 sentence window."""
    path_sent_ids = set()
    for eid in path:
        info = g.get_event_info(eid)
        path_sent_ids.add(info.get("sent_id", 0))
    expanded = set()
    for sid in path_sent_ids:
        for s in range(max(0, sid - 1), min(len(g.sentences), sid + 2)):
            expanded.add(s)
    return sorted(expanded)


def _get_path_relation_subtypes(g, path):
    """Return list of full relation subtype strings for each hop. Forward-only."""
    subtypes = []
    for i in range(len(path) - 1):
        src, tgt = path[i], path[i + 1]
        key = None
        for out_tgt, edge_type, edge_sub in g.get_out_neighbors(src):
            if out_tgt == tgt:
                key = f"{edge_type}/{edge_sub}" if edge_sub else edge_type
                break
        subtypes.append(key or "UNKNOWN")
    return subtypes


def _classify_difficulty(hops):
    """Hop-based difficulty: 1=Easy, 2=Medium, 3+=Hard."""
    if hops <= 1:
        return "Easy"
    elif hops == 2:
        return "Medium"
    else:
        return "Hard"


def _compute_relation_distribution(rel_subtypes):
    """Compute relation distribution label from subtype list."""
    rel_type_set = set()
    for rs in rel_subtypes:
        if rs.startswith("CAUSE"):
            rel_type_set.add("CAUSE")
        elif rs.startswith("TEMPORAL"):
            rel_type_set.add("TEMPORAL")
        elif rs.startswith("SUBEVENT"):
            rel_type_set.add("SUBEVENT")

    if len(rel_type_set) >= 2:
        return "MIXED"
    elif "CAUSE" in rel_type_set:
        return "CAUSE"
    elif "SUBEVENT" in rel_type_set:
        return "SUBEVENT"
    else:
        return "TEMPORAL"


def build_path_info(g, path):
    """Build a serializable dict for a sampled path."""
    try:
        src_eid, tgt_eid = path[0], path[-1]
        tgt_info = g.get_event_info(tgt_eid)
        hops = len(path) - 1

        supporting_sents = _get_supporting_sentences(g, path)
        sent_texts = [(sid, g.get_sentence(sid)) for sid in supporting_sents]
        rel_subtypes = _get_path_relation_subtypes(g, path)

        events_detail = []
        for eid in path:
            info = g.get_event_info(eid)
            events_detail.append({
                "id": eid,
                "type": info["type"],
                "trigger": info["trigger"],
                "sent_id": info["sent_id"],
                "offset": info.get("offset", [])
            })

        return {
            "doc_id": g.doc_id,
            "title": g.title,
            "difficulty": _classify_difficulty(hops),
            "hops": hops,
            "path": path,
            "relation_subtypes": rel_subtypes,
            "relation_distribution": _compute_relation_distribution(rel_subtypes),
            "events": events_detail,
            "answer_event_id": tgt_eid,
            "answer_trigger": tgt_info["trigger"],
            "supporting_sentences": sent_texts,
            "num_supporting_sentences": len(sent_texts)
        }
    except Exception:
        return None


def sample_from_doc(g, target_counts, rng):
    """
    Sample directed paths from one document.
    - Easy: 1-hop
    - Medium: 2-hop
    - Hard: 3-hop
    """
    event_ids = set(g.events.keys())

    # ---- Easy: all 1-hop outgoing edges ----
    easy_candidates = []
    seen_1hop = set()
    for src in event_ids:
        for tgt, _, _ in g.get_out_neighbors(src):
            if tgt not in event_ids:
                continue
            key = (src, tgt)
            if key in seen_1hop:
                continue
            seen_1hop.add(key)
            info = build_path_info(g, [src, tgt])
            if info:
                easy_candidates.append(info)

    # ---- Medium: enumerate 2-hop directed paths ----
    medium_candidates = []
    seen_2hop = set()
    for src in event_ids:
        for mid, _, _ in g.get_out_neighbors(src):
            if mid not in event_ids:
                continue
            for tgt, _, _ in g.get_out_neighbors(mid):
                if tgt not in event_ids or tgt == src:
                    continue
                key = (src, mid, tgt)
                if key in seen_2hop:
                    continue
                seen_2hop.add(key)
                info = build_path_info(g, [src, mid, tgt])
                if info:
                    medium_candidates.append(info)

    # ---- Hard: 3-hop directed paths via BFS ----
    MAX_HARD_TOTAL = 300
    MAX_START_NODES = 20

    hard_candidates = []
    seen_3hop = set()
    path_counter = [0]

    start_nodes = rng.sample(list(event_ids), min(len(event_ids), MAX_START_NODES))

    def bfs_3hop(start):
        queue = [(start, [start])]
        visited_paths = set()

        while queue and path_counter[0] < MAX_HARD_TOTAL:
            current, path = queue.pop(0)
            if len(path) == 4:  # 3 hops reached
                path_counter[0] += 1
                key = tuple(path)
                if key in seen_3hop:
                    continue
                seen_3hop.add(key)
                info = build_path_info(g, path)
                if info:
                    hard_candidates.append(info)
                continue

            if len(path) > 3:
                continue

            for tgt, _, _ in g.get_out_neighbors(current):
                if tgt in path:
                    continue
                new_path = path + [tgt]
                path_key = "|".join(new_path)
                if path_key in visited_paths:
                    continue
                visited_paths.add(path_key)
                queue.append((tgt, new_path))

    for start in start_nodes:
        if path_counter[0] >= MAX_HARD_TOTAL:
            break
        bfs_3hop(start)

    # Deduplicate
    def dedup(lst):
        seen = set()
        out = []
        for item in lst:
            key = "|".join(item["path"])
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out

    easy_candidates = dedup(easy_candidates)
    medium_candidates = dedup(medium_candidates)
    hard_candidates = dedup(hard_candidates)

    # Sample
    sampled = []
    for level in ["Easy", "Medium", "Hard"]:
        pool = {"Easy": easy_candidates, "Medium": medium_candidates, "Hard": hard_candidates}[level]
        n = min(target_counts[level], len(pool))
        if pool:
            sampled.extend(rng.sample(pool, n))

    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="event_qg/data/raw")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--num_docs", type=int, default=100)
    parser.add_argument("--samples_per_level", type=int, default=30)
    parser.add_argument("--output_dir", default="event_qg/outputs")
    args = parser.parse_args()

    target_counts = {
        "Easy": args.samples_per_level,
        "Medium": args.samples_per_level,
        "Hard": args.samples_per_level
    }

    filepath = Path(args.data_dir) / f"{args.split}.jsonl"
    print(f"Loading docs from {filepath}...")
    docs = load_jsonl(filepath, n=args.num_docs)
    print(f"Loaded {len(docs)} documents")

    all_sampled = []
    stats = {"Easy": 0, "Medium": 0, "Hard": 0}
    rel_dist_counter = defaultdict(int)

    for i, doc in enumerate(docs):
        g = EventGraph(doc)
        seed = hash(doc.get("id", str(i))) % (2**31)
        rng = random.Random(seed)
        sampled = sample_from_doc(g, target_counts, rng)
        for s in sampled:
            all_sampled.append(s)
            stats[s["difficulty"]] += 1
            rel_dist_counter[s["relation_distribution"]] += 1
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(docs)} docs, E={stats['Easy']} M={stats['Medium']} H={stats['Hard']}")

    print(f"\n=== Path Sampling Report ===")
    print(f"Split: {args.split} | Docs: {len(docs)}")
    print(f"Sampled: Easy={stats['Easy']}, Medium={stats['Medium']}, Hard={stats['Hard']}")
    print(f"Total: {len(all_sampled)}")
    print(f"Relation distribution: {dict(rel_dist_counter)}")

    # Save JSONL
    output_path = Path(args.output_dir) / "sampled_paths_preview.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_sampled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nSaved to {output_path}")

    # Save report
    report = {
        "split": args.split,
        "num_docs": len(docs),
        "total_sampled": len(all_sampled),
        "difficulty_counts": stats,
        "relation_distribution": dict(rel_dist_counter),
    }
    report_path = Path(args.output_dir) / "path_sampling_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()