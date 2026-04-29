"""
Path-level instance sampler with controlled relation distribution.
Key changes vs v1:
  - Directed paths: only follow outgoing relation direction
  - RD uses full subtype (TEMPORAL/BEFORE, CAUSE/PRECONDITION, etc.)
  - Hard rule tightened: 3-hop + RD>=2 OR (ES>=3 AND causal/mixed)
  - Supporting context: path sentences + ±1 sentence window
  - Formal evaluation on test/valid splits (train only for tuning)
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
import random

from graph_builder import EventGraph, load_jsonl
from difficulty_scorer import DifficultyScorer


def build_path_info(g, path, score, scorer):
    """Build a serializable dict for a sampled path."""
    try:
        src_eid, tgt_eid = path[0], path[-1]
        src_info = g.get_event_info(src_eid)
        tgt_info = g.get_event_info(tgt_eid)

        # Supporting sentences: path event sentences + ±1 window
        supporting_sents = scorer.get_supporting_sentences(path)
        sent_texts = [(sid, g.get_sentence(sid)) for sid in supporting_sents]

        # Relation subtypes on path (full subtypes)
        rel_subtypes = scorer.get_path_relation_subtypes(path)

        # Relation distribution label
        rel_type_set = set()
        for rs in rel_subtypes:
            if rs.startswith("CAUSE"):
                rel_type_set.add("CAUSE")
            elif rs.startswith("TEMPORAL"):
                rel_type_set.add("TEMPORAL")
            elif rs.startswith("SUBEVENT"):
                rel_type_set.add("SUBEVENT")

        if len(rel_type_set) >= 2:
            rel_dist_label = "MIXED"
        elif "CAUSE" in rel_type_set:
            rel_dist_label = "CAUSE"
        elif "SUBEVENT" in rel_type_set:
            rel_dist_label = "SUBEVENT"
        else:
            rel_dist_label = "TEMPORAL"

        events_detail = []
        for eid in path:
            info = g.get_event_info(eid)
            events_detail.append({
                "id": eid,
                "type": info["type"],
                "trigger": info["trigger"],
                "sent_id": info["sent_id"]
            })

        return {
            "doc_id": g.doc_id,
            "title": g.title,
            "difficulty": score["difficulty"],
            "difficulty_score": score["D"],
            "PL": score["PL"],
            "RD": score["RD"],
            "ES": score["ES"],
            "EA": score["EA"],
            "path": path,
            "relation_subtypes": rel_subtypes,
            "relation_distribution": rel_dist_label,
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
    Sample directed paths from one document with controlled relation distribution.
    - Easy: 1-hop, any relation subtype, D=4-6
    - Medium: 2-hop directed path, D=7-9
    - Hard: 3-hop with tightened criteria (see below)
    """
    scorer = DifficultyScorer(g)
    event_ids = set(g.events.keys())

    # ---- Easy: all 1-hop outgoing edges with Easy score ----
    easy_candidates = []
    seen_1hop = set()
    for src in event_ids:
        for tgt, edge_type, edge_sub in g.get_out_neighbors(src):
            if tgt not in event_ids:
                continue
            key = (src, tgt)
            if key in seen_1hop:
                continue
            seen_1hop.add(key)
            score = scorer.score_path([src, tgt])
            if score["difficulty"] == "Easy":
                info = build_path_info(g, [src, tgt], score, scorer)
                if info:
                    easy_candidates.append(info)

    # ---- Medium: enumerate 2-hop directed paths with Medium score ----
    medium_candidates = []
    seen_2hop = set()
    for mid in event_ids:
        for src, _, _ in g.get_out_neighbors(mid):
            for tgt, _, _ in g.get_out_neighbors(mid):
                if src == tgt or src not in event_ids or tgt not in event_ids:
                    continue
                key = (src, mid, tgt)
                if key in seen_2hop:
                    continue
                seen_2hop.add(key)
                score = scorer.score_path([src, mid, tgt])
                if score["difficulty"] == "Medium":
                    info = build_path_info(g, [src, mid, tgt], score, scorer)
                    if info:
                        medium_candidates.append(info)

    # ---- Hard: 3-hop directed paths with tightened criteria ----
    # Use BFS from sampled start nodes to avoid combinatorial explosion
    MAX_HARD_TOTAL = 300
    MAX_START_NODES = 20

    hard_candidates = []
    seen_3hop = set()
    path_counter = [0]

    start_nodes = rng.sample(list(event_ids), min(len(event_ids), MAX_START_NODES))

    def bfs_3hop(start):
        """BFS up to 3 hops from start, collect hard paths."""
        # Queue: (current_node, path_so_far)
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
                score = scorer.score_path(path)
                if score["difficulty"] == "Hard":
                    path_subtypes = scorer.get_path_relation_subtypes(path)
                    type_set = set()
                    for rs in path_subtypes:
                        if rs.startswith("CAUSE"):
                            type_set.add("CAUSE")
                        elif rs.startswith("TEMPORAL"):
                            type_set.add("TEMPORAL")
                        elif rs.startswith("SUBEVENT"):
                            type_set.add("SUBEVENT")

                    cond1 = score["RD"] >= 2
                    cond2 = score["ES"] >= 3 and (len(type_set) >= 2 or "CAUSE" in type_set)
                    if cond1 or cond2:
                        info = build_path_info(g, path, score, scorer)
                        if info:
                            hard_candidates.append(info)
                continue

            if len(path) > 3:
                continue

            for tgt, _, _ in g.get_out_neighbors(current):
                if tgt in path:  # avoid cycles within this path
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
    parser.add_argument("--split", default="valid")  # formal eval on valid (test has no relation annotations)
    parser.add_argument("--num_docs", type=int, default=100)  # larger sample for formal eval
    parser.add_argument("--samples_per_level", type=int, default=30)
    parser.add_argument("--output_dir", default="event_qg/outputs")
    parser.add_argument("--tune_split", default="train")  # for hyperparameter tuning
    parser.add_argument("--tune_docs", type=int, default=20)  # small tune set
    args = parser.parse_args()

    # ---- Step 1: tune on small train split ----
    print("=== Tuning phase: small train split ===")
    tune_file = Path(args.data_dir) / f"{args.tune_split}.jsonl"
    tune_docs = load_jsonl(tune_file, n=args.tune_docs)
    print(f"Loaded {len(tune_docs)} tune docs")

    tune_stats = {"Easy": 0, "Medium": 0, "Hard": 0}
    tune_rel_dist = defaultdict(int)
    for i, doc in enumerate(tune_docs):
        g = EventGraph(doc)
        seed = hash(doc.get("id", str(i))) % (2**31)
        rng = random.Random(seed)
        target = {"Easy": 5, "Medium": 5, "Hard": 5}
        sampled = sample_from_doc(g, target, rng)
        for s in sampled:
            tune_stats[s["difficulty"]] += 1
            tune_rel_dist[s["relation_distribution"]] += 1

    print(f"Tune stats: {dict(tune_stats)}")
    print(f"Tune rel dist: {dict(tune_rel_dist)}")

    # ---- Step 2: formal sample from test/valid ----
    print(f"\n=== Formal phase: {args.split} split, {args.num_docs} docs ===")
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

    score_dist = defaultdict(int)
    for s in all_sampled:
        score_dist[s["difficulty_score"]] += 1
    print(f"Score distribution: {sorted(score_dist.items())}")

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
        "score_distribution": dict(score_dist),
        "tune_stats": tune_stats,
        "tune_rel_dist": dict(tune_rel_dist)
    }
    report_path = Path(args.output_dir) / "path_sampling_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()