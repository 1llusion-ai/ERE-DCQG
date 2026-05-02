"""
Path-level instance sampler with hop-based difficulty.
Difficulty is determined solely by hop count:
  Easy = 1 hop, Medium = 2 hops, Hard = 3 hops.
Directed paths: only follow outgoing relation direction.
"""
import random
from collections import defaultdict

from dcqg.graph import EventGraph
from dcqg.utils.jsonl import read_jsonl


def _get_supporting_sentences(g, path):
    """Get supporting sentence IDs: path event sentences +/- 1 sentence window."""
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
