"""
Four-dimensional difficulty scoring for event reasoning paths.
RD uses full relation subtype (e.g., TEMPORAL/BEFORE, CAUSE/PRECONDITION, SUBEVENT).

Dimensions:
  PL (Path Length): 1 hop=1, 2 hops=2, >=3 hops=3
  RD (Relation Diversity): 1 subtype=1, 2 subtypes=2, 3 subtypes=3
  ES (Evidence Span): unique supporting sentences -> <=2=1, 3-5=2, >=6=3
  EA (Event Ambiguity): none=1, moderate=2, high=3

Difficulty Score: D = PL + RD + ES + EA (4-12)
  Easy: 4-6
  Medium: 7-9
  Hard: 10-12
"""
from collections import defaultdict


class DifficultyScorer:
    """Compute four-dimensional difficulty score for an event reasoning path."""

    def __init__(self, event_graph):
        self.g = event_graph

    def compute_path_length_score(self, path):
        """PL: 1 hop=1, 2 hops=2, >=3 hops=3"""
        hops = len(path) - 1
        if hops <= 1:
            return 1
        elif hops == 2:
            return 2
        else:
            return 3

    def compute_relation_diversity_score(self, path):
        """
        RD: number of distinct relation subtypes on the path (full subtype string).
        e.g., TEMPORAL/BEFORE, CAUSE/PRECONDITION, SUBEVENT.
        """
        rel_subtypes = set()
        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]
            # Try forward direction first, then reverse (bidirectional check)
            found = False
            for direction in [(src, tgt), (tgt, src)]:
                s, t = direction
                for out_tgt, edge_type, edge_sub in self.g.get_out_neighbors(s):
                    if out_tgt == t:
                        key = f"{edge_type}/{edge_sub}" if edge_sub else edge_type
                        rel_subtypes.add(key)
                        found = True
                        break
                if found:
                    break

        n = len(rel_subtypes)
        if n <= 1:
            return 1
        elif n == 2:
            return 2
        else:
            return 3

    def get_supporting_sentences(self, path):
        """
        Get supporting sentences: path event sentences + ±1 sentence context.
        Returns list of (sent_id, sentence_text) with ±1 sentence window.
        """
        path_sent_ids = set()
        for eid in path:
            info = self.g.get_event_info(eid)
            path_sent_ids.add(info.get("sent_id", 0))

        # Expand by ±1
        expanded = set()
        for sid in path_sent_ids:
            for s in range(max(0, sid - 1), min(len(self.g.sentences), sid + 2)):
                expanded.add(s)

        return sorted(expanded)

    def compute_evidence_span_score(self, path):
        """
        ES: number of unique supporting sentences (path events + ±1 context sentences).
        Bucketed: <=2=1, 3-5=2, >=6=3.
        """
        supporting = self.get_supporting_sentences(path)
        n = len(supporting)
        if n <= 2:
            return 1
        elif n <= 5:
            return 2
        else:
            return 3

    def compute_event_ambiguity_score(self, path):
        """
        EA: check for coreference chains, similar event types, multiple candidates
        in the ±2 sentence window around path events.
        none=1, moderate=2, high=3.
        """
        path_events = {eid for eid in path}
        path_sent_ids = set()
        for eid in path:
            info = self.g.get_event_info(eid)
            path_sent_ids.add(info.get("sent_id", 0))

        # Get all events in ±2 sentence window
        nearby = []
        for eid, e in self.g.events.items():
            if eid in path_events:
                continue
            mentions = e.get("mention", [])
            if mentions:
                sid = mentions[0].get("sent_id", 0)
                if any(abs(sid - psid) <= 2 for psid in path_sent_ids):
                    nearby.append((eid, e, sid))

        if not nearby:
            return 1  # none

        ambiguity = 0

        # 1. Same event type in nearby sentences
        path_types = {self.g.get_event_info(eid)["type"] for eid in path}
        same_type_count = sum(1 for _, e, _ in nearby if e.get("type", "") in path_types)
        if same_type_count >= 2:
            ambiguity += 2
        elif same_type_count == 1:
            ambiguity += 1

        # 2. Similar trigger words
        path_triggers = {self.g.get_event_info(eid)["trigger"].lower() for eid in path}
        similar_count = sum(
            1 for _, e, _ in nearby
            if any(m.get("trigger_word", "").lower() in path_triggers for m in e.get("mention", []))
        )
        if similar_count >= 2:
            ambiguity += 2
        elif similar_count == 1:
            ambiguity += 1

        # 3. Multiple events sharing relation subtypes with the path
        path_rel_subtypes = set()
        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]
            for out_tgt, edge_type, edge_sub in self.g.get_out_neighbors(src):
                if out_tgt == tgt:
                    key = f"{edge_type}/{edge_sub}" if edge_sub else edge_type
                    path_rel_subtypes.add(key)

        candidate_count = 0
        for eid, _, _ in nearby:
            for out_tgt, edge_type, edge_sub in self.g.get_out_neighbors(eid):
                key = f"{edge_type}/{edge_sub}" if edge_sub else edge_type
                if key in path_rel_subtypes:
                    candidate_count += 1
                    break

        if candidate_count >= 3:
            ambiguity += 2
        elif candidate_count == 2:
            ambiguity += 1

        if ambiguity == 0:
            return 1
        elif ambiguity <= 2:
            return 2
        else:
            return 3

    def score_path(self, path):
        """
        Compute full four-dimensional score for a path.
        Returns dict with PL, RD, ES, EA, D, difficulty level.
        """
        pl = self.compute_path_length_score(path)
        rd = self.compute_relation_diversity_score(path)
        es = self.compute_evidence_span_score(path)
        ea = self.compute_event_ambiguity_score(path)
        D = pl + rd + es + ea

        if D <= 6:
            level = "Easy"
        elif D <= 9:
            level = "Medium"
        else:
            level = "Hard"

        return {
            "PL": pl,
            "RD": rd,
            "ES": es,
            "EA": ea,
            "D": D,
            "difficulty": level
        }

    def get_path_relation_subtypes(self, path):
        """Return list of full relation subtype strings for each hop in path."""
        subtypes = []
        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]
            key = None
            for direction in [(src, tgt), (tgt, src)]:
                s, t = direction
                for out_tgt, edge_type, edge_sub in self.g.get_out_neighbors(s):
                    if out_tgt == t:
                        key = f"{edge_type}/{edge_sub}" if edge_sub else edge_type
                        break
                if key:
                    break
            subtypes.append(key or "UNKNOWN")
        return subtypes


def compute_all_dimensions(graph, path):
    scorer = DifficultyScorer(graph)
    return scorer.score_path(path)