"""Narrative Evidence Graph extractor for FairytaleQA.

Extracts structured graphs from narrative QA evidence chains.
Each graph captures causal/motivational/temporal relationships between
story events, grounded in the required evidence sentences.

Used for difficulty-controlled QG: graphs make Hard reasoning chains
explicit and verifiable.
"""
import json
import re

from dcqg.utils.api_client import call_openai_compatible
from dcqg.utils.config import get_api_config


# --- Schema constants ---

VALID_NODE_TYPES = {
    "action", "state", "emotion", "goal", "motivation", "belief",
    "outcome", "consequence", "problem", "attempt", "resolution", "description",
}

VALID_EDGE_RELATIONS = {
    "temporal_before", "causes", "motivates", "explains",
    "results_in", "enables", "prevents", "contrasts_with",
    "same_character", "supports_inference",
}

VALID_EVIDENCE_ROLES = {"anchor", "bridge", "answer", "answer_bridge", "context"}
VALID_NECESSITY = {"weak", "partial", "strong"}
VALID_CONFIDENCE = {"high", "medium", "low"}


def _split_sentences(text):
    """Split text into sentences. Same as fairytale_evidence_audit._split_sentences."""
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if s.strip()]


def _build_graph_prompt(candidate, difficulty="Hard"):
    """Build LLM prompt for narrative evidence graph extraction."""
    section = candidate.get("story_section", "")
    sentences = _split_sentences(section)
    sent_lines = "\n".join(f"  [S{j}] {s}" for j, s in enumerate(sentences))

    req_ids = candidate.get("required_evidence_sentences", [])
    bridge_ids = candidate.get("bridge_sentence_ids", [])
    answer = candidate.get("answer", "") or candidate.get("answer1", "")
    answer2 = candidate.get("answer2", "")
    if answer2:
        answer += f" / {answer2}"

    reasoning_op = candidate.get("reasoning_operation", "N/A")
    necessity_type = candidate.get("necessity_type", "N/A")

    # Build explicit sentence listing for required evidence
    req_sent_listing = []
    for sid in req_ids:
        role_tag = " [BRIDGE]" if sid in bridge_ids else ""
        if sid < len(sentences):
            req_sent_listing.append(f"  [S{sid}]{role_tag} {sentences[sid]}")
    req_sent_text = "\n".join(req_sent_listing)

    # Difficulty-aware mandatory requirements
    if difficulty == "Easy":
        mandatory = f"""## MANDATORY requirements

1. Create at least 1 node for the required evidence sentence(s).
2. For the sentence that directly states the answer, set evidence_role="answer".
3. Edges are optional for Easy questions.
4. Do NOT return empty nodes list."""
    elif difficulty == "Medium":
        mandatory = f"""## MANDATORY requirements

1. Create nodes for each required evidence sentence (total {len(req_ids)} nodes minimum).
2. For the sentence(s) that directly state the answer, set evidence_role="answer".
3. Create at least 1 edge showing how the nodes connect.
4. Do NOT return empty nodes/edges lists."""
    else:  # Hard
        mandatory = f"""## MANDATORY requirements

1. Create EXACTLY one node per required evidence sentence (total {len(req_ids)} nodes minimum).
2. For sentences marked [BRIDGE], set evidence_role="bridge" on that node.
3. For the sentence(s) that directly state the answer, set evidence_role="answer".
4. For the sentence that provides the starting context, set evidence_role="anchor".
5. Create edges showing how the nodes connect causally/temporally to the answer.
6. At least 2 edges with necessity="strong".
7. Do NOT return empty nodes/edges lists. You MUST extract at least {len(req_ids)} nodes."""

    return f"""You are an expert narrative analyst. Extract a Narrative Evidence Graph
from the story section below, grounded in the required evidence sentences.

## Story

Story: {candidate.get("story_name", "unknown")}
Question: {candidate.get("question", "")}
Answer: {answer}

Reasoning operation: {reasoning_op}
Necessity type: {necessity_type}
Target difficulty: {difficulty}

## Required evidence sentences (you MUST create nodes for ALL of these)

{req_sent_text}

Required evidence sentence IDs: {req_ids}
Bridge sentence IDs (marked [BRIDGE] above): {bridge_ids}

{mandatory}

## Node schema

Each node: {{"id": "N1", "type": "action|state|emotion|goal|motivation|belief|outcome|consequence|problem|attempt|resolution|description", "sentence_id": int, "text": "what happens in this sentence", "participants": ["names"], "evidence_role": "anchor|bridge|answer|context", "confidence": "high|medium|low"}}

## Edge schema

Each edge: {{"source": "N1", "target": "N2", "relation": "temporal_before|causes|motivates|explains|results_in|enables|prevents|contrasts_with|same_character|supports_inference", "necessity": "strong|partial|weak", "reason": "brief reason"}}

## Output

Return a JSON object with "nodes" and "edges" arrays:
{{"nodes": [...], "edges": [...]}}

Return ONLY the JSON object, no other text."""


def _parse_graph_response(resp):
    """Parse LLM response into (nodes, edges, parse_ok)."""
    if not resp:
        return None, None, False

    for attempt_fn in [
        lambda r: json.loads(r),
        lambda r: json.loads(r[r.index("{"):r.rindex("}") + 1]),
        lambda r: json.loads(
            re.sub(r'^```(?:json)?\s*', '', r.strip(), flags=re.IGNORECASE)
            .rstrip('`').strip()
        ),
    ]:
        try:
            data = attempt_fn(resp)
            if isinstance(data, dict) and "nodes" in data and "edges" in data:
                nodes = data["nodes"]
                edges = data["edges"]
                if isinstance(nodes, list) and isinstance(edges, list):
                    return nodes, edges, True
        except (json.JSONDecodeError, ValueError):
            pass

    return None, None, False


def _validate_graph(nodes, edges, req_ids, bridge_ids, difficulty="Hard"):
    """Validate graph structure. Returns (valid, reason, diagnostics).

    Difficulty-aware validation:
    - Easy: allow one answer node, no edge/bridge/path requirement
    - Medium: require >= 2 nodes if available, >= 1 edge, bridge check only if bridge_ids non-empty
    - Hard: unchanged (>= 3 nodes, >= 2 edges, all checks)
    """
    reasons = []
    req_set = set(req_ids)
    bridge_set = set(bridge_ids)

    if not isinstance(nodes, list) or not isinstance(edges, list):
        return False, "nodes or edges not lists", {}

    # Check 8: no empty text
    for n in nodes:
        if not isinstance(n, dict):
            return False, "node is not a dict", {}
        if not n.get("text", "").strip():
            reasons.append(f"node {n.get('id', '?')} has empty text")

    # Check 9: participants list exists
    for n in nodes:
        if "participants" not in n:
            reasons.append(f"node {n.get('id', '?')} missing participants")

    # Difficulty-aware node/edge thresholds
    if difficulty == "Easy":
        min_nodes = max(1, len(req_ids))
        min_edges = 0
    elif difficulty == "Medium":
        min_nodes = max(2, len(req_ids))
        min_edges = 1
    else:  # Hard
        min_nodes = 3
        min_edges = 2

    # Check 5: node count
    if len(nodes) < min_nodes:
        reasons.append(f"only {len(nodes)} nodes (need >= {min_nodes})")

    # Check 6: edge count
    if len(edges) < min_edges:
        reasons.append(f"only {len(edges)} edges (need >= {min_edges})")

    # Build node ID set
    node_ids = {n.get("id") for n in nodes if isinstance(n, dict)}

    # Check 1: node sentence_id in required_evidence_sentences
    for n in nodes:
        sid = n.get("sentence_id")
        if isinstance(sid, (int, float)):
            sid = int(sid)
        if sid not in req_set:
            reasons.append(f"node {n.get('id', '?')} sentence_id {sid} not in required evidence")

    # Check 2: bridge_sentence_ids must have bridge-role nodes
    # Skip for Easy (bridge_set often empty) and Medium when no bridge sentences
    if difficulty == "Hard" or (difficulty == "Medium" and bridge_set):
        bridge_sids_covered = set()
        for n in nodes:
            if n.get("evidence_role") in ("bridge", "answer_bridge"):
                sid = n.get("sentence_id")
                if isinstance(sid, (int, float)):
                    bridge_sids_covered.add(int(sid))
        missing_bridge = bridge_set - bridge_sids_covered
        if missing_bridge:
            reasons.append(f"bridge sentences {sorted(missing_bridge)} have no bridge-role node")

    # Check 3: answer node exists (answer or answer_bridge)
    has_answer = any(n.get("evidence_role") in ("answer", "answer_bridge") for n in nodes)
    if not has_answer:
        reasons.append("no answer-role node found")

    # Check 4: all edges reference existing node IDs
    for e in edges:
        src = e.get("source")
        tgt = e.get("target")
        if src not in node_ids:
            reasons.append(f"edge source {src} not in nodes")
        if tgt not in node_ids:
            reasons.append(f"edge target {tgt} not in nodes")

    # Check 7: at least one connected path to answer node
    # Skip for Easy (single node has no path) and Medium (2-node path is trivial)
    if difficulty == "Hard":
        answer_ids = {n.get("id") for n in nodes if n.get("evidence_role") in ("answer", "answer_bridge")}
        if answer_ids and node_ids:
            adj = {}
            for e in edges:
                s, t = e.get("source"), e.get("target")
                if s and t:
                    adj.setdefault(s, []).append(t)
            reachable = False
            for start in node_ids - answer_ids:
                visited = {start}
                queue = [start]
                while queue:
                    cur = queue.pop(0)
                    if cur in answer_ids:
                        reachable = True
                        break
                    for nxt in adj.get(cur, []):
                        if nxt not in visited:
                            visited.add(nxt)
                            queue.append(nxt)
                if reachable:
                    break
            if not reachable:
                reasons.append("no connected path to answer node")

    if not reasons:
        return True, "all checks passed", {
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    return False, "; ".join(reasons[:5]), {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "fail_reasons": reasons,
    }


def _normalize_node(n, idx):
    """Normalize and validate a node dict."""
    node_id = n.get("id", f"N{idx + 1}")
    ntype = n.get("type", "description")
    if ntype not in VALID_NODE_TYPES:
        ntype = "description"

    sid = n.get("sentence_id")
    if isinstance(sid, (int, float)):
        sid = int(sid)
    else:
        sid = 0

    role = n.get("evidence_role", "context")
    if role not in VALID_EVIDENCE_ROLES:
        role = "context"

    conf = n.get("confidence", "medium")
    if conf not in VALID_CONFIDENCE:
        conf = "medium"

    participants = n.get("participants", [])
    if not isinstance(participants, list):
        participants = []

    return {
        "id": node_id,
        "type": ntype,
        "sentence_id": sid,
        "text": str(n.get("text", "")),
        "participants": participants,
        "evidence_role": role,
        "source": "llm_extracted",
        "confidence": conf,
    }


def _normalize_edge(e):
    """Normalize and validate an edge dict."""
    relation = e.get("relation", "supports_inference")
    if relation not in VALID_EDGE_RELATIONS:
        relation = "supports_inference"

    necessity = e.get("necessity", "partial")
    if necessity not in VALID_NECESSITY:
        necessity = "partial"

    sids = e.get("sentence_ids", [])
    if not isinstance(sids, list):
        sids = []

    return {
        "source": str(e.get("source", "")),
        "target": str(e.get("target", "")),
        "relation": relation,
        "sentence_ids": [int(s) for s in sids if isinstance(s, (int, float))],
        "necessity": necessity,
        "reason": str(e.get("reason", "")),
    }


class NarrativeGraphExtractor:
    """Extract narrative evidence graphs from FairytaleQA candidates."""

    def __init__(self, model=None, max_retries=2):
        cfg = get_api_config()
        self.api_url = cfg["SILICONFLOW_API_URL"]
        self.api_key = cfg["SILICONFLOW_API_KEY"]
        self.model = model or cfg["JUDGE_MODEL"]
        self.max_retries = max_retries

    def extract(self, candidate, difficulty=None):
        """Extract a narrative graph from a single candidate.

        Args:
            candidate: dict with story_section, required_evidence_sentences, etc.
            difficulty: "Easy", "Medium", or "Hard". If None, inferred from candidate.

        Returns graph record dict.
        """
        if difficulty is None:
            difficulty = candidate.get("evidence_difficulty", "Hard")

        req_ids = candidate.get("required_evidence_sentences", [])
        bridge_ids = candidate.get("bridge_sentence_ids", [])

        prompt = _build_graph_prompt(candidate, difficulty=difficulty)

        raw_resp = None
        nodes = None
        edges = None
        parse_ok = False

        # Difficulty-aware min_nodes for retry logic
        if difficulty == "Easy":
            min_nodes = max(1, len(req_ids))
        elif difficulty == "Medium":
            min_nodes = max(2, len(req_ids))
        else:
            min_nodes = max(3, len(req_ids))

        for attempt in range(self.max_retries + 1):
            try:
                resp = call_openai_compatible(
                    prompt,
                    api_url=self.api_url,
                    api_key=self.api_key,
                    model=self.model,
                    temperature=0.0,
                    max_tokens=3000,
                    json_mode=True,
                    system="You are a precise narrative graph extractor. Return only valid JSON.",
                    timeout=120,
                )
                raw_resp = resp
                nodes, edges, parse_ok = _parse_graph_response(resp)
                if parse_ok and nodes and len(nodes) >= min_nodes:
                    break
                if parse_ok and attempt < self.max_retries:
                    prompt = prompt + f"\n\nPREVIOUS ATTEMPT returned only {len(nodes or [])} nodes. You MUST return at least {min_nodes} nodes, one per required evidence sentence."
            except Exception as e:
                raw_resp = f"ERROR: {e}"
                if attempt == self.max_retries:
                    break

        # Normalize
        if nodes is not None and edges is not None:
            nodes = [_normalize_node(n, i) for i, n in enumerate(nodes)]
            edges = [_normalize_edge(e) for e in edges]
        else:
            nodes = []
            edges = []

        # Post-process: auto-label bridge nodes
        bridge_set = set(bridge_ids)
        for n in nodes:
            sid = n.get("sentence_id")
            role = n.get("evidence_role")
            if sid in bridge_set:
                if role == "answer":
                    n["evidence_role"] = "answer_bridge"
                elif role == "context":
                    n["evidence_role"] = "bridge"

        # Validate with difficulty-aware rules
        graph_valid, validation_reason, diagnostics = _validate_graph(
            nodes, edges, req_ids, bridge_ids, difficulty=difficulty
        )

        # Build graph record
        record = {
            "story_name": candidate.get("story_name", ""),
            "question": candidate.get("question", ""),
            "answer": candidate.get("answer", "") or candidate.get("answer1", ""),
            "target_difficulty": difficulty,
            "attribute": candidate.get("attribute", ""),
            "reasoning_operation": candidate.get("reasoning_operation", ""),
            "necessity_type": candidate.get("necessity_type", ""),
            "required_evidence_sentences": req_ids,
            "bridge_sentence_ids": bridge_ids,
            "nodes": nodes,
            "edges": edges,
            "graph_valid": graph_valid,
            "graph_validation_reason": validation_reason,
            "diagnostics": diagnostics,
            "trace": {
                "prompt": prompt,
                "raw": raw_resp or "",
                "parse_ok": parse_ok,
                "model": self.model,
            },
        }

        return record
