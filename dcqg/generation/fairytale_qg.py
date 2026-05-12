"""FairytaleQA evidence-role-aware question generation.

Four methods for difficulty-controlled QG on narrative QA:
  1. Direct QG: story + answer + difficulty definition, no graph
  2. ICL QG: same + few-shot examples
  3. SelfRefine QG: generate + critique + revise, no graph
  4. Ours: narrative evidence graph guided QG

All methods use answer-conditioned QG:
  Input = story_section + target_answer + target_difficulty
  Output = generated_question
"""
import json
import re
import time

from dcqg.utils.api_client import call_openai_compatible
from dcqg.utils.config import get_api_config
from dcqg.generation.parser import parse_json_response
from dcqg.question_filter.grammar import grammar_filter


# ── Difficulty definitions ─────────────────────────────────────

DIFFICULTY_INSTRUCTIONS = {
    "Easy": (
        "The question should be answerable from one sentence in the story. "
        "It asks for information directly stated in or near the answer sentence. "
        "No bridge or summary reasoning needed."
    ),
    "Medium": (
        "The question requires connecting two pieces of information: "
        "an anchor/context sentence plus the answer sentence. "
        "It may require simple temporal, causal, or referential linking."
    ),
    "Hard": (
        "The question requires three or more pieces of evidence working together. "
        "The answer is NOT identifiable from the answer sentence alone. "
        "The question asks about motivation, cause, or reasoning that spans multiple sentences. "
        "Without the bridge sentences, the answer becomes ambiguous or unanswerable."
    ),
}


def difficulty_instruction(difficulty):
    """Return the difficulty definition string."""
    return DIFFICULTY_INSTRUCTIONS.get(difficulty, DIFFICULTY_INSTRUCTIONS["Hard"])


# ── Answer-role-aware focus ────────────────────────────────────

ANSWER_FOCUS_TEMPLATES = {
    "motivation": {
        "label": "motivation / underlying purpose",
        "strategy": (
            "Ask about the LARGER PURPOSE or UNDERLYING MOTIVATION. "
            "Good: 'What larger goal was the character pursuing by doing X?' "
            "Good: 'What deeper reason drove the character's actions?' "
            "BAD: 'Why did the character do X?' (too local/immediate) "
            "The target answer IS the motivation — the question must frame it as the driving force, not an immediate cause."
        ),
        "example_question": "What larger goal was the character pursuing when they decided to act?",
    },
    "bridge": {
        "label": "connecting event / turning point",
        "strategy": (
            "Ask about the CONNECTING EVENT or TURNING POINT in the chain. "
            "Good: 'What development connected the earlier event to the later outcome?' "
            "Good: 'What turning point led from the initial situation to the final result?' "
            "The target answer is a link in the reasoning chain — frame it as the pivotal connection."
        ),
        "example_question": "What pivotal development connected the initial event to the final outcome?",
    },
    "outcome": {
        "label": "final outcome / eventual result",
        "strategy": (
            "Ask about the FINAL OUTCOME or EVENTUAL RESULT. "
            "Good: 'What ultimately resulted from the chain of events?' "
            "Good: 'What was the final consequence of the character's journey?' "
            "BAD: 'What happened next?' (too local) "
            "The target answer is the end of a causal chain — frame it as the ultimate result."
        ),
        "example_question": "What ultimately resulted from the earlier chain of events?",
    },
    "count": {
        "label": "repeated pattern / frequency",
        "strategy": (
            "Ask about a REPEATED PATTERN or FREQUENCY. "
            "Good: 'How many times did the character perform this action?' "
            "Good: 'What pattern emerged from the character's repeated behavior?' "
            "The target answer is a count or pattern — ask about the frequency, not the reason."
        ),
        "example_question": "How many times did the character repeat this action?",
    },
    "state": {
        "label": "character state / feeling / belief",
        "strategy": (
            "Ask about the CHARACTER'S STATE, FEELING, or BELIEF. "
            "Good: 'How did the character come to feel this way?' "
            "Good: 'What state of mind was the character in after the events?' "
            "BAD: 'Why did the character feel X?' (implies the feeling is the cause, not the result) "
            "The target answer describes a state — frame the question to ask about how that state emerged."
        ),
        "example_question": "How did the character come to be in this state?",
    },
}


def classify_answer_focus(nodes, target_answer):
    """Classify the target answer's role in the narrative graph.

    Returns (focus_key, answer_role, answer_node_type) where:
      focus_key: one of motivation, bridge, outcome, count, state
      answer_role: the evidence_role of the matching node
      answer_node_type: the node type of the matching node
    """
    # Check if target answer looks like a count/pattern
    ans_lower = (target_answer or "").strip().lower().rstrip(" .")
    count_words = {"once", "twice", "three times", "four times", "five times",
                   "one time", "two times", "many times", "several times",
                   "first", "second", "third", "1", "2", "3", "4", "5"}
    if ans_lower in count_words or (ans_lower.endswith(" times") and ans_lower.split()[0].isdigit()):
        return "count", "count_pattern", "count"

    # Find answer or answer_bridge nodes
    answer_nodes = [n for n in nodes if n.get("evidence_role") in ("answer", "answer_bridge")]
    if not answer_nodes:
        # Fallback: use any node
        answer_nodes = nodes[:1] if nodes else []

    if not answer_nodes:
        return "state", "unknown", "unknown"

    # Use the first answer node (primary answer evidence)
    an = answer_nodes[0]
    role = an.get("evidence_role", "answer")
    ntype = an.get("type", "state")

    # Classify based on role + type
    if role == "anchor":
        if ntype in ("motivation", "goal", "emotion"):
            return "motivation", role, ntype
        return "state", role, ntype

    if role in ("bridge", "answer_bridge"):
        if ntype in ("outcome", "consequence", "result"):
            return "outcome", role, ntype
        if ntype in ("motivation", "goal", "emotion"):
            return "motivation", role, ntype
        return "bridge", role, ntype

    # role == "answer"
    if ntype in ("outcome", "consequence", "result"):
        return "outcome", role, ntype
    if ntype in ("motivation", "goal", "emotion"):
        return "motivation", role, ntype
    if ntype in ("state", "belief"):
        return "state", role, ntype
    if ntype in ("action", "description", "problem"):
        return "bridge", role, ntype

    return "state", role, ntype


# ── Graph-structured difficulty policy ─────────────────────────

RELATION_PRIORITY = {
    "motivates": 6, "causes": 5, "explains": 4,
    "results_in": 3, "enables": 2, "prevents": 2,
    "temporal_before": 1, "contrasts_with": 1,
    "same_character": 0, "supports_inference": 1,
}

RELATION_WORDING = {
    "motivates": "Ask about goal, reason, intention, motivation.",
    "causes": "Ask about consequence or outcome.",
    "results_in": "Ask about consequence or outcome.",
    "explains": "Ask what explains a later state/action.",
    "enables": "Ask what made something possible.",
    "prevents": "Ask what made something impossible.",
    "temporal_before": "Use only as ordering support, not as main Hard signal.",
}


def select_graph_substructure(nodes, edges, difficulty, target_answer):
    """Select a difficulty-appropriate graph substructure.

    Returns dict with:
      selected_nodes, selected_edges, graph_policy, graph_policy_reason,
      relation_chain, evidence_roles_used
    """
    if not nodes:
        return {
            "selected_nodes": [], "selected_edges": [],
            "graph_policy": "graph_invalid",
            "graph_policy_reason": "no nodes available",
            "relation_chain": [], "evidence_roles_used": [],
        }

    # Find answer node
    answer_nodes = [n for n in nodes if n.get("evidence_role") in ("answer", "answer_bridge")]
    answer_node = answer_nodes[0] if answer_nodes else nodes[0]

    def _edge_priority(e):
        return RELATION_PRIORITY.get(e.get("relation", ""), 0)

    def _edges_into(node_id):
        """Edges where node_id is the target (directed into)."""
        return [e for e in edges if e.get("target") == node_id]

    def _edges_touching(node_id):
        """Edges where node_id is source or target (undirected)."""
        return [e for e in edges if e.get("source") == node_id or e.get("target") == node_id]

    def _node_by_id(nid):
        for n in nodes:
            if n["id"] == nid:
                return n
        return None

    def _roles_and_chain(sel_nodes, sel_edges):
        roles = []
        for n in sel_nodes:
            r = n.get("evidence_role", "context")
            if r not in roles:
                roles.append(r)
        chain = [e.get("relation", "") for e in sel_edges]
        return roles, chain

    # ── Easy: answer node only ──
    if difficulty == "Easy":
        sel_nodes = [answer_node]
        sel_edges = []
        roles, chain = _roles_and_chain(sel_nodes, sel_edges)
        return {
            "selected_nodes": sel_nodes, "selected_edges": sel_edges,
            "graph_policy": "answer_only",
            "graph_policy_reason": f"Easy: using answer node {answer_node['id']} only",
            "relation_chain": chain, "evidence_roles_used": roles,
        }

    # ── Medium: answer + best one related node ──
    if difficulty == "Medium":
        into_edges = _edges_into(answer_node["id"])
        if into_edges:
            best_edge = max(into_edges, key=_edge_priority)
            other_id = best_edge["source"]
            other_node = _node_by_id(other_id)
            sel_nodes = [answer_node, other_node] if other_node else [answer_node]
            sel_edges = [best_edge]
            reason = f"Medium: answer {answer_node['id']} + {other_id} via {best_edge['relation']}"
        else:
            # Fallback: answer + any bridge/context node
            bridge_nodes = [n for n in nodes
                           if n.get("evidence_role") in ("bridge", "anchor", "context")
                           and n["id"] != answer_node["id"]]
            if bridge_nodes:
                other_node = bridge_nodes[0]
                sel_nodes = [answer_node, other_node]
                sel_edges = [e for e in _edges_touching(answer_node["id"])
                            if (e.get("source") == other_node["id"] or e.get("target") == other_node["id"])][:1]
                reason = f"Medium fallback: answer {answer_node['id']} + {other_node['id']} (no directed edge into answer)"
            else:
                sel_nodes = [answer_node]
                sel_edges = []
                reason = f"Medium fallback: answer node only (no related nodes found)"
        roles, chain = _roles_and_chain(sel_nodes, sel_edges)
        gp = "two_node_relation"
        return {
            "selected_nodes": sel_nodes, "selected_edges": sel_edges,
            "graph_policy": gp,
            "graph_policy_reason": reason,
            "relation_chain": chain, "evidence_roles_used": roles,
        }

    # ── Hard: 3+ node chain into answer ──
    # Build adjacency for path finding
    adj = {}  # node_id -> list of (edge, neighbor_id)
    for e in edges:
        src, tgt = e.get("source"), e.get("target")
        adj.setdefault(src, []).append((e, tgt))
        adj.setdefault(tgt, []).append((e, src))

    def _find_best_path(start_id, target_id, directed=True):
        """BFS to find best path from start to target, maximizing cumulative relation priority."""
        from collections import deque
        best_path = None
        best_score = -1
        queue = deque([(start_id, [(None, start_id)], 0)])
        visited = {start_id}
        max_depth = 5
        while queue:
            curr, path, score = queue.popleft()
            if len(path) > max_depth:
                continue
            if curr == target_id and len(path) > 1:
                if score > best_score:
                    best_score = score
                    best_path = path
                continue
            for edge, neigh in adj.get(curr, []):
                if neigh in visited:
                    continue
                # For directed, only follow edges where curr is source
                if directed and edge.get("source") != curr:
                    continue
                ep = _edge_priority(edge)
                visited.add(neigh)
                queue.append((neigh, path + [(edge, neigh)], score + ep))
        return best_path

    # Try directed paths from non-answer nodes into answer
    non_answer_ids = [n["id"] for n in nodes if n["id"] != answer_node["id"]]
    best_directed = None
    best_directed_score = -1
    for start_id in non_answer_ids:
        path = _find_best_path(start_id, answer_node["id"], directed=True)
        if path and len(path) >= 3:
            score = sum(_edge_priority(e) for e, _ in path if e)
            if score > best_directed_score:
                best_directed_score = score
                best_directed = path

    if best_directed and len(best_directed) >= 3:
        # Extract selected nodes and edges from path
        sel_node_ids = [nid for _, nid in best_directed]
        sel_edges_list = [e for e, _ in best_directed if e]
        sel_nodes = [_node_by_id(nid) for nid in sel_node_ids if _node_by_id(nid)]
        roles, chain = _roles_and_chain(sel_nodes, sel_edges_list)
        pure_temporal = all(r == "temporal_before" for r in chain) if chain else False
        reason = f"Hard: directed path {' → '.join(sel_node_ids)}"
        if pure_temporal:
            reason += " [pure_temporal_chain]"
        return {
            "selected_nodes": sel_nodes, "selected_edges": sel_edges_list,
            "graph_policy": "multi_node_chain",
            "graph_policy_reason": reason,
            "relation_chain": chain, "evidence_roles_used": roles,
        }

    # Undirected fallback
    best_undirected = None
    best_undirected_score = -1
    for start_id in non_answer_ids:
        path = _find_best_path(start_id, answer_node["id"], directed=False)
        if path and len(path) >= 3:
            score = sum(_edge_priority(e) for e, _ in path if e)
            if score > best_undirected_score:
                best_undirected_score = score
                best_undirected = path

    if best_undirected and len(best_undirected) >= 3:
        sel_node_ids = [nid for _, nid in best_undirected]
        sel_edges_list = [e for e, _ in best_undirected if e]
        sel_nodes = [_node_by_id(nid) for nid in sel_node_ids if _node_by_id(nid)]
        roles, chain = _roles_and_chain(sel_nodes, sel_edges_list)
        pure_temporal = all(r == "temporal_before" for r in chain) if chain else False
        reason = f"Hard: undirected_fallback path {' → '.join(sel_node_ids)}"
        if pure_temporal:
            reason += " [pure_temporal_chain]"
        return {
            "selected_nodes": sel_nodes, "selected_edges": sel_edges_list,
            "graph_policy": "multi_node_chain",
            "graph_policy_reason": reason,
            "relation_chain": chain, "evidence_roles_used": roles,
        }

    # Expand: answer + best adjacent strong edges
    touching = _edges_touching(answer_node["id"])
    touching.sort(key=_edge_priority, reverse=True)
    sel_nodes = [answer_node]
    sel_edges_list = []
    seen_ids = {answer_node["id"]}
    for e in touching[:2]:
        other_id = e["target"] if e["source"] == answer_node["id"] else e["source"]
        if other_id not in seen_ids:
            n = _node_by_id(other_id)
            if n:
                sel_nodes.append(n)
                sel_edges_list.append(e)
                seen_ids.add(other_id)
    roles, chain = _roles_and_chain(sel_nodes, sel_edges_list)
    reason = f"Hard: expanded from answer node with {len(sel_edges_list)} edges (no 3+ node path found)"
    return {
        "selected_nodes": sel_nodes, "selected_edges": sel_edges_list,
        "graph_policy": "multi_node_chain",
        "graph_policy_reason": reason,
        "relation_chain": chain, "evidence_roles_used": roles,
    }


# ── ICL examples (narrative domain) ────────────────────────────

NARRATIVE_ICL_EXAMPLES = {
    "Easy": """Example:
Story: The princess lived in a small castle by the river. Every morning she walked along the water.
Target answer: "by the river"
Output: {"question": "Where did the princess live?", "answer": "by the river", "reasoning_type": "direct"}""",

    "Medium": """Example:
Story: [S0] The king announced a contest. [S1] The bravest knight would win the princess's hand. [S2] Sir Arthur volunteered first.
Target answer: "Sir Arthur"
Output: {"question": "Who was the first to volunteer after the king announced the contest?", "answer": "Sir Arthur", "reasoning_type": "chain"}""",

    "Hard": """Example 1:
Story: [S0] The queen was jealous of the princess's beauty. [S1] She ordered a huntsman to take the princess into the forest. [S2] The huntsman could not bring himself to harm her. [S3] He returned with a boar's heart instead.
Target answer: "he could not bring himself to harm her"
Output: {"question": "Why did the huntsman return with a boar's heart instead of the princess's?", "answer": "he could not bring himself to harm her", "reasoning_type": "cross_sentence"}

Example 2:
Story: [S0] The princes learned that the youth had saved the princesses. [S1] A great jealousy took possession of them. [S2] They took counsel together how to get the better of the youth. [S3] They suddenly threw themselves on the youth and strangled him.
Target answer: "they were jealous of the youth"
Output: {"question": "Why did the princes suddenly attack their comrade who had saved the princesses?", "answer": "they were jealous of the youth", "reasoning_type": "cross_sentence"}""",
}


# ── Format helpers ─────────────────────────────────────────────

def _split_sentences(text):
    """Split text into sentences."""
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if s.strip()]


def _format_story_context(story_section, sentence_ids=None):
    """Format story section with sentence IDs.

    If sentence_ids is provided, mark those with *.
    """
    if not story_section:
        return ""
    sentences = _split_sentences(story_section)

    if sentence_ids is None:
        return "\n".join(f"[S{i}] {s}" for i, s in enumerate(sentences))

    sid_set = set(sentence_ids)
    lines = []
    for i, s in enumerate(sentences):
        marker = " *" if i in sid_set else ""
        lines.append(f"[S{i}] {s}{marker}")
    return "\n".join(lines)


def _format_graph_for_prompt(nodes, edges):
    """Format narrative graph nodes and edges for prompt."""
    node_lines = []
    for n in nodes:
        role = n.get("evidence_role", "context")
        ntype = n.get("type", "description")
        sid = n.get("sentence_id", "?")
        text = n.get("text", "")
        participants = ", ".join(n.get("participants", []))
        node_lines.append(
            f"  {n['id']} [{ntype}, role={role}, S{sid}]: {text}"
            f" (participants: {participants})"
        )

    edge_lines = []
    for e in edges:
        edge_lines.append(
            f"  {e['source']} --[{e['relation']}, necessity={e['necessity']}]--> {e['target']}: {e.get('reason', '')}"
        )

    return "Nodes:\n" + "\n".join(node_lines) + "\n\nEdges:\n" + "\n".join(edge_lines)


# ── Degenerate output detection ────────────────────────────────

def _is_degenerate(text):
    """Detect degenerate LLM output (dots, repetition, gibberish)."""
    if not text:
        return True
    t = text.strip()
    # All dots or punctuation
    if len(t) > 10 and all(c in '.?! \n' for c in t):
        return True
    # Repeated ". . ." or "... . ." pattern
    if re.search(r'(\.\s*){5,}', t):
        return True
    # Repetitive pattern (e.g., "WhyWhy why why")
    words = t.split()
    if len(words) > 5:
        unique = set(w.lower() for w in words)
        if len(unique) <= 2:
            return True
    # Same token repeats 3+ times consecutively
    for i in range(len(words) - 2):
        if words[i].lower() == words[i+1].lower() == words[i+2].lower():
            return True
    # Too short
    if len(t) < 5:
        return True
    # High punctuation/dot ratio (>40% non-alphanumeric)
    if len(t) > 20:
        non_alpha = sum(1 for c in t if not c.isalnum() and not c.isspace())
        if non_alpha / len(t) > 0.4:
            return True
    return False


def _question_length_ok(question):
    """Check question is 6-35 words and has no repeated token spam."""
    words = question.strip().split()
    if len(words) < 6 or len(words) > 35:
        return False
    # Same token repeats 3+ times consecutively
    for i in range(len(words) - 2):
        if words[i].lower() == words[i+1].lower() == words[i+2].lower():
            return False
    return True


# ── Prompt builders ────────────────────────────────────────────

def build_direct_prompt(story_section, target_answer, difficulty):
    """Direct QG: story + answer + difficulty definition. No graph, no examples."""
    ctx = _format_story_context(story_section)
    diff_def = difficulty_instruction(difficulty)

    hard_extra = ""
    if difficulty == "Hard":
        hard_extra = """
CRITICAL for Hard difficulty:
- The question MUST require reading at least 3 sentences to understand the answer.
- Ask "Why did [character] do [action]?" where the WHY is explained in a different sentence from the action.
- The reader needs: (1) sentence showing the motivation/cause, (2) sentence showing the development, (3) sentence showing the result/answer.
- If the answer could be found by reading just 1-2 sentences, the question is NOT Hard.
- Example: Story has [S0] "The queen was jealous", [S1] "She planned revenge", [S2] "She poisoned the apple". Hard question: "Why did the queen poison the apple?" (needs all 3 sentences)."""

    return f"""Generate one reading-comprehension question about the story.

Story:
{ctx}

Target answer: "{target_answer}"

Difficulty: {difficulty}
{diff_def}{hard_extra}

Requirements:
1. The question must be answerable using only the story above.
2. The natural answer to the question must be: "{target_answer}"
3. Do NOT mention "{target_answer}" in the question itself.
4. The question must start with a question word (Who/What/Where/When/Why/How) and end with "?".
5. Output exactly one JSON object, nothing else.

Output: {{"question": "...", "answer": "{target_answer}", "reasoning_type": "direct|chain|cross_sentence"}}"""


def build_icl_prompt(story_section, target_answer, difficulty):
    """ICL QG: story + answer + difficulty + few-shot examples. No graph."""
    ctx = _format_story_context(story_section)
    diff_def = difficulty_instruction(difficulty)
    examples = NARRATIVE_ICL_EXAMPLES.get(difficulty, NARRATIVE_ICL_EXAMPLES["Hard"])

    hard_extra = ""
    if difficulty == "Hard":
        hard_extra = """
CRITICAL: The question MUST require 3+ sentences to answer. Ask about motivation or cause that spans multiple sentences. The answer sentence alone should NOT be sufficient."""

    return f"""Generate one reading-comprehension question.

Examples of {difficulty} questions:
{examples}

Now generate your own:

Story:
{ctx}

Target answer: "{target_answer}"

Difficulty: {difficulty}
{diff_def}{hard_extra}

Requirements:
1. Answerable using only the story.
2. Natural answer must be: "{target_answer}"
3. Do NOT mention "{target_answer}" in the question.
4. Start with a question word, end with "?".
5. Output exactly one JSON object.

Output: {{"question": "...", "answer": "{target_answer}", "reasoning_type": "direct|chain|cross_sentence"}}"""


def build_self_refine_gen_prompt(story_section, target_answer, difficulty):
    """SelfRefine step 1: generate initial question (same as Direct)."""
    return build_direct_prompt(story_section, target_answer, difficulty)


def build_self_refine_critique_prompt(question, story_section, target_answer, difficulty):
    """SelfRefine step 2: critique the generated question."""
    ctx = _format_story_context(story_section)

    hard_check = ""
    if difficulty == "Hard":
        hard_check = """
6. Single-sentence check: Could this question be answered by reading JUST the sentence that contains the answer, without any other sentences? If yes, it fails the Hard requirement."""

    return f"""Critique this question:

Story:
{ctx}

Question: "{question}"
Target answer: "{target_answer}"
Target difficulty: {difficulty}

Check:
1. Grammar: Grammatically correct and natural?
2. Answerable: Can it be answered from the story alone?
3. Target match: Does the natural answer match "{target_answer}" in meaning?
4. Leakage: Does the question contain "{target_answer}"?
5. Difficulty match: Does it match {difficulty} level?{hard_check}

Reply ONLY: {{"issues": [], "overall_quality": "good"}} or {{"issues": ["..."], "overall_quality": "needs_revision"}}"""


def build_self_refine_revise_prompt(question, critique_text, story_section, target_answer, difficulty):
    """SelfRefine step 3: revise based on critique."""
    ctx = _format_story_context(story_section)
    diff_def = difficulty_instruction(difficulty)

    hard_extra = ""
    if difficulty == "Hard":
        hard_extra = """
- For Hard: the question must require reading 2+ sentences. Ask about WHY or HOW."""

    return f"""Fix the issues in this question.

Story:
{ctx}

Original: "{question}"
Issues found: {critique_text}
Target answer: "{target_answer}" (do NOT include in question)
Target difficulty: {difficulty}{hard_extra}

Output ONLY: {{"question": "...", "answer": "{target_answer}", "reasoning_type": "direct|chain|cross_sentence"}}"""


def build_ours_prompt(story_section, target_answer, difficulty, nodes, edges,
                     required_evidence_sentences, bridge_sentence_ids,
                     reasoning_operation, necessity_type, graph_substructure=None):
    """Ours: narrative evidence graph guided QG with graph-structured difficulty policy.

    When graph_substructure is provided, only the selected subgraph is shown
    in the prompt — this is the difficulty-control signal.
    """
    # Use selected subgraph if provided, otherwise fall back to full graph
    if graph_substructure and graph_substructure.get("selected_nodes"):
        prompt_nodes = graph_substructure["selected_nodes"]
        prompt_edges = graph_substructure.get("selected_edges", [])
        gp = graph_substructure.get("graph_policy", "unknown")
        gp_reason = graph_substructure.get("graph_policy_reason", "")
        gp_roles = graph_substructure.get("evidence_roles_used", [])
        gp_chain = graph_substructure.get("relation_chain", [])
    else:
        prompt_nodes = nodes
        prompt_edges = edges
        gp = "legacy"
        gp_reason = "no substructure provided"
        gp_roles = []
        gp_chain = []

    # Derive evidence sentence IDs from selected nodes
    selected_sids = sorted(set(n.get("sentence_id", 0) for n in prompt_nodes))
    ctx = _format_story_context(story_section, sentence_ids=selected_sids or required_evidence_sentences)
    diff_def = difficulty_instruction(difficulty)
    graph_str = _format_graph_for_prompt(prompt_nodes, prompt_edges)

    # Build evidence role summary from selected nodes
    role_summary = {}
    for n in prompt_nodes:
        role = n.get("evidence_role", "context")
        sid = n.get("sentence_id", "?")
        role_summary.setdefault(role, []).append(f"S{sid}")

    role_lines = []
    for role in ["anchor", "bridge", "answer", "answer_bridge", "context"]:
        if role in role_summary:
            role_lines.append(f"  {role}: {', '.join(role_summary[role])}")
    roles_str = "\n".join(role_lines)

    # Build edge chain description from selected edges
    edge_chain = " -> ".join(
        f"{e['source']}({e['relation']})" for e in prompt_edges
    )
    if prompt_edges:
        last_target = prompt_edges[-1].get("target", "")
        edge_chain += f" -> {last_target}"

    # Classify answer role and determine question focus
    focus_key, answer_role, answer_node_type = classify_answer_focus(nodes, target_answer)
    focus = ANSWER_FOCUS_TEMPLATES[focus_key]

    # Build answer focus guide
    focus_guide = f"""
ANSWER ROLE: The target answer "{target_answer}" has role={answer_role}, node_type={answer_node_type}.
QUESTION FOCUS: {focus['label']}
FOCUS STRATEGY: {focus['strategy']}"""

    # Relation-aware wording for selected edges
    relation_guidance = ""
    if gp_chain:
        wording_parts = []
        for rel in dict.fromkeys(gp_chain):  # unique, ordered
            if rel in RELATION_WORDING:
                wording_parts.append(f"  {rel}: {RELATION_WORDING[rel]}")
        if wording_parts:
            relation_guidance = "\nRELATION GUIDANCE:\n" + "\n".join(wording_parts)

    # Graph policy section
    policy_section = f"""
GRAPH POLICY: {gp}
Policy reason: {gp_reason}
Selected evidence roles: {', '.join(gp_roles) if gp_roles else 'none'}
Selected relation chain: {' → '.join(gp_chain) if gp_chain else 'none'}"""

    # Difficulty-specific guidance
    difficulty_guidance = ""
    if difficulty == "Easy":
        difficulty_guidance = """
EASY REQUIREMENTS:
- Use ONLY the answer node. The question must be answerable from one sentence.
- Do NOT mention bridge events. Do NOT ask motivation or causal-chain questions.
- Focus: direct fact, local action/state, local description."""
    elif difficulty == "Medium":
        difficulty_guidance = """
MEDIUM REQUIREMENTS:
- Use the answer node + one supporting node.
- Use relation-aware phrasing for one bridge relation only.
- Do NOT require full chain reasoning.
- Focus: local cause, local consequence, simple state/action explanation."""
    elif difficulty == "Hard":
        bridge_sents = [f"S{n['sentence_id']}" for n in prompt_nodes if n.get("evidence_role") in ("bridge", "answer_bridge")]
        anchor_sents = [f"S{n['sentence_id']}" for n in prompt_nodes if n.get("evidence_role") == "anchor"]
        answer_sents = [f"S{n['sentence_id']}" for n in prompt_nodes if n.get("evidence_role") in ("answer", "answer_bridge")]
        total_sents = len(anchor_sents) + len(bridge_sents) + len(answer_sents)

        difficulty_guidance = f"""
HARD REQUIREMENTS:
- Use the full reasoning chain (anchor → bridge → answer).
- The question must require reading ALL {total_sents} sentences: {anchor_sents} + {bridge_sents} + {answer_sents}.
- Prefer causal/motivation/explanation framing.
- The question must NOT be answerable from the answer sentence alone.
- If any sentence is removed, the reader cannot fully answer the question.
- CRITICAL: Match the question focus to the answer role. Do NOT ask a local/immediate question if the answer is about a larger purpose or final outcome."""

    # Forbidden patterns based on focus
    forbidden = ""
    if focus_key == "motivation":
        forbidden = '\nFORBIDDEN: Do NOT ask "Why did [character] do [specific action]?" — this is too local. Ask about the LARGER PURPOSE.'
    elif focus_key == "outcome":
        forbidden = '\nFORBIDDEN: Do NOT ask "What happened next?" — this is too local. Ask about the ULTIMATE RESULT.'
    elif focus_key == "state":
        forbidden = '\nFORBIDDEN: Do NOT ask "Why did [character] feel X?" as if the feeling is a cause. Ask HOW the character came to be in that state.'
    elif focus_key == "count":
        forbidden = '\nFORBIDDEN: Do NOT ask "Why did this happen?" — the answer is a COUNT, not a reason. Ask about frequency or pattern.'

    return f"""Generate a reading-comprehension question guided by the narrative evidence graph.

Story:
{ctx}

Target answer: "{target_answer}"

Difficulty: {difficulty}
{diff_def}

Selected Evidence Subgraph:
{graph_str}
{policy_section}

Evidence roles:
{roles_str}

Reasoning chain: {edge_chain}
Reasoning operation: {reasoning_operation}
Necessity type: {necessity_type}
{relation_guidance}
{focus_guide}{difficulty_guidance}{forbidden}

Requirements:
1. Answerable from the story alone.
2. Natural answer must be: "{target_answer}"
3. Do NOT mention "{target_answer}" in the question.
4. Start with a question word, end with "?".
5. Match the question focus to the answer role (see ANSWER ROLE and QUESTION FOCUS above).
6. Output exactly one JSON object.

Output: {{"question": "...", "answer": "{target_answer}", "reasoning_type": "direct|chain|cross_sentence"}}"""


def build_repair_prompt(story_section, target_answer, difficulty, focus_key,
                        required_evidence_sentences, graph_substructure=None):
    """Compact repair prompt for degenerate/parse failures. No graph dump.

    Uses selected subgraph nodes' sentence IDs when available, so repair
    attempts preserve the same graph_policy as the full prompt.
    """
    # Use selected subgraph sentence IDs if available
    if graph_substructure and graph_substructure.get("selected_nodes"):
        repair_sids = sorted(set(
            n.get("sentence_id", 0) for n in graph_substructure["selected_nodes"]
        ))
    else:
        repair_sids = required_evidence_sentences

    ctx = _format_story_context(story_section, sentence_ids=repair_sids or required_evidence_sentences)
    diff_def = difficulty_instruction(difficulty)
    focus = ANSWER_FOCUS_TEMPLATES[focus_key]

    gp = graph_substructure.get("graph_policy", "") if graph_substructure else ""
    gp_hint = f"\nGraph policy: {gp}" if gp else ""

    return f"""Generate ONE reading-comprehension question.

Story (evidence sentences only):
{ctx}

Target answer: "{target_answer}"
Difficulty: {difficulty} — {diff_def}
Question focus: {focus['label']}{gp_hint}

Rules:
1. Question must be answerable from the story above.
2. Natural answer must be: "{target_answer}"
3. Do NOT mention "{target_answer}" in the question.
4. Start with a question word, end with "?".
5. Question must be 6-35 words, grammatically correct.
6. No repeated words or filler tokens.
7. Output exactly one JSON object, nothing else.

Output: {{"question": "...", "answer": "{target_answer}", "reasoning_type": "direct|chain|cross_sentence"}}"""


# ── LLM call helpers ───────────────────────────────────────────

def _call_llm(prompt, temperature=0.1, max_tokens=250, timeout=90):
    """Single API call using json_mode=True. Returns raw text or error string."""
    cfg = get_api_config()
    try:
        resp = call_openai_compatible(
            prompt,
            api_url=cfg["SILICONFLOW_API_URL"],
            api_key=cfg["SILICONFLOW_API_KEY"],
            model=cfg["MODEL"],
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            system="Output ONLY a valid JSON object. Follow the question requirements EXACTLY.",
            timeout=timeout,
        )
        return resp.strip() if resp else "ERROR: empty response"
    except Exception as e:
        return f"ERROR: {e}"


def _call_judge(prompt, temperature=0.0, max_tokens=300, timeout=90):
    """Judge API call using json_mode=True to avoid truncation."""
    cfg = get_api_config()
    try:
        resp = call_openai_compatible(
            prompt,
            api_url=cfg["SILICONFLOW_API_URL"],
            api_key=cfg["SILICONFLOW_API_KEY"],
            model=cfg["JUDGE_MODEL"],
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            system="You are a precise JSON-only evaluator. Return only valid JSON.",
            timeout=timeout,
        )
        return resp if resp else "ERROR: empty response"
    except Exception as e:
        return f"ERROR: {e}"


def _parse_json(text):
    """Parse JSON from LLM response. Returns dict or None."""
    if not text or text.startswith("ERROR"):
        return None
    result = parse_json_response(text)
    if isinstance(result, dict):
        return result
    return None


def _validate_question(question, target_answer):
    """Basic validation of generated question. Returns (ok, reason)."""
    if not question or not question.strip():
        return False, "empty"
    q = question.strip()
    if "?" not in q:
        return False, "no question mark"
    if not q.endswith("?"):
        # Fix: allow if ? is near end
        if not q.rstrip().endswith("?"):
            return False, "does not end with ?"
    # Check answer leakage (exact match only, not substring)
    if target_answer:
        ans_clean = target_answer.strip().lower()
        q_lower = q.lower()
        # Check if the full answer phrase appears in the question
        if len(ans_clean) > 3 and ans_clean in q_lower:
            return False, "answer leakage"
    # Grammar filter
    grammar_ok, grammar_reason = grammar_filter(q)
    if not grammar_ok:
        return False, f"grammar: {grammar_reason}"
    return True, "ok"


# ── Generation functions ───────────────────────────────────────

def generate_direct(story_section, target_answer, difficulty, max_retries=2):
    """Generate question using Direct QG method."""
    question = ""
    reason = "unknown"
    gen = None
    last_prompt = ""
    last_raw = ""

    for attempt in range(max_retries + 1):
        prompt = build_direct_prompt(story_section, target_answer, difficulty)
        temp = 0.1 + min(attempt * 0.1, 0.3)
        raw = _call_llm(prompt, temperature=temp)
        last_prompt = prompt
        last_raw = raw

        if _is_degenerate(raw):
            reason = "degenerate output"
            continue

        gen = _parse_json(raw)
        question = gen.get("question", "") if gen else ""
        ok, reason = _validate_question(question, target_answer)
        if ok:
            return {
                "generated_question": question,
                "method": "Direct",
                "generation_prompt": prompt,
                "generation_raw": raw,
                "parse_ok": True,
            }, 1 + attempt

    return {
        "generated_question": question,
        "method": "Direct",
        "generation_prompt": last_prompt,
        "generation_raw": last_raw,
        "parse_ok": gen is not None,
        "generation_error": reason,
    }, max_retries + 1


def generate_icl(story_section, target_answer, difficulty, max_retries=2):
    """Generate question using ICL QG method."""
    question = ""
    reason = "unknown"
    gen = None
    last_prompt = ""
    last_raw = ""

    for attempt in range(max_retries + 1):
        prompt = build_icl_prompt(story_section, target_answer, difficulty)
        temp = 0.1 + min(attempt * 0.1, 0.3)
        raw = _call_llm(prompt, temperature=temp)
        last_prompt = prompt
        last_raw = raw

        if _is_degenerate(raw):
            reason = "degenerate output"
            continue

        gen = _parse_json(raw)
        question = gen.get("question", "") if gen else ""
        ok, reason = _validate_question(question, target_answer)
        if ok:
            return {
                "generated_question": question,
                "method": "ICL",
                "generation_prompt": prompt,
                "generation_raw": raw,
                "parse_ok": True,
            }, 1 + attempt

    return {
        "generated_question": question,
        "method": "ICL",
        "generation_prompt": last_prompt,
        "generation_raw": last_raw,
        "parse_ok": gen is not None,
        "generation_error": reason,
    }, max_retries + 1


def generate_self_refine(story_section, target_answer, difficulty, max_retries=2):
    """Generate question using SelfRefine method (generate + critique + revise)."""
    # Step 1: Generate
    gen_prompt = build_self_refine_gen_prompt(story_section, target_answer, difficulty)
    gen_raw = _call_llm(gen_prompt)

    if _is_degenerate(gen_raw):
        return {
            "generated_question": "",
            "method": "SelfRefine",
            "generation_prompt": gen_prompt,
            "generation_raw": gen_raw,
            "parse_ok": False,
            "generation_error": "degenerate initial output",
        }, 1

    gen = _parse_json(gen_raw)
    question = gen.get("question", "") if gen else ""

    if not question:
        return {
            "generated_question": "",
            "method": "SelfRefine",
            "generation_prompt": gen_prompt,
            "generation_raw": gen_raw,
            "parse_ok": False,
            "generation_error": "initial generation failed",
        }, 1

    # Step 2: Critique
    critique_prompt = build_self_refine_critique_prompt(
        question, story_section, target_answer, difficulty
    )
    critique_raw = _call_llm(critique_prompt, max_tokens=200)
    critique = _parse_json(critique_raw)

    needs_revision = True
    if critique and isinstance(critique, dict):
        quality = critique.get("overall_quality", "needs_revision")
        if quality == "good":
            needs_revision = False

    # Step 3: Revise (if needed)
    final_question = question
    final_raw = gen_raw
    final_prompt = gen_prompt

    if needs_revision:
        # Extract issues text for the revise prompt
        if critique and isinstance(critique, dict):
            issues = critique.get("issues", [])
            critique_text = "; ".join(str(i) for i in issues) if issues else "needs improvement"
        else:
            critique_text = "needs improvement"

        for attempt in range(max_retries):
            revise_prompt = build_self_refine_revise_prompt(
                question, critique_text, story_section, target_answer, difficulty
            )
            revise_raw = _call_llm(revise_prompt)

            if _is_degenerate(revise_raw):
                continue

            revise_gen = _parse_json(revise_raw)
            revised_q = revise_gen.get("question", "") if revise_gen else ""
            ok, reason = _validate_question(revised_q, target_answer)
            if ok:
                final_question = revised_q
                final_raw = revise_raw
                final_prompt = revise_prompt
                break

    ok, reason = _validate_question(final_question, target_answer)

    return {
        "generated_question": final_question,
        "method": "SelfRefine",
        "generation_prompt": final_prompt,
        "generation_raw": final_raw,
        "parse_ok": gen is not None,
        "generation_error": None if ok else reason,
    }, 3


def _self_check_ours(question, story_section, target_answer, focus_key=None,
                     difficulty="Hard", graph_policy=None):
    """Lightweight self-check for Ours questions.

    Checks:
    1. Natural answer matches target_answer
    2. Sentence count requirement (difficulty-aware)
    3. Question focus matches answer role (if focus_key provided)
    4. Graph policy compliance (if graph_policy provided)

    Returns (ok, reason, focus_match, graph_policy_compliance).
    """
    ctx = _format_story_context(story_section)

    focus_check = ""
    if focus_key and focus_key in ANSWER_FOCUS_TEMPLATES:
        focus_label = ANSWER_FOCUS_TEMPLATES[focus_key]["label"]
        focus_check = f"""
3. Does the QUESTION FOCUS match "{focus_label}"? For example, if the answer is about a larger purpose, the question should ask about the larger purpose, not an immediate/local cause."""

    # Difficulty-aware sentence count check
    if difficulty == "Easy":
        sentence_check = "2. Is the question answerable from one sentence (the answer sentence alone)?"
    elif difficulty == "Medium":
        sentence_check = "2. Does answering require reading at least 2 sentences (not just 1)?"
    else:  # Hard
        sentence_check = "2. Does answering require reading 3 or more sentences (not just 1-2)?"

    # Graph policy compliance check
    policy_check = ""
    if graph_policy:
        if graph_policy == "answer_only":
            policy_check = """
4. Is this question answerable from the answer sentence ALONE (no bridge or multi-sentence reasoning needed)?"""
        elif graph_policy in ("two_node_relation", "two_node_relation_fallback"):
            policy_check = """
4. Does the question reference exactly one supporting relationship (not a full chain)?"""
        elif graph_policy == "multi_node_chain":
            policy_check = """
4. Does the question require a multi-step reasoning chain (not answerable from one sentence)?"""

    prompt = f"""Quick check on this reading-comprehension question.

Story:
{ctx}

Question: "{question}"
Expected answer: "{target_answer}"

Check these things:
1. Does answering this question naturally lead to "{target_answer}" (or a close paraphrase)?
{sentence_check}{focus_check}{policy_check}

Return ONLY: {{"answer_match": "yes|no", "meets_sentence_req": "yes|no", "focus_match": "yes|no", "graph_policy_compliance": "yes|no", "reason": "brief"}}"""

    raw = _call_judge(prompt, temperature=0.0, max_tokens=200)
    parsed = _parse_json(raw)

    if not parsed or not isinstance(parsed, dict):
        # On parse failure, don't block — pass through
        return True, "self-check parse failure, passing through", "unknown", "unknown"

    ans_match = parsed.get("answer_match", "yes") == "yes"
    # Backward compatibility: check meets_sentence_req first, fall back to needs_3_plus
    meets_req = parsed.get("meets_sentence_req", parsed.get("needs_3_plus", "yes")) == "yes"
    # For Easy, always pass sentence requirement
    if difficulty == "Easy":
        meets_req = True
    focus_ok = parsed.get("focus_match", "yes") == "yes"
    policy_ok = parsed.get("graph_policy_compliance", "yes") == "yes"

    if ans_match and meets_req and focus_ok and policy_ok:
        return True, "ok", "yes", "yes"

    reasons = []
    if not ans_match:
        reasons.append("answer mismatch")
    if not meets_req:
        if difficulty == "Medium":
            reasons.append("needs only 1 sentence")
        else:
            reasons.append("needs only 1-2 sentences")
    if not focus_ok:
        reasons.append("focus mismatch")
    if not policy_ok:
        reasons.append(f"graph_policy non-compliant ({graph_policy})")
    return False, "; ".join(reasons), "yes" if focus_ok else "no", "yes" if policy_ok else "no"


def generate_ours(story_section, target_answer, difficulty, nodes, edges,
                  required_evidence_sentences, bridge_sentence_ids,
                  reasoning_operation, necessity_type, max_retries=3):
    """Generate question using Narrative Evidence Graph guided method (Ours).

    Robust retry: first attempts use full prompt, later attempts use compact
    repair prompt for degenerate/parse failures.
    """
    # Select graph substructure once (difficulty-controlled)
    graph_sub = select_graph_substructure(nodes, edges, difficulty, target_answer)

    # Classify answer focus once
    focus_key, answer_role, answer_node_type = classify_answer_focus(nodes, target_answer)

    # Common fields for all return dicts
    gp_fields = {
        "graph_policy": graph_sub["graph_policy"],
        "graph_policy_reason": graph_sub["graph_policy_reason"],
        "selected_node_ids": [n["id"] for n in graph_sub["selected_nodes"]],
        "selected_edge_relations": [e.get("relation", "") for e in graph_sub["selected_edges"]],
        "evidence_roles_used": graph_sub["evidence_roles_used"],
        "relation_chain": graph_sub["relation_chain"],
    }

    question = ""
    reason = "unknown"
    gen = None
    last_prompt = ""
    last_raw = ""
    best_focus_match = "unknown"
    best_policy_compliance = "unknown"
    attempts_trace = []

    for attempt in range(max_retries + 1):
        # After 2 full-prompt failures, switch to compact repair prompt
        use_repair = (attempt >= 2 and reason in (
            "degenerate output", "empty", "parse failure"))
        if use_repair:
            prompt = build_repair_prompt(
                story_section, target_answer, difficulty, focus_key,
                required_evidence_sentences, graph_substructure=graph_sub,
            )
            temp = 0.1
            prompt_type = "repair"
        else:
            prompt = build_ours_prompt(
                story_section, target_answer, difficulty, nodes, edges,
                required_evidence_sentences, bridge_sentence_ids,
                reasoning_operation, necessity_type, graph_substructure=graph_sub,
            )
            temp = 0.05 if attempt == 0 else 0.1
            prompt_type = "full"

        raw = _call_llm(prompt, temperature=temp)
        last_prompt = prompt
        last_raw = raw

        # Trace every attempt
        trace_entry = {
            "attempt": attempt,
            "prompt_type": prompt_type,
            "temperature": temp,
            "raw_prefix": (raw[:200] if raw else ""),
            "degenerate": _is_degenerate(raw),
            "parse_ok": False,
            "question": "",
            "validate_reason": "",
            "self_check_reason": "",
        }

        if _is_degenerate(raw):
            reason = "degenerate output"
            trace_entry["validate_reason"] = reason
            attempts_trace.append(trace_entry)
            continue

        gen = _parse_json(raw)
        trace_entry["parse_ok"] = gen is not None
        if not gen:
            reason = "parse failure"
            trace_entry["validate_reason"] = reason
            attempts_trace.append(trace_entry)
            continue

        question = gen.get("question", "") if gen else ""
        trace_entry["question"] = question[:100]

        # Hard output length guard
        if question and not _question_length_ok(question):
            reason = "question length out of range"
            trace_entry["validate_reason"] = reason
            attempts_trace.append(trace_entry)
            continue

        ok, reason = _validate_question(question, target_answer)
        trace_entry["validate_reason"] = reason
        if ok:
            # Self-check: verify answer match, sentence count, focus, and graph policy
            sc_ok, sc_reason, focus_match, policy_compliance = _self_check_ours(
                question, story_section, target_answer, focus_key,
                difficulty=difficulty, graph_policy=graph_sub["graph_policy"],
            )
            best_focus_match = focus_match
            best_policy_compliance = policy_compliance
            trace_entry["self_check_reason"] = sc_reason
            attempts_trace.append(trace_entry)
            if sc_ok:
                return {
                    "generated_question": question,
                    "method": "Ours",
                    "generation_prompt": prompt,
                    "generation_raw": raw,
                    "parse_ok": True,
                    "self_check_pass": True,
                    "answer_role": answer_role,
                    "answer_node_type": answer_node_type,
                    "question_focus": focus_key,
                    "focus_match": focus_match,
                    "graph_policy_compliance": policy_compliance,
                    "attempts_trace": attempts_trace,
                    "repair_attempted": any(t["prompt_type"] == "repair" for t in attempts_trace),
                    "repair_success": any(t["prompt_type"] == "repair" and t == attempts_trace[-1] for t in attempts_trace),
                    **gp_fields,
                }, 1 + attempt
            else:
                reason = f"self-check failed: {sc_reason}"
                continue
        else:
            attempts_trace.append(trace_entry)
            continue

    return {
        "generated_question": question,
        "method": "Ours",
        "generation_prompt": last_prompt,
        "generation_raw": last_raw,
        "parse_ok": gen is not None,
        "generation_error": reason,
        "answer_role": answer_role,
        "answer_node_type": answer_node_type,
        "question_focus": focus_key,
        "focus_match": best_focus_match,
        "graph_policy_compliance": best_policy_compliance,
        "attempts_trace": attempts_trace,
        "repair_attempted": any(t["prompt_type"] == "repair" for t in attempts_trace),
        "repair_success": False,
        **gp_fields,
    }, max_retries + 1


# ── Judges ─────────────────────────────────────────────────────

def quality_judge(question, story_section, target_answer, difficulty):
    """Judge 1: Quality assessment.

    Returns dict with: answerable, asks_expected_answer,
    final_answer_consistent, answer_leakage, fluent, reason.
    """
    if not question:
        return {
            "answerable": "no", "asks_expected_answer": "no",
            "final_answer_consistent": "no", "answer_leakage": "no",
            "fluency": "no", "reason": "empty question",
            "quality_pass": False,
        }

    ctx = _format_story_context(story_section)

    prompt = f"""You are a question quality judge. Evaluate the generated question.

Story:
{ctx}

Generated question: "{question}"
Target answer: "{target_answer}"

Rate each criterion:
1. answerable: Can this question be answered from the story? (yes/partial/no)
2. asks_expected_answer: If someone answers this question naturally, would the answer match "{target_answer}" in meaning? Accept paraphrases and partial matches. (yes/partial/no)
3. final_answer_consistent: Is the answer "{target_answer}" consistent with what the story says? (yes/partial/no)
4. answer_leakage: Does the question contain the exact phrase "{target_answer}"? (yes/no)
5. fluent: Is the question grammatically correct and natural-sounding? (yes/no)

Return ONLY a JSON object:
{{"answerable": "yes|partial|no", "asks_expected_answer": "yes|partial|no", "final_answer_consistent": "yes|partial|no", "answer_leakage": "yes|no", "fluency": "yes|no", "reason": "brief explanation"}}"""

    raw = _call_judge(prompt, temperature=0.0, max_tokens=300)
    parsed = _parse_json(raw)

    if not parsed or not isinstance(parsed, dict):
        return {
            "answerable": "no", "asks_expected_answer": "no",
            "final_answer_consistent": "no", "answer_leakage": "no",
            "fluency": "no", "reason": f"judge parse error: {raw[:100]}",
            "quality_pass": False, "quality_judge_raw": raw,
        }

    # Determine quality pass (loose: allows partial)
    qp = (
        parsed.get("answerable") in ("yes", "partial")
        and parsed.get("asks_expected_answer") in ("yes", "partial")
        and parsed.get("final_answer_consistent") in ("yes", "partial")
        and parsed.get("answer_leakage") == "no"
        and parsed.get("fluency") == "yes"
    )

    # Strict quality pass: requires yes on all (no partial)
    sqp = (
        parsed.get("answerable") == "yes"
        and parsed.get("asks_expected_answer") == "yes"
        and parsed.get("final_answer_consistent") == "yes"
        and parsed.get("answer_leakage") == "no"
        and parsed.get("fluency") == "yes"
    )

    parsed["quality_pass"] = qp
    parsed["strict_quality_pass"] = sqp
    parsed["quality_judge_raw"] = raw
    return parsed


def difficulty_evidence_judge(question, story_section, target_answer, difficulty,
                              required_evidence_sentences=None, bridge_sentence_ids=None):
    """Judge 2: Difficulty and evidence dependency assessment.

    Returns dict with: predicted_difficulty, answer_sentence_alone_sufficient,
    bridge_required, required_evidence_sentences_used, bridge_removal_effect, reason.
    """
    if not question:
        return {
            "predicted_difficulty": "Easy", "answer_sentence_alone_sufficient": "yes",
            "bridge_required": "no", "required_evidence_sentences_used": [],
            "bridge_removal_effect": "none", "reason": "empty question",
        }

    ctx = _format_story_context(story_section)

    prompt = f"""You are a reading-comprehension difficulty judge. Evaluate the question below.

Story:
{ctx}

Question: "{question}"
Expected answer: "{target_answer}"

TASK: Determine the MINIMUM number of sentences a reader must read to correctly answer this question.

Step 1: Find the sentence that directly contains or states the answer. This is the "answer sentence."
Step 2: Check if the answer sentence ALONE is enough to correctly answer the question.
- If YES: the question needs 1 sentence.
- If NO: identify what other sentence(s) are needed. Count them.
Step 3: For each additional sentence needed, explain WHY it is needed (motivation, context, cause, disambiguation, etc.)

Step 4: Assign difficulty based on total sentences needed:
- 1 sentence = Easy
- 2 sentences = Medium
- 3 or more sentences = Hard

IMPORTANT: Be precise about counting. If the question asks about motivation or cause, and the motivation is in a different sentence from the action/answer, count BOTH sentences. If there is a chain (A causes B causes C), count all links in the chain.

Return a JSON object:
{{
  "num_sentences_needed": <integer>,
  "answer_sentence_id": <int or null>,
  "other_sentence_ids": [<int>, ...],
  "answer_sentence_alone_sufficient": "yes|partial|no",
  "bridge_required": "yes|no",
  "bridge_removal_effect": "none|harder|ambiguous|unanswerable",
  "predicted_difficulty": "Easy|Medium|Hard",
  "reason": "brief explanation"
}}"""

    raw = _call_judge(prompt, temperature=0.0, max_tokens=800)
    parsed = _parse_json(raw)

    # Short retry on parse failure
    if not parsed or not isinstance(parsed, dict):
        short_prompt = f"""Count the minimum sentences needed to answer this question in the story.

Story:
{ctx}

Question: "{question}"
Answer: "{target_answer}"

Return ONLY a JSON object:
{{"num_sentences_needed": <int>, "predicted_difficulty": "Easy|Medium|Hard", "answer_sentence_alone_sufficient": "yes|no", "bridge_required": "yes|no", "bridge_removal_effect": "none|harder|ambiguous|unanswerable", "reason": "brief"}}"""
        raw2 = _call_judge(short_prompt, temperature=0.0, max_tokens=400)
        parsed = _parse_json(raw2)
        if parsed and isinstance(parsed, dict):
            raw = raw2

    if not parsed or not isinstance(parsed, dict):
        return {
            "predicted_difficulty": "judge_error",
            "answer_sentence_alone_sufficient": "judge_error",
            "bridge_required": "judge_error",
            "required_evidence_sentences_used": [],
            "bridge_removal_effect": "judge_error",
            "reason": f"parse error after retry: {raw[:200]}",
            "difficulty_judge_raw": raw,
            "difficulty_judge_prompt": prompt,
            "difficulty_judge_status": "parse_error",
            "difficulty_judge_parse_ok": False,
        }

    # Derive difficulty from num_sentences_needed if available
    num_needed = parsed.get("num_sentences_needed")
    if isinstance(num_needed, (int, float)):
        num_needed = int(num_needed)
        if num_needed >= 3:
            parsed["predicted_difficulty"] = "Hard"
        elif num_needed == 2:
            parsed["predicted_difficulty"] = "Medium"
        else:
            parsed["predicted_difficulty"] = "Easy"

    # Normalize
    if parsed.get("predicted_difficulty") not in ("Easy", "Medium", "Hard"):
        parsed["predicted_difficulty"] = "judge_error"
    if parsed.get("answer_sentence_alone_sufficient") not in ("yes", "partial", "no"):
        parsed["answer_sentence_alone_sufficient"] = "judge_error"
    if parsed.get("bridge_required") not in ("yes", "partial", "no"):
        parsed["bridge_required"] = "judge_error"
    if parsed.get("bridge_removal_effect") not in ("none", "harder", "ambiguous", "unanswerable"):
        parsed["bridge_removal_effect"] = "judge_error"

    # Build required_evidence_sentences_used from answer + other
    if "required_evidence_sentences_used" not in parsed:
        answer_sid = parsed.get("answer_sentence_id")
        other_sids = parsed.get("other_sentence_ids", [])
        all_sids = []
        if isinstance(answer_sid, (int, float)):
            all_sids.append(int(answer_sid))
        if isinstance(other_sids, list):
            all_sids.extend(int(s) for s in other_sids if isinstance(s, (int, float)))
        parsed["required_evidence_sentences_used"] = sorted(set(all_sids))
    elif not isinstance(parsed.get("required_evidence_sentences_used"), list):
        parsed["required_evidence_sentences_used"] = []

    parsed["difficulty_judge_raw"] = raw
    parsed["difficulty_judge_status"] = "ok"
    parsed["difficulty_judge_parse_ok"] = True
    parsed["difficulty_judge_prompt"] = prompt
    return parsed


# ── Evidence coverage diagnostic ───────────────────────────────

def compute_evidence_coverage(difficulty_judge_result, target_required_sentence_ids, target_bridge_sentence_ids):
    """Compare judge-identified evidence sentences vs target candidate's evidence.

    Returns dict with coverage metrics.
    """
    judge_status = difficulty_judge_result.get("difficulty_judge_status", "ok")
    judge_used = difficulty_judge_result.get("required_evidence_sentences_used", [])
    if not isinstance(judge_used, list):
        judge_used = []

    target_req = target_required_sentence_ids or []
    target_bridge = target_bridge_sentence_ids or []

    target_set = set(int(s) for s in target_req if isinstance(s, (int, float)))
    bridge_set = set(int(s) for s in target_bridge if isinstance(s, (int, float)))
    judge_set = set(int(s) for s in judge_used if isinstance(s, (int, float)))

    # Target evidence coverage: overlap / len(target_required)
    if target_set:
        overlap = target_set & judge_set
        coverage = len(overlap) / len(target_set)
    else:
        overlap = set()
        coverage = 0.0

    # Uses all target required sentences
    uses_all = target_set.issubset(judge_set) if target_set else False

    # Uses bridge sentences
    if bridge_set:
        bridge_overlap = bridge_set & judge_set
        if bridge_overlap == bridge_set:
            uses_bridge = "yes"
        elif bridge_overlap:
            uses_bridge = "partial"
        else:
            uses_bridge = "no"
    else:
        uses_bridge = "no"

    num_judge_used = len(judge_set)
    predicted_diff = difficulty_judge_result.get("predicted_difficulty", "judge_error")

    # Hard realization pass (exact-id diagnostic, legacy)
    hrp_threshold = 2.0 / 3.0 - 1e-9
    hrp = (
        judge_status == "ok"
        and num_judge_used >= 3
        and uses_bridge in ("yes", "partial")
        and coverage >= hrp_threshold
        and predicted_diff == "Hard"
    )

    return {
        "target_required_sentence_ids": sorted(target_set),
        "judge_used_sentence_ids": sorted(judge_set),
        "target_evidence_coverage": round(coverage, 3),
        "uses_all_target_required_sentences": "yes" if uses_all else "no",
        "uses_bridge_sentences": uses_bridge,
        "num_judge_used_sentences": num_judge_used,
        "hard_realization_pass": "yes" if hrp else "no",
    }


def semantic_evidence_match_judge(question, story_section, target_answer,
                                  target_required_sentence_ids, judge_used_sentence_ids):
    """Judge whether judge-used sentences support the same reasoning chain as target evidence.

    Returns dict with semantic_evidence_match and semantic_match_reason.
    """
    if not question or not judge_used_sentence_ids:
        return {
            "semantic_evidence_match": "no",
            "semantic_match_reason": "no question or no judge-used sentences",
        }

    ctx = _format_story_context(story_section)
    target_sids = sorted(int(s) for s in (target_required_sentence_ids or [])
                         if isinstance(s, (int, float)))
    judge_sids = sorted(int(s) for s in (judge_used_sentence_ids or [])
                        if isinstance(s, (int, float)))

    prompt = f"""You are evaluating whether two sets of evidence sentences support the same reasoning chain.

Story:
{ctx}

Generated question: "{question}"
Target answer: "{target_answer}"

Target evidence sentences (from the original annotation): {target_sids}
Judge-identified evidence sentences (from difficulty judge): {judge_sids}

TASK: Do the judge-identified sentences support the same reasoning chain as the target evidence sentences?
- "yes": The judge-used sentences cover the same causal/motivational/temporal chain, even if specific sentence IDs differ slightly.
- "partial": The judge-used sentences cover part of the chain but miss a key link or include irrelevant sentences.
- "no": The judge-used sentences support a completely different reasoning path or are off-topic.

Return ONLY: {{"semantic_evidence_match": "yes|partial|no", "semantic_match_reason": "brief explanation"}}"""

    raw = _call_judge(prompt, temperature=0.0, max_tokens=200)
    parsed = _parse_json(raw)

    if not parsed or not isinstance(parsed, dict):
        return {
            "semantic_evidence_match": "judge_error",
            "semantic_match_reason": f"parse error: {raw[:100]}",
        }

    match_val = parsed.get("semantic_evidence_match", "no")
    if match_val not in ("yes", "partial", "no"):
        match_val = "judge_error"

    return {
        "semantic_evidence_match": match_val,
        "semantic_match_reason": parsed.get("semantic_match_reason", ""),
    }
