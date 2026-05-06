"""Difficulty-aware prompts for PathQG-HardAware generation.

Contains:
- FEW_SHOT_* examples for Easy/Medium/Hard
- DIFFICULTY_DEFINITIONS_HA
- prompt_pathqg_easy/medium/hard: structured prompts
- fmt_ctx: format supporting sentences
"""
from dcqg.utils.text import simple_stem


# ── FEW-SHOT EXAMPLES (difficulty-specific) ──────────────────

FEW_SHOT_EASY = """Example 1 (Easy - 1-hop question):
Events: "attack" -> "destroy"
Context: [S0] The army launched an attack on the city. [S1] The city was destroyed.
Question: "What happened to the city after the army attacked it?"
- This is Easy: asks about one event directly following another.
- Uses a simple "after X, what happened to Y?" pattern."""

FEW_SHOT_MEDIUM = """Example 2 (Medium - must reference start event + one intermediate event):
Events: "resign" -> "appoint" -> "implement"
Context: [S0] The CEO resigned. [S1] The board appointed a new leader. [S2] The new CEO implemented reforms.
Question: "What did the new CEO implement after the board appointed a replacement following the CEO's resignation?"
- This is Medium: the question mentions both "resignation" and "appointed", requiring the solver to connect two steps.
- The answer event ("implement") is the final event, not mentioned in the question."""

FEW_SHOT_HARD = """Example 3 (Hard - must bind 2+ prior events in the question):
Events: "announce" -> "protest" -> "cancel"
Context: [S0] The government announced budget cuts. [S1] Citizens protested the decision. [S2] Officials canceled the policy.
Question: "After the government announced budget cuts and citizens protested, what did officials do regarding the policy?"
- This is Hard: the question EXPLICITLY names two prior events ("announced budget cuts" AND "citizens protested").
- The solver must understand the causal chain: announcement -> protest -> cancellation.
- The answer cannot be found by reading only the last sentence - you need to understand the full chain.
- Note: the gold answer "cancel" is NEVER mentioned in the question."""

FEW_SHOT_HARD_IMPLICIT = """Example (Hard - implicit chain):

Path: "announced" -> "protested" -> "canceled" -> "resigned"
Context:
[S0] The government announced sweeping budget cuts to the education sector.
[S1] Thousands of citizens protested the decision in front of parliament.
[S2] Officials canceled the planned cuts after facing mounting pressure.
[S3] The finance minister resigned amid the political fallout.

GOOD question: "What long-term political consequence resulted from the public backlash against the government's austerity measures?"
Why Hard: The solver must trace: austerity measures (announced) -> public backlash (protested) -> cancellation -> resignation.
The question does NOT name the answer. The solver must follow the chain.

BAD question: "What happened to the finance minister after the budget cuts were announced?"
Why Easy: The solver just reads [S3] to find "resigned". No chain reasoning needed.
"""


# ── DIFFICULTY DEFINITIONS (CrossQG-style) ───────────────────

DIFFICULTY_DEFINITIONS_HA = {
    "Easy": "Easy questions are straightforward, answerable from a single sentence in the context.",
    "Medium": "Medium questions require connecting information from 2-3 sentences.",
    "Hard": "Hard questions require synthesizing information across multiple sentences and reasoning chains.",
}


# ── BANNED SHORTCUT PHRASES (Hard only) ──────────────────────
# Re-exported from dcqg.path.direction for convenience
BANNED_PATTERNS_HARD = [
    r"(?i)what\s+was\s+the\s+final\s+outcome",
    r"(?i)what\s+action\s+was\s+taken",
    r"(?i)what\s+happened\s+after\s+the\s+incident",
    r"(?i)what\s+happened\s+after\s+the\s+event",
    r"(?i)what\s+happened\s+as\s+a\s+result",
    r"(?i)what\s+was\s+the\s+result",
    r"(?i)what\s+did\s+\w+\s+do\s+after\s+the\s+(incident|event|crash|accident|disaster)",
    r"(?i)following\s+the\s+(incident|event|occurrence)",
]


# ── CONTEXT FORMATTER ────────────────────────────────────────

def fmt_ctx(supporting):
    """Format supporting_sentences as context string."""
    return "\n".join(s if isinstance(s, str) else f"[S{s[0]}] {s[1]}" for s in supporting)


# ── DIFFICULTY-AWARE PROMPTS ─────────────────────────────────

def prompt_pathqg_easy(item):
    """Easy: 1-hop, simple question. CrossQG-style structured prompt."""
    events = item["events"]
    path_str = " -> ".join(f'"{e["trigger"]}"' for e in events)
    final = events[-1]["trigger"]
    ctx = fmt_ctx(item.get("supporting_sentences", []))
    rel_types = item.get("relation_subtypes", [])
    rel_str = ", ".join(rel_types) if rel_types else "N/A"
    answer_phrase = item.get("gold_answer_phrase", final)
    answer_sentence = item.get("gold_answer_sentence", "")
    event_type = item.get("gold_event_type", events[-1].get("type", ""))

    return f"""{DIFFICULTY_DEFINITIONS_HA["Easy"]}

{FEW_SHOT_EASY}

Context:
{ctx}

Target final event:
  Trigger: "{final}"
  Answer meaning: "{answer_phrase}"
  Event type: {event_type}
  Sentence: "{answer_sentence}"

Event Path:
{path_str}

Relation Sequence:
{rel_str}

Requirements for Easy:
- Ask about what happened after a single event.
- A 1-hop question is acceptable.
- Reference at least 1 path event.
- The question's natural answer should identify the final event.
- Do NOT copy the exact answer phrase into the question.
- Question must start with a question word and end with "?".

Output Format:
{{"question": "...", "answer": "...", "reasoning_type": "direct"}}"""


def prompt_pathqg_medium(item):
    """Medium: 1 prior event, connect 2+ sentences. CrossQG-style structured prompt."""
    events = item["events"]
    path_str = " -> ".join(f'"{e["trigger"]}"' for e in events)
    final = events[-1]["trigger"]
    prior_events = [e["trigger"] for e in events[:-1]]
    prior_list = ", ".join(f'"{t}"' for t in prior_events)
    ctx = fmt_ctx(item.get("supporting_sentences", []))
    rel_types = item.get("relation_subtypes", [])
    rel_str = ", ".join(rel_types) if rel_types else "N/A"
    answer_phrase = item.get("gold_answer_phrase", final)
    answer_sentence = item.get("gold_answer_sentence", "")
    event_type = item.get("gold_event_type", events[-1].get("type", ""))

    return f"""{DIFFICULTY_DEFINITIONS_HA["Medium"]}

{FEW_SHOT_MEDIUM}

Context:
{ctx}

Target final event:
  Trigger: "{final}"
  Answer meaning: "{answer_phrase}"
  Event type: {event_type}
  Sentence: "{answer_sentence}"

Event Path:
{path_str}

Relation Sequence:
{rel_str}

Requirements for Medium:
- Ask about the final event in a way that requires understanding how it connects to earlier events.
- Reference at least 1 PRIOR event from: {prior_list} (not the final event).
- The question should connect information from at least 2 context sentences.
- The solver should need to understand the relationship between events.
- The question's natural answer should identify the final event.
- Do NOT copy the exact answer phrase into the question.
- Question must start with a question word and end with "?".

Output Format:
{{"question": "...", "answer": "...", "reasoning_type": "chain"}}"""


def prompt_pathqg_hard(item):
    """Hard: 2+ prior events, cross-sentence. CrossQG-style structured prompt."""
    events = item["events"]
    path_str = " -> ".join(f'"{e["trigger"]}"' for e in events)
    final = events[-1]["trigger"]
    prior_events = [e["trigger"] for e in events[:-1]]
    prior_list = ", ".join(f'"{t}"' for t in prior_events)
    ctx = fmt_ctx(item.get("supporting_sentences", []))
    rel_types = item.get("relation_subtypes", [])
    rel_str = ", ".join(rel_types) if rel_types else "N/A"
    answer_phrase = item.get("gold_answer_phrase", final)
    answer_sentence = item.get("gold_answer_sentence", "")
    event_type = item.get("gold_event_type", events[-1].get("type", ""))

    return f"""{DIFFICULTY_DEFINITIONS_HA["Hard"]}

{FEW_SHOT_HARD}

Context:
{ctx}

Target final event:
  Trigger: "{final}"
  Answer meaning: "{answer_phrase}"
  Event type: {event_type}
  Sentence: "{answer_sentence}"

Event Path:
{path_str}

Relation Sequence:
{rel_str}

CRITICAL Requirements for Hard (you MUST follow ALL):
1. EXPLICITLY mention at least TWO prior events from: {prior_list}.
   - Use SPECIFIC event names, not vague references like "the incident".
   - GOOD: "After X announced cuts and Y protested, what did officials do?"
   - BAD: "What was the final outcome after the incident?"

2. FORBIDDEN phrases (do NOT use ANY):
   - "final outcome" / "final result"
   - "what happened after the incident/event/crash"
   - "what action was taken"
   - "as a result" / "what was the result"

3. The question must NOT be answerable from a single sentence.
   - The solver must connect at least two pieces of information.

4. The question's natural answer should identify the final event.
   - Do NOT copy the exact answer phrase into the question.

5. Question must start with a question word and end with "?".

Output Format:
{{"question": "...", "answer": "...", "reasoning_type": "cross_sentence"}}"""


def prompt_pathqg_hard_implicit(item):
    """Hard: implicit chain, 3+ event reasoning without listing the chain explicitly."""
    events = item["events"]
    path_str = " -> ".join(f'"{e["trigger"]}"' for e in events)
    final = events[-1]["trigger"]
    prior_events = [e["trigger"] for e in events[:-1]]
    prior_list = ", ".join(f'"{t}"' for t in prior_events)
    ctx = fmt_ctx(item.get("supporting_sentences", []))
    rel_types = item.get("relation_subtypes", [])
    rel_str = ", ".join(rel_types) if rel_types else "N/A"
    answer_phrase = item.get("gold_answer_phrase", final)
    answer_sentence = item.get("gold_answer_sentence", "")
    event_type = item.get("gold_event_type", events[-1].get("type", ""))
    event_ids = [e.get("id", "") for e in events]

    return f"""Hard questions require 3+ event reasoning. The solver must discover intermediate events from context.

{FEW_SHOT_HARD_IMPLICIT}

Context:
{ctx}

Target final event:
  Trigger: "{final}"
  Answer meaning: "{answer_phrase}"
  Event type: {event_type}
  Sentence: "{answer_sentence}"

Event Path (for reference only — do NOT list all triggers in the question):
{path_str}

Event IDs (for hidden_path_events output): {", ".join(event_ids)}

Relation Sequence:
{rel_str}

Requirements for Hard:
1. The question must require tracing 3+ events to answer (not answerable from 1 sentence).
2. Use one of these patterns (adapt to your events):
   a) "What [consequence/outcome/restriction] resulted from [description of chain start]?"
   b) "What [result type] concluded the sequence triggered by [start event description]?"
   c) "What [impact/change] did [entity] face after [description of situation]?"
3. Do NOT use "What happened when X?" — that is answerable from 1 sentence (too easy).
4. At most 2 prior trigger words. Use descriptions: {prior_list}
5. Include a specific anchor (participant, entity, location) from the context.
6. Do NOT copy the answer phrase. Start with question word, end with "?".

Output Format:
{{"question": "...", "answer": "...", "reasoning_type": "implicit_chain", "hidden_path_events": ["event_id", ...], "expected_steps": "3+"}}"""


# ── ANCHOR HELPER ──────────────────────────────────────────

import re

_STOPWORDS = {"The", "This", "That", "What", "When", "Where", "After", "Before",
              "Which", "Their", "They", "There", "However", "Meanwhile", "Also",
              "But", "And", "Yet", "Still", "Then", "From", "With", "About"}


def _extract_anchors(events, supporting_sentences):
    """Extract entity/location/date anchors from supporting sentences."""
    anchors = []
    for s in supporting_sentences[:4]:
        text = s[1] if isinstance(s, (list, tuple)) else str(s)
        caps = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
        for c in caps:
            if len(c) > 3 and c not in _STOPWORDS:
                anchors.append(c)
    seen = set()
    unique = []
    for a in anchors:
        if a not in seen:
            seen.add(a)
            unique.append(a)
    return ", ".join(unique[:5]) if unique else "[entity/location from context]"


# ── MULTI-STRATEGY HARD PROMPTS ────────────────────────────

def _answer_type_guidance(event_type, answer_phrase):
    """Return question framing guidance based on the final event type."""
    et = event_type.lower() if event_type else ""
    ap = answer_phrase.lower() if answer_phrase else ""

    # Sign agreement / Treaty / Resolution
    if any(kw in et for kw in ("sign_agreement", "agreement", "treaty", "resolution")):
        return (
            'Ask: "Which agreement/date/action formally ended or resolved ...?"\n'
            '  or: "What formal resolution resulted from ...?"\n'
            f'  The expected answer is a specific agreement/date/action: "{answer_phrase}"'
        )
    # Preventing / Letting / Permission / Restriction
    if any(kw in et for kw in ("prevent", "letting", "permission", "restriction", "prohibit", "forbid", "allow")):
        return (
            'Ask: "What restriction/permission was ultimately imposed ...?"\n'
            '  or: "What specific constraint resulted from ...?"\n'
            f'  The expected answer is a specific restriction/permission: "{answer_phrase}"'
        )
    # Death / Injury / Damage
    if any(kw in et for kw in ("death", "injury", "damage", "destroy", "kill", "harm")):
        return (
            'Ask: "What final harm/outcome resulted from ...?"\n'
            '  or: "What specific consequence did [entity] suffer after ...?"\n'
            f'  The expected answer is a specific harm/outcome: "{answer_phrase}"'
        )
    # Arrest / Conviction / Sentence
    if any(kw in et for kw in ("arrest", "convict", "sentence", "charge", "trial")):
        return (
            'Ask: "What legal action/outcome ultimately followed ...?"\n'
            f'  The expected answer is a specific legal outcome: "{answer_phrase}"'
        )
    # Transfer / Acquisition / Ownership
    if any(kw in et for kw in ("transfer", "acqui", "ownership", "purchase", "buy", "sell")):
        return (
            'Ask: "What transfer/change of ownership resulted from ...?"\n'
            f'  The expected answer is a specific transfer: "{answer_phrase}"'
        )
    # Default: use generic but answer-aligned framing
    return (
        f'Ask a question whose natural answer is: "{answer_phrase}"\n'
        '  Use "what" framing (not "why"). Ask for the specific result/outcome/action.\n'
        '  GOOD: "What [specific result] resulted from ...?"\n'
        '  BAD: "Why did X lead to Y?" (unless the answer IS an explanation)'
    )


def prompt_hidden_endpoint(item):
    """Hard: mention only the start event, hide intermediate and final. Solver must trace 3+ steps."""
    events = item["events"]
    path_str = " -> ".join(f'"{e["trigger"]}"' for e in events)
    final = events[-1]["trigger"]
    ctx = fmt_ctx(item.get("supporting_sentences", []))
    rel_types = item.get("relation_subtypes", [])
    rel_str = ", ".join(rel_types) if rel_types else "N/A"
    answer_phrase = item.get("gold_answer_phrase", final)
    event_type = item.get("gold_event_type", events[-1].get("type", ""))
    event_ids = [e.get("id", "") for e in events]
    start = events[0]["trigger"]
    middle_triggers = [e["trigger"] for e in events[1:-1]]
    anchors = _extract_anchors(events, item.get("supporting_sentences", []))
    ans_guidance = _answer_type_guidance(event_type, answer_phrase)

    return f"""Generate a HARD question requiring 3+ reasoning steps. The solver must discover intermediate and final events from context.

Event path (for reference):
{path_str}

Context:
{ctx}

Target final event: "{final}"
Expected answer phrase: "{answer_phrase}"
Event type: {event_type}
Event IDs: {", ".join(event_ids)}
Relations: {rel_str}

=== CRITICAL: ANSWER TARGET ===
Your question MUST be answerable with: "{answer_phrase}"
This is NOT negotiable. If the question's natural answer would be something else, REWRITE.

{ans_guidance}

=== RULES ===

1. You MAY mention the starting event "{start}" (or a close paraphrase).
2. You MUST NOT mention ANY of these events: {", ".join(f'"{t}"' for t in middle_triggers + [final])}
   These are INTERMEDIATE and FINAL events that the solver must discover.
3. The question must require reading 3+ context sentences to answer.
4. Include an entity/location from: {anchors}
5. Do NOT copy the answer phrase into the question. Start with question word, end with "?".
6. Use "what" questions. Do NOT use "why" questions.
7. DO NOT ask about intermediate events (outcry, forces rushing, siege, etc.).
   Ask ONLY about the FINAL RESULT: "{answer_phrase}".

=== VERIFICATION ===
Check your question:
- Does it mention "{final}"? -> REWRITE
- Does it mention any intermediate event ({", ".join(f'"{t}"' for t in middle_triggers)})? -> REWRITE
- Can it be answered from 1 sentence? -> REWRITE (must require chain reasoning)
- Would the expected answer be "{answer_phrase}"? If NO -> REWRITE
- Does it ask about an intermediate event instead of the final result? -> REWRITE

GOOD: "What [specific result type] followed [entity]'s {start}?"  (answer = "{answer_phrase}")
BAD: "What outcry followed the destruction?"  (asks about intermediate, not "{answer_phrase}")
BAD: "What forces rushed to reinforce?"  (asks about intermediate event)

Output: {{"question": "...", "answer": "...", "reasoning_type": "hidden_endpoint", "hidden_path_events": ["event_id", ...], "expected_steps": "3+"}}"""


def prompt_relation_composition(item):
    """Hard: ask about the composed relation, mention only start event."""
    events = item["events"]
    path_str = " -> ".join(f'"{e["trigger"]}"' for e in events)
    final = events[-1]["trigger"]
    ctx = fmt_ctx(item.get("supporting_sentences", []))
    rel_types = item.get("relation_subtypes", [])
    rel_str = ", ".join(rel_types) if rel_types else "N/A"
    answer_phrase = item.get("gold_answer_phrase", final)
    event_type = item.get("gold_event_type", events[-1].get("type", ""))
    event_ids = [e.get("id", "") for e in events]
    start = events[0]["trigger"]
    middle_triggers = [e["trigger"] for e in events[1:-1]]
    anchors = _extract_anchors(events, item.get("supporting_sentences", []))
    ans_guidance = _answer_type_guidance(event_type, answer_phrase)

    return f"""Generate a HARD question about the composed effect of a multi-step event chain.

Event path (reference): {path_str}
Context:
{ctx}

Target final event: "{final}"
Expected answer phrase: "{answer_phrase}"
Event type: {event_type}
Event IDs: {", ".join(event_ids)}
Relations: {rel_str}

=== CRITICAL: ANSWER TARGET ===
Your question MUST be answerable with: "{answer_phrase}"
This is NOT negotiable. If the question's natural answer would be something else, REWRITE.

{ans_guidance}

=== RULES ===
1. You MAY mention the starting event "{start}".
2. Do NOT mention intermediate events ({", ".join(f'"{t}"' for t in middle_triggers)}) or the final event "{final}".
3. Ask about the FINAL result/action/outcome — NOT about intermediate events.
4. The solver must trace {len(events)} events across {len(events)} context sentences.
5. Include entity from: {anchors}
6. Do NOT copy answer phrase. Start with question word, end with "?".
7. BANNED: "Why did..." questions. Use "What restriction/outcome/action/result..." instead.
8. DO NOT ask about intermediate events (outcry, siege, forces, etc.).
   Ask ONLY about the FINAL RESULT: "{answer_phrase}".

=== VERIFICATION ===
Check your question:
- Does it start with "Why"? -> REWRITE (use "What")
- Would the expected answer be "{answer_phrase}"? If NO -> REWRITE
- Does it mention "{final}" or intermediate events? -> REWRITE
- Does it ask about an intermediate event? -> REWRITE

GOOD: "What [restriction/outcome/action] resulted from [entity]'s {start}?"  (answer = "{answer_phrase}")
BAD: "What outcry followed the destruction?"  (asks about intermediate, not "{answer_phrase}")
BAD: "Why did {start} lead to changes?"  (asks for reason, not the answer)

Output: {{"question": "...", "answer": "...", "reasoning_type": "relation_composition", "hidden_path_events": ["event_id", ...], "expected_steps": "3+"}}"""


def prompt_contrastive(item):
    """Hard: mention only start event, solver must trace chain to disambiguate."""
    events = item["events"]
    path_str = " -> ".join(f'"{e["trigger"]}"' for e in events)
    final = events[-1]["trigger"]
    ctx = fmt_ctx(item.get("supporting_sentences", []))
    rel_types = item.get("relation_subtypes", [])
    rel_str = ", ".join(rel_types) if rel_types else "N/A"
    answer_phrase = item.get("gold_answer_phrase", final)
    event_ids = [e.get("id", "") for e in events]
    start = events[0]["trigger"]
    middle_triggers = [e["trigger"] for e in events[1:-1]]
    anchors = _extract_anchors(events, item.get("supporting_sentences", []))

    return f"""Generate a HARD question where the solver must trace a multi-step chain to find the answer.

Event path (reference): {path_str}
Context:
{ctx}

Target final event: "{final}" (answer: "{answer_phrase}")
Event IDs: {", ".join(event_ids)}
Relations: {rel_str}

=== RULES ===
1. You MAY mention the starting event "{start}".
2. Do NOT mention intermediate ({", ".join(f'"{t}"' for t in middle_triggers)}) or final "{final}" events.
3. Describe a situation where the solver must follow the chain to determine the outcome.
4. The question must require reading 3+ context sentences.
5. Include entity from: {anchors}
6. Do NOT copy answer phrase. Start with question word, end with "?".

GOOD: "What ultimately happened to [entity] after {start}?"
BAD: "After {start} and {middle_triggers[0] if middle_triggers else 'X'}, what happened?"

Output: {{"question": "...", "answer": "...", "reasoning_type": "contrastive", "hidden_path_events": ["event_id", ...], "expected_steps": "3+"}}"""


def prompt_missing_bridge(item):
    """Hard: mention only start, solver must discover bridge events to reach the end."""
    events = item["events"]
    path_str = " -> ".join(f'"{e["trigger"]}"' for e in events)
    final = events[-1]["trigger"]
    ctx = fmt_ctx(item.get("supporting_sentences", []))
    rel_types = item.get("relation_subtypes", [])
    rel_str = ", ".join(rel_types) if rel_types else "N/A"
    answer_phrase = item.get("gold_answer_phrase", final)
    event_ids = [e.get("id", "") for e in events]
    start = events[0]["trigger"]
    middle_triggers = [e["trigger"] for e in events[1:-1]]
    anchors = _extract_anchors(events, item.get("supporting_sentences", []))

    return f"""Generate a HARD question where the solver must discover what happened between the start and the end of a chain.

Event path (reference): {path_str}
Context:
{ctx}

Target final event: "{final}" (answer: "{answer_phrase}")
Event IDs: {", ".join(event_ids)}
Relations: {rel_str}

=== RULES ===
1. You MAY mention the starting event "{start}".
2. Do NOT mention intermediate ({", ".join(f'"{t}"' for t in middle_triggers)}) or final "{final}" events.
3. Ask what the ultimate consequence/outcome was after {start}.
4. The solver must discover {len(events)-1} events by reading the context.
5. Include entity from: {anchors}
6. Do NOT copy answer phrase. Start with question word, end with "?".

GOOD: "What was the final outcome for [entity] after {start}?"
BAD: "Between {start} and {final}, what happened?" (names both endpoints)

Output: {{"question": "...", "answer": "...", "reasoning_type": "missing_bridge", "hidden_path_events": ["event_id", ...], "expected_steps": "3+"}}"""
