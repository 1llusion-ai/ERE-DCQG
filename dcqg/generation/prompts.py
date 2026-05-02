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
