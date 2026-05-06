"""Repair prompts for hard-aware question generation.

- build_repair_prompt: construct a repair prompt for a failed question
- build_alignment_repair_prompt: repair prompt for alignment failures
- REPAIRABLE_REASONS: set of failure reasons that can be fixed via repair
"""
from dcqg.generation.prompts import fmt_ctx


REPAIRABLE_REASONS = {
    "no question mark", "bad start", "word repetition", "trigger leakage",
    "empty", "parse error", "too short", "excessive repetition", "looping trigram",
    "not a dict", "no common English words",
    "banned phrase", "only 0 prior events mentioned, need >=2",
    "only 1 prior events mentioned, need >=2",
    "path_binding", "path_coverage", "too_explicit",
    "double_question", "alignment_drift", "drift",
}


def build_repair_prompt(item, failed_question, failure_reason, difficulty, covered_indices=None):
    """Build repair prompt for hard-aware generation.
    If covered_indices is provided, list specific uncovered events.
    """
    events = item["events"]
    path_str = " -> ".join(f'"{e["trigger"]}"' for e in events)
    final = events[-1]["trigger"]
    prior_events = [e["trigger"] for e in events[:-1]]
    prior_list = ", ".join(f'"{t}"' for t in prior_events)
    ctx = fmt_ctx(item.get("supporting_sentences", []))

    fix_hints = {
        "no question mark": "End the question with a question mark (?)",
        "bad start": "Start with What/Who/When/Where/Why/How/Did/Was/Were/Is/Are",
        "word repetition": "Remove repeated words, write grammatically correct English",
        "trigger leakage": f'Do NOT use the word "{final}" or its synonyms anywhere in the question',
        "empty": "Write a complete question",
        "parse error": "Output ONLY valid JSON with 'question' and 'reasoning_type' keys",
        "too short": "Write a longer, more complete question that includes event details",
        "excessive repetition": "Remove repeated words, write naturally",
        "looping trigram": "Write naturally, avoid repeating the same phrase patterns",
    }

    if "banned phrase" in failure_reason:
        fix_hints["banned phrase"] = "Avoid template phrases like 'final outcome' or 'what happened after the incident'. Instead, use SPECIFIC event names from the path."
    if "only" in failure_reason and "prior events mentioned" in failure_reason:
        fix_hints["insufficient events"] = f"Mention at least TWO specific events from: {prior_list}. Name them explicitly in the question."
    if "too_explicit" in failure_reason:
        fix_hints["too_explicit"] = f"""Your question lists too many prior event triggers explicitly.
Do NOT name 2+ events from: {prior_list}.
Instead, use an IMPLICIT chain approach:
- Name at most 1 prior event trigger (preferably the start event).
- DESCRIBE other events using different words (not their trigger words).
  Example: "public outcry" instead of "protested", "the inquiry" instead of "investigated".
- The question must still reference 2+ prior events by meaning — just not by trigger words.
- The solver must discover intermediate events by reading the context.
Example: "What decisive action concluded the escalation that began with the initial announcement?"
(describes "protested" as "escalation", "canceled" as "decisive action", uses only "announcement" as trigger)
"""
    if "path_binding" in failure_reason:
        min_req = {"Easy": 1, "Medium": 1, "Hard": 2}.get(difficulty, 1)
        check_events = events[:-1] if difficulty in ("Medium", "Hard") else events
        uncovered = [e for i, e in enumerate(check_events) if not covered_indices or i not in covered_indices]
        if uncovered:
            event_list = "\n".join(f'  - "{e["trigger"]}" ({e.get("type", "event")})' for e in uncovered[:3])
            fix_hints["path_binding"] = f"""Your question did not mention enough prior events (covered {len(covered_indices or [])}/{min_req} required).
Please rewrite to explicitly reference these events:
{event_list}
Do NOT mention the target answer: "{final}"
Use the specific event trigger words or clear descriptions."""
        else:
            fix_hints["path_binding"] = f"Your question must explicitly mention at least {min_req} events from the path. Use the specific event trigger words."

    # Path coverage failure (from quality filter, not generation-time check)
    if "path_coverage" in failure_reason:
        # Extract missing events from the failure reason if available
        missing_events = [e["trigger"] for e in events[:-1]]
        missing_list = ", ".join(f'"{t}"' for t in missing_events)
        fix_hints["path_coverage"] = f"""Your question only covers too few prior events.
It must explicitly reference prior events from: {missing_list}
Rewrite the question so these events are clearly mentioned BEFORE asking about the final event.
Do NOT mention the target answer: "{final}"
Example: "After {missing_events[0] if missing_events else 'X'} and {missing_events[1] if len(missing_events) > 1 else 'Y'}, what happened?" """

    fix = fix_hints.get(failure_reason, f"Fix this issue: {failure_reason}")

    hard_extra = ""
    if difficulty == "Hard":
        hard_extra = f"""
HARD-SPECIFIC (implicit chain):
- At most 1 prior event trigger from: {prior_list} may appear in the question.
- BUT you must DESCRIBE at least 2 prior events using different words (not trigger words).
  Example: "public outcry" for "protested", "the inquiry" for "investigated".
- The question must include at least one anchor (participant, entity, or consequence type).
- Question must require connecting 3+ sentences to answer."""

    return f"""Your previous output was rejected.
Rejected: "{failed_question}"
Issue: {failure_reason}
-> {fix}

Generate a corrected {difficulty} question:
Events: {path_str}
Context: {ctx}
- Answer is "{final}", do NOT mention it
{hard_extra}
- Question must start with a question word and end with ?
- Output ONLY: {{"question": "...", "reasoning_type": "..."}}"""


def build_alignment_repair_prompt(item, failed_question, alignment_reason, difficulty="Hard"):
    """Build repair prompt specifically for alignment failures.

    When the question asks about intermediate causes/actions instead of the final answer,
    this prompt instructs the LLM to rewrite so the natural answer matches the expected phrase.
    """
    events = item["events"]
    path_str = " -> ".join(f'"{e["trigger"]}"' for e in events)
    final = events[-1]["trigger"]
    prior_events = [e["trigger"] for e in events[:-1]]
    prior_list = ", ".join(f'"{t}"' for t in prior_events)
    ctx = fmt_ctx(item.get("supporting_sentences", []))
    answer_phrase = item.get("gold_answer_phrase", final)
    start = events[0]["trigger"]
    middle_triggers = [e["trigger"] for e in events[1:-1]]

    return f"""Your question was rejected because it drifted away from the expected answer.

Rejected question: "{failed_question}"
Problem: {alignment_reason}

Your question asks about intermediate causes/actions. Rewrite it so the natural answer is exactly:
"{answer_phrase}"

Keep the question hard by using prior events as constraints, but the requested answer must be the final outcome/restriction/agreement/action.

Context:
{ctx}

Event path (reference): {path_str}
Expected answer: "{answer_phrase}"

=== RULES ===
1. You MAY mention the starting event "{start}" or describe it in other words.
2. Do NOT mention intermediate events ({", ".join(f'"{t}"' for t in middle_triggers)}) or the final event "{final}".
3. Ask about the FINAL RESULT/OUTCOME — NOT about intermediate causes or reactions.
4. The question must require reading 3+ context sentences to answer.
5. Do NOT use double questions (no "How did X, and what Y?").
6. Use a SINGLE "What" question focused on the final answer.
7. Do NOT copy the answer phrase into the question.

GOOD: "What [specific result] resulted from [entity]'s {start}?"  (answer = "{answer_phrase}")
BAD: "How did {start} influence [intermediate event]?"  (asks about intermediate, not final answer)
BAD: "What outcry followed the destruction?"  (asks about intermediate event)

Output: {{"question": "...", "answer": "{answer_phrase}", "reasoning_type": "alignment_repair", "hidden_path_events": ["event_id", ...], "expected_steps": "3+"}}"""
