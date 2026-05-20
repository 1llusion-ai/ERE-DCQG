"""Answer-Grounded Evidence Planner.

Given story + target answer + target difficulty (NO original question),
identifies the evidence sentences needed to ask a difficulty-appropriate
question whose answer is the target answer.
"""

import json
import re
import time

from dcqg.utils.api_client import call_openai_compatible
from dcqg.utils.config import get_api_config
from dcqg.generation.parser import parse_json_response
from dcqg.difficulty.definitions import (
    difficulty_definition,
    evidence_definitions_block,
)


# ── Sentence splitting ────────────────────────────────────────────

def _split_sentences(text):
    """Dialogue-aware sentence splitter. Delegates to dcqg.utils.text.split_sentences."""
    from dcqg.utils.text import split_sentences
    return split_sentences(text)


# ── Prompt ────────────────────────────────────────────────────────

def build_answer_grounded_evidence_prompt(story_name, story_section, target_answer,
                                           target_difficulty, local_or_sum="",
                                           attribute="", ex_or_im=""):
    """Build the evidence planning prompt.

    IMPORTANT: The original FairytaleQA question is NOT included.
    The model plans evidence based on story + answer + difficulty only.
    """
    sentences = _split_sentences(story_section)
    story_text = "\n".join(f"[S{i}] {s}" for i, s in enumerate(sentences))
    n_sentences = len(sentences)

    # Difficulty-specific operational guidance. The canonical difficulty
    # definition is injected separately; this block gives planning instructions.
    diff_guidance = {
        "Easy": (
            "Plan for an EASY question:\n"
            "- Identify exactly 1 sentence that directly states or closely paraphrases the answer.\n"
            "- Set answer_directly_found=yes and answer_sentence_alone_sufficient=yes.\n"
            "- No bridge sentences; set bridge_required=no.\n"
            "- If the answer must be inferred or requires multiple sentences, set target_difficulty_feasible=partial or no."
        ),
        "Medium": (
            "Plan for a MEDIUM question:\n"
            "- Follow either Case 1 (inferred answer, 1 necessary sentence, simple inference) or Case 2 (direct answer, multiple necessary sentences, simple synthesis).\n"
            "- Do NOT plan complex implicit or multi-step reasoning.\n"
            "- Set bridge_required based on whether a support sentence connects context to the answer.\n"
            "- If the answer is direct and one-sentence only, set target_difficulty_feasible=partial.\n"
            "- If the answer is inferred from multiple sentences, set target_difficulty_feasible=partial or no."
        ),
        "Hard": (
            "Plan for a HARD question:\n"
            "- Identify at least 2 required evidence sentences, preferably 3+ when the story supports it.\n"
            "- The plan must require at least one inference across multiple evidence sentences.\n"
            "- Suitable forms include aggregation, comparison, temporal/causal chaining, separated-event resolution, or tracking multiple entities.\n"
            "- Set answer_sentence_alone_sufficient=no or partial.\n"
            "- If the answer is directly found, or only one sentence is needed, set target_difficulty_feasible=no or partial.\n"
            "- Prefer causal_chain, motivation_chain, summary_synthesis, repeated_event_count, or disambiguation as reasoning_operation."
        ),
    }.get(target_difficulty, "")

    # Metadata line (for report only, not the core task)
    meta_parts = [f"Story: {story_name}"]
    if local_or_sum:
        meta_parts.append(f"answer-scope: {local_or_sum}")
    if attribute:
        meta_parts.append(f"attribute: {attribute}")
    if ex_or_im:
        meta_parts.append(f"style: {ex_or_im}")

    prompt = f"""You are an evidence planner for narrative question generation.

Given a story section, a target answer, and a target difficulty level,
identify the sentences a reader must read to answer a question at that
difficulty whose answer is the target answer.

{' | '.join(meta_parts)}

## Story Section
{story_text}

## Target Answer
"{target_answer}"

## Target Difficulty: {target_difficulty}

Evidence Definition:
{evidence_definitions_block()}

Difficulty Definition:
{difficulty_definition(target_difficulty)}

{diff_guidance}

## Output Fields

Return a JSON object with these fields:

1. **answer_sentence_id** (int): The sentence index [S?] that most directly contains
   or implies the target answer. This sentence MUST be in required_evidence_sentences.

2. **answer_directly_found** (str): "yes" or "no".
   - "yes": the target answer, or a close paraphrase, is directly found in the text
   - "no": the target answer must be inferred

3. **required_evidence_sentences** (list[int]): All sentence indices a reader MUST
   read to correctly answer the question. Sorted ascending. Minimum length 1.

4. **anchor_sentence_ids** (list[int]): Sentences that establish context, setting,
   characters, or situation. The "starting point" sentences. Can overlap with
   required_evidence_sentences but NOT with answer_sentence_id.

5. **bridge_sentence_ids** (list[int]): Sentences that connect anchor context to
   the answer. Must NOT include answer_sentence_id. Empty for Easy.

6. **answer_sentence_alone_sufficient** (str): "yes", "partial", or "no".
   - "yes": reading ONLY the answer sentence is enough
   - "partial": the answer sentence gives hints but one other sentence helps
   - "no": the answer sentence alone is ambiguous

7. **bridge_required** (str): "yes" or "no".
   - "yes": reader must cross at least one bridge sentence to connect context to answer
   - "no": no bridging needed

8. **reasoning_operation** (str): The type of reasoning needed.
   Choose from: explicit_lookup, local_inference, causal_chain, motivation_chain,
   summary_synthesis, disambiguation

9. **necessity_type** (str): Why are the extra sentences needed?
   Choose from: answer_local, one_relation, causal_bridge, motivation_bridge,
   summary_synthesis, disambiguation, weak_or_invalid

10. **evidence_plan_valid** (str): "yes" or "no".
   - "yes": the plan is coherent and matches the target difficulty
   - "no": the evidence does not support the requested difficulty

11. **evidence_plan_reason** (str): One sentence explaining the plan.

12. **target_difficulty_feasible** (str): "yes", "partial", or "no".
    - "yes": the story section has good evidence for this difficulty
    - "partial": evidence exists but is suboptimal for this difficulty
    - "no": the story section cannot support a question at this difficulty

## Important Rules

- All sentence indices must be valid: 0 to {n_sentences - 1}.
- answer_sentence_id MUST be in required_evidence_sentences.
- bridge_sentence_ids must NOT overlap answer_sentence_id.
- anchor_sentence_ids can overlap required_evidence_sentences but NOT answer_sentence_id.
- The directness and number of required_evidence_sentences must match the difficulty target.
- Do NOT fabricate sentences — only use indices that exist in the story above.
- Return ONLY the JSON object, no other text."""

    return prompt


# ── API Call ──────────────────────────────────────────────────────

def _call_evidence_planner(prompt, model=None, temperature=0.0, max_tokens=1200, timeout=120):
    """Call the LLM for evidence planning. Uses json_mode=True."""
    cfg = get_api_config()
    if model is None:
        model = cfg.get("JUDGE_MODEL", "Qwen/Qwen2.5-32B-Instruct")

    try:
        resp = call_openai_compatible(
            prompt,
            api_url=cfg["SILICONFLOW_API_URL"],
            api_key=cfg["SILICONFLOW_API_KEY"],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            system="You are a precise JSON-only evidence planner. Return only valid JSON.",
            timeout=timeout,
        )
        return resp if resp else "ERROR: empty response"
    except Exception as e:
        return f"ERROR: {e}"


# ── Parse ─────────────────────────────────────────────────────────

def _parse_evidence_plan(raw):
    """Parse JSON from LLM response. Returns dict or None."""
    if not raw or raw.startswith("ERROR"):
        return None
    result = parse_json_response(raw)
    if isinstance(result, dict):
        return result
    return None


# ── Validation ────────────────────────────────────────────────────

def validate_evidence_plan(plan, n_sentences, target_difficulty):
    """Validate and correct an evidence plan.

    Returns (plan, contradictions: list[str]).
    """
    contradictions = []

    if not isinstance(plan, dict):
        return plan, ["plan is not a dict"]

    # Required fields
    required_fields = [
        "answer_sentence_id", "required_evidence_sentences",
        "anchor_sentence_ids", "bridge_sentence_ids",
        "answer_directly_found", "answer_sentence_alone_sufficient", "bridge_required",
        "reasoning_operation", "necessity_type",
        "evidence_plan_valid", "evidence_plan_reason",
        "target_difficulty_feasible",
    ]
    for f in required_fields:
        if f not in plan:
            plan[f] = None
            contradictions.append(f"missing field: {f}")

    # Get answer_sentence_id
    ans_id = plan.get("answer_sentence_id")
    if ans_id is None or not isinstance(ans_id, (int, float)):
        plan["evidence_plan_valid"] = "no"
        contradictions.append("answer_sentence_id invalid or missing")
        return plan, contradictions
    ans_id = int(ans_id)

    # Validate answer_sentence_id
    if ans_id < 0 or ans_id >= n_sentences:
        plan["evidence_plan_valid"] = "no"
        contradictions.append(f"answer_sentence_id={ans_id} out of range [0, {n_sentences - 1}]")
        return plan, contradictions

    # Get required_evidence_sentences
    req = plan.get("required_evidence_sentences", [])
    if not isinstance(req, list) or len(req) == 0:
        plan["evidence_plan_valid"] = "no"
        contradictions.append("required_evidence_sentences empty or not a list")
        return plan, contradictions
    req = [int(x) for x in req]

    # Validate all req indices
    invalid_req = [x for x in req if x < 0 or x >= n_sentences]
    if invalid_req:
        plan["evidence_plan_valid"] = "no"
        contradictions.append(f"invalid required_evidence_sentences: {invalid_req}")
        return plan, contradictions

    # answer_sentence_id must be in required_evidence_sentences
    if ans_id not in req:
        req.append(ans_id)
        req = sorted(set(req))
        contradictions.append(f"answer_sentence_id={ans_id} not in required, added")

    # Validate bridge
    bridge = plan.get("bridge_sentence_ids", [])
    if isinstance(bridge, list):
        bridge = [int(x) for x in bridge]
        invalid_bridge = [x for x in bridge if x < 0 or x >= n_sentences]
        if invalid_bridge:
            contradictions.append(f"invalid bridge ids removed: {invalid_bridge}")
            bridge = [x for x in bridge if 0 <= x < n_sentences]
        if ans_id in bridge:
            contradictions.append(f"answer_sentence_id={ans_id} in bridge, removed")
            bridge = [x for x in bridge if x != ans_id]
        plan["bridge_sentence_ids"] = bridge
    else:
        plan["bridge_sentence_ids"] = []
        contradictions.append("bridge_sentence_ids not a list, reset to []")

    # Validate anchor
    anchor = plan.get("anchor_sentence_ids", [])
    if isinstance(anchor, list):
        anchor = [int(x) for x in anchor]
        invalid_anchor = [x for x in anchor if x < 0 or x >= n_sentences]
        if invalid_anchor:
            contradictions.append(f"invalid anchor ids removed: {invalid_anchor}")
            anchor = [x for x in anchor if 0 <= x < n_sentences]
        if ans_id in anchor:
            contradictions.append(f"answer_sentence_id={ans_id} in anchor, removed")
            anchor = [x for x in anchor if x != ans_id]
        plan["anchor_sentence_ids"] = anchor
    else:
        plan["anchor_sentence_ids"] = []
        contradictions.append("anchor_sentence_ids not a list, reset to []")

    # Recompute
    plan["required_evidence_sentences"] = sorted(set(req))
    plan["num_required_sentences"] = len(plan["required_evidence_sentences"])
    plan["evidence_span"] = [
        min(plan["required_evidence_sentences"]),
        max(plan["required_evidence_sentences"]),
    ]

    # Difficulty consistency checks
    num_req = plan["num_required_sentences"]
    direct = plan.get("answer_directly_found", "")
    asa = plan.get("answer_sentence_alone_sufficient", "")
    br = plan.get("bridge_required", "")
    feasible = plan.get("target_difficulty_feasible", "")
    if direct not in ("yes", "no"):
        direct = "yes" if asa == "yes" else "no"
        plan["answer_directly_found"] = direct
        contradictions.append("answer_directly_found missing or invalid, inferred from ASA")

    if target_difficulty == "Easy":
        if num_req != 1:
            contradictions.append(f"Easy with num_req={num_req} (expected 1)")
        if direct != "yes":
            contradictions.append(f"Easy with answer_directly_found={direct} (expected yes)")
        if asa != "yes" and asa not in ("?", None):
            contradictions.append(f"Easy with ASA={asa} (expected yes)")

    elif target_difficulty == "Medium":
        medium_ok = (
            (direct == "no" and num_req == 1)
            or (direct == "yes" and num_req >= 2)
        )
        if not medium_ok:
            contradictions.append(
                f"Medium with answer_directly_found={direct}, num_req={num_req}"
            )

    elif target_difficulty == "Hard":
        if direct != "no":
            contradictions.append(f"Hard with answer_directly_found={direct} (expected no)")
        if num_req < 2:
            contradictions.append(f"Hard with num_req={num_req} < 2")
        if br != "yes":
            # summary_synthesis and count aggregation may not need a bridge
            ro = plan.get("reasoning_operation", "")
            nt = plan.get("necessity_type", "")
            if ro in ("summary_synthesis", "repeated_event_count") or "summary" in nt.lower() or "count" in nt.lower():
                # Accept bridge_required=no for multi-evidence aggregation
                pass
            else:
                contradictions.append(f"Hard with bridge_required={br} (expected yes)")
        if asa == "yes":
            # asa=yes is still a contradiction for Hard
            ro = plan.get("reasoning_operation", "")
            if ro not in ("summary_synthesis", "repeated_event_count"):
                contradictions.append(f"Hard with ASA=yes (expected no/partial)")

    return plan, contradictions


# ── Main planner ──────────────────────────────────────────────────

def plan_evidence(story_name, story_section, target_answer, target_difficulty,
                   model=None, local_or_sum="", attribute="", ex_or_im=""):
    """Run the answer-grounded evidence planner.

    Args:
        story_name: str
        story_section: str — the story text
        target_answer: str — the answer to plan for
        target_difficulty: "Easy" | "Medium" | "Hard"
        model: str or None — model override
        local_or_sum, attribute, ex_or_im: metadata (report only, NOT in core prompt)

    Returns:
        dict with plan fields + trace fields.
    """
    sentences = _split_sentences(story_section)
    n_sentences = len(sentences)

    if n_sentences == 0:
        return {
            "evidence_plan_valid": "no",
            "evidence_plan_reason": "empty story section",
            "answer_grounded_evidence_parse_ok": False,
            "answer_grounded_evidence_model": model or get_api_config().get("JUDGE_MODEL", "?"),
            "original_question_present_in_prompt": False,
            "n_story_sentences": 0,
            "contradictions": ["empty story section"],
        }

    prompt = build_answer_grounded_evidence_prompt(
        story_name, story_section, target_answer, target_difficulty,
        local_or_sum=local_or_sum, attribute=attribute, ex_or_im=ex_or_im,
    )

    raw = _call_evidence_planner(prompt, model=model)

    # Parse
    plan = _parse_evidence_plan(raw)
    parse_ok = plan is not None

    if not parse_ok:
        plan = {
            "evidence_plan_valid": "no",
            "evidence_plan_reason": f"parse error: {raw[:200] if raw else 'None'}",
        }

    # Validate
    plan, contradictions = validate_evidence_plan(plan, n_sentences, target_difficulty)

    # Add trace fields
    plan["answer_grounded_evidence_prompt"] = prompt
    plan["answer_grounded_evidence_raw"] = raw
    plan["answer_grounded_evidence_parse_ok"] = parse_ok
    plan["answer_grounded_evidence_model"] = model or get_api_config().get("JUDGE_MODEL", "?")
    plan["original_question_present_in_prompt"] = False
    plan["n_story_sentences"] = n_sentences
    plan["contradictions"] = contradictions

    return plan
