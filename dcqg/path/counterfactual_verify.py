"""Stage B: Counterfactual evidence verification.

For each evidence sentence identified in Stage A, remove it from the story and
ask an LLM whether the question can still be answered.  Three independent calls
with varied system prompts produce a majority vote, validating that each
evidence sentence is truly *necessary*, not merely *relevant*.
"""
import json
import logging

from dcqg.utils.api_client import call_openai_compatible
from dcqg.path.fairytale_evidence_audit import _split_sentences, classify_difficulty

logger = logging.getLogger(__name__)

_SYSTEM_PROMPTS = [
    "You are an evidence necessity analyst.",
    "You are a reading comprehension expert evaluating evidence.",
    "You are a careful analyst assessing question answerability.",
]


def build_counterfactual_prompt(sentences, question, answer, removed_idx):
    """Build prompt with sentence[removed_idx] replaced by '[REMOVED]'."""
    story_lines = []
    for i, s in enumerate(sentences):
        if i == removed_idx:
            story_lines.append(f"  [S{i}] [REMOVED]")
        else:
            story_lines.append(f"  [S{i}] {s}")
    story_block = "\n".join(story_lines)

    return f"""Below is a story with one sentence removed, followed by a question and its correct answer.

Story (sentence S{removed_idx} has been removed):
{story_block}

Question: {question}
Answer: {answer}

Given the story with sentence [S{removed_idx}] removed, can you still determine the answer to the question?

Think carefully:
- If the remaining sentences still contain enough information to arrive at the correct answer, respond "yes".
- If removing this sentence makes the answer impossible or ambiguous to determine, respond "no".

Return ONLY a JSON object:
{{"can_still_answer": "yes" or "no", "reasoning": "one sentence explanation"}}"""


def _parse_vote(raw):
    """Parse LLM response into a vote string."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError, TypeError, AttributeError):
            return None

    val = data.get("can_still_answer", "").strip().lower()
    if val == "yes":
        return "not_necessary"
    if val == "no":
        return "necessary"
    return None


def verify_single_sentence(sentences, question, answer, removed_idx,
                           api_url, api_key, model, n_runs=3):
    """Run counterfactual check for one sentence.

    Returns dict with sentence_id, votes, majority, and confidence.
    """
    prompt = build_counterfactual_prompt(sentences, question, answer, removed_idx)
    votes = []

    for run_i in range(n_runs):
        system = _SYSTEM_PROMPTS[run_i % len(_SYSTEM_PROMPTS)]
        try:
            raw = call_openai_compatible(
                prompt,
                api_url=api_url,
                api_key=api_key,
                model=model,
                temperature=0.0,
                max_tokens=200,
                # Some OpenAI-compatible providers/models reject response_format
                # even when they can follow a JSON-only prompt.
                json_mode=False,
                system=system,
                timeout=90,
            )
            vote = _parse_vote(raw)
            if vote:
                votes.append(vote)
            else:
                logger.warning("Unparseable response for S%d run %d: %s",
                               removed_idx, run_i, raw)
        except Exception as e:
            logger.warning("API error for S%d run %d: %s", removed_idx, run_i, e)

    if not votes:
        # All runs failed — conservatively keep as necessary
        return {
            "sentence_id": removed_idx,
            "votes": [],
            "majority": "unverified",
            "confidence": 0.0,
        }

    necessary_count = votes.count("necessary")
    not_necessary_count = votes.count("not_necessary")

    if necessary_count >= not_necessary_count:
        majority = "necessary"
        agreement = necessary_count
    else:
        majority = "not_necessary"
        agreement = not_necessary_count

    return {
        "sentence_id": removed_idx,
        "votes": votes,
        "majority": majority,
        "confidence": agreement / len(votes),
    }


def verify_candidate(candidate, api_url, api_key, model, n_runs=3):
    """Run counterfactual verification on all required evidence sentences.

    Args:
        candidate: dict from FairytaleEvidenceAuditor.audit_batch() with fields
            story_section, question, answer1, required_evidence_sentences,
            bridge_sentence_ids, evidence_difficulty, etc.

    Returns dict with verified_evidence_sentences, dropped_evidence_sentences,
    verification_details, verified_num_required, verified_difficulty.
    """
    sentences = _split_sentences(candidate.get("story_section", ""))
    question = candidate.get("question", "")
    answer = candidate.get("answer1", "")
    if candidate.get("answer2"):
        answer += f" / {candidate['answer2']}"

    required_ids = candidate.get("required_evidence_sentences", [])

    if not sentences or not question or not required_ids:
        return {
            "verified_evidence_sentences": required_ids,
            "dropped_evidence_sentences": [],
            "verification_details": [],
            "verified_num_required": len(required_ids),
            "verified_answer_directly_found": candidate.get(
                "answer_directly_found", "no"
            ),
            "verified_difficulty": candidate.get("evidence_difficulty", "Easy"),
        }

    details = []
    verified = []
    dropped = []

    for sid in required_ids:
        if sid < 0 or sid >= len(sentences):
            dropped.append(sid)
            continue

        result = verify_single_sentence(
            sentences, question, answer, sid,
            api_url, api_key, model, n_runs=n_runs,
        )
        details.append(result)

        if result["majority"] in ("necessary", "unverified"):
            verified.append(sid)
        else:
            dropped.append(sid)

    # Reclassify difficulty with the verified (possibly smaller) evidence set
    verified_bridge = [s for s in candidate.get("bridge_sentence_ids", [])
                       if s in verified]

    # Build a modified assessment dict for classify_difficulty
    assessment = {
        "answer_directly_found": candidate.get("answer_directly_found", "no"),
        "answer_sentence_alone_sufficient": (
            "yes" if candidate.get("answer_directly_found") == "yes" and len(verified) == 1 else
            candidate.get("answer_sentence_alone_sufficient", "partial")
        ),
        "num_required_sentences": len(verified),
        "required_evidence_sentences": verified,
        "bridge_sentence_ids": verified_bridge,
        "bridge_removal_effect": candidate.get("bridge_removal_effect", "none"),
        "necessity_type": candidate.get("necessity_type", "background_context"),
    }

    # If we dropped sentences, bridge_removal_effect may weaken
    if dropped and len(verified) < 3:
        assessment["bridge_removal_effect"] = "harder"

    verified_difficulty = classify_difficulty(assessment)

    return {
        "verified_evidence_sentences": verified,
        "dropped_evidence_sentences": dropped,
        "verification_details": details,
        "verified_num_required": len(verified),
        "verified_answer_directly_found": assessment["answer_directly_found"],
        "verified_difficulty": verified_difficulty,
    }
