"""FairytaleQA Evidence Audit.

For each QA pair, assess evidence difficulty using the story section as context.
Determines whether the answer can be found from a single sentence or requires
multi-sentence evidence chains.

This is the narrative-QA counterpart to dcqg.path.evidence_necessity
(which audits MAVEN-ERE event-based evidence).

Phase 1: audit only — no question generation.
"""
import json
import re

from dcqg.utils.api_client import call_openai_compatible
from dcqg.utils.config import get_api_config


def _split_sentences(text):
    """Split text into sentences. Simple heuristic for narrative text."""
    if not text:
        return []
    # Split on sentence-ending punctuation followed by space or end
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter empty
    return [s.strip() for s in parts if s.strip()]


def _build_evidence_prompt(qa_records):
    """Build LLM prompt for evidence difficulty assessment of FairytaleQA pairs.

    Each record has: story_name, story_section, question, answer1, and metadata.
    """
    parts = []
    for i, rec in enumerate(qa_records):
        sentences = _split_sentences(rec["story_section"])
        sent_lines = "\n".join(f"  [S{j}] {s}" for j, s in enumerate(sentences))
        answer = rec["answer1"]
        if rec.get("answer2"):
            answer += f" / {rec['answer2']}"

        parts.append(
            f'--- QA Pair {i + 1} ---\n'
            f'Story: {rec["story_name"]}\n'
            f'Question: {rec["question"]}\n'
            f'Answer: {answer}\n'
            f'Metadata: local-or-sum={rec.get("local_or_sum", "N/A")}, '
            f'attribute={rec.get("attribute", "N/A")}, '
            f'ex-or-im={rec.get("ex_or_im", "N/A")}\n'
            f'Story section sentences:\n{sent_lines}'
        )

    qa_block = "\n\n".join(parts)

    return f"""You are an expert evidence analyst for narrative reading comprehension.

For each QA pair below, determine how many sentences from the story section
a reader MUST read to answer the question correctly.

## CRITICAL DISTINCTION

There are two different reasons extra sentences might be needed:

1. BACKGROUND CONTEXT: Extra sentences help understand the setting, characters,
   or situation. But the answer itself is directly stated in one sentence.

2. ANSWER IDENTIFICATION: The answer cannot be determined from a single sentence.
   The reader must combine information from multiple sentences to identify
   the correct answer.

Example of BACKGROUND CONTEXT (NOT Hard):
  Q: "What did the wolf do?"
  S3: "The wolf crept through the forest."
  S1: "Once upon a time, in a dark forest, there lived a wolf."
  -> S3 alone answers the question. S1 is background. answer_sentence_alone_sufficient=yes.

Example of ANSWER IDENTIFICATION (Hard):
  Q: "Why did the princess leave the castle?"
  S5: "She packed her bags that night."
  S3: "The king had forbidden her from seeing the young knight."
  S4: "The knight was banished to the eastern mountains."
  -> S5 says she left but not why. S3+S4 explain the motivation.
     answer_sentence_alone_sufficient=no, necessity_type=motivation_bridge.

## Evidence assessment fields

For each QA pair, provide:

1. answer_sentence_alone_sufficient:
   Is the most likely answer sentence alone enough to answer the question?
   - "yes": the sentence directly states the answer
   - "partial": the sentence gives a hint but needs one other sentence
   - "no": the sentence alone is ambiguous or insufficient

2. section_evidence_sufficient:
   Is the entire story section enough to answer the question?
   - "yes": the section fully contains the answer
   - "partial": the section helps but some inference is needed
   - "no": the question requires information beyond this section

3. full_context_needed:
   Does answering require the full story context (beyond the section)?
   - "yes": must read beyond the section
   - "partial": section is mostly sufficient
   - "no": section is fully sufficient

4. required_evidence_sentences:
   List of sentence indices [S0, S1, ...] that are REQUIRED to answer.

5. bridge_sentence_ids:
   Sentence indices that connect context to the answer. Empty if none.

6. num_required_sentences:
   How many sentences are required (len of required_evidence_sentences).

7. reasoning_operation:
   What kind of reasoning is needed?
   - "explicit_lookup": answer is directly stated
   - "temporal_order": answer depends on event sequence
   - "causal_chain": answer reached through cause-effect
   - "motivation": answer requires understanding character motivation
   - "character_state": answer about character feelings/beliefs
   - "disambiguation": multiple possible answers, need context to resolve
   - "summary_inference": answer requires summarizing multiple facts
   - "contrast": answer requires comparing/contrasting

8. bridge_removal_effect:
   If bridge sentences are removed:
   - "none": answer still findable
   - "harder": answer harder but possible
   - "ambiguous": answer becomes ambiguous
   - "unanswerable": answer cannot be determined

9. necessity_type:
   Why are extra sentences needed?
   - "background_context": extra sentences provide background only
   - "answer_identification": extra sentences needed to find the answer
   - "disambiguation": extra sentences resolve ambiguity
   - "causal_bridge": extra sentences provide causal chain
   - "temporal_bridge": extra sentences establish time sequence
   - "motivation_bridge": extra sentences explain character motivation
   - "summary_synthesis": answer requires synthesizing multiple facts

10. evidence_necessity_reason:
    One sentence explaining why.

## QA Pairs to assess

{qa_block}

## Output

Return a JSON object with key "assessments" containing a list, one entry per
QA pair.  Each entry MUST include ALL fields:
{{
  "qa_id": 1,
  "answer_sentence_alone_sufficient": "no",
  "section_evidence_sufficient": "yes",
  "full_context_needed": "no",
  "required_evidence_sentences": [2, 3, 5],
  "bridge_sentence_ids": [3],
  "num_required_sentences": 3,
  "reasoning_operation": "motivation",
  "bridge_removal_effect": "ambiguous",
  "necessity_type": "motivation_bridge",
  "evidence_necessity_reason": "S5 states the action but S3 and S4 explain
    the motivation. Without S3, the reader cannot determine why the character
    acted."
}}

Important:
- required_evidence_sentences must ONLY contain valid sentence indices from
  the story section shown above (0-based).
- Do NOT mark Hard merely because background context helps understand.
- If the answer is directly stated in one sentence, set
  answer_sentence_alone_sufficient=yes even if other sentences provide context.

Return ONLY the JSON object, no other text."""


def _parse_assessments(resp, num_qa):
    """Parse LLM response into a list of assessment dicts."""
    if not resp:
        return None

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
            if isinstance(data, dict) and "assessments" in data:
                assessments = data["assessments"]
                if isinstance(assessments, list) and len(assessments) == num_qa:
                    return assessments
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# Valid values
_VALID_ABILITY = {"yes", "partial", "no"}
_VALID_REMOVAL = {"none", "harder", "ambiguous", "unanswerable"}
_VALID_NEC_TYPE = {
    "background_context", "answer_identification", "disambiguation",
    "causal_bridge", "temporal_bridge", "motivation_bridge", "summary_synthesis",
}
_VALID_REASONING_OP = {
    "explicit_lookup", "temporal_order", "causal_chain", "motivation",
    "character_state", "disambiguation", "summary_inference", "contrast",
}


def _validate_assessment(a, num_sentences):
    """Validate and normalize an assessment dict.

    Args:
        a: raw assessment dict from LLM
        num_sentences: total sentences in the story section

    Returns:
        (assessment_dict, contradiction_count)
    """
    contradictions = 0
    valid_ids = set(range(num_sentences))

    # Validate lists
    raw_req = a.get("required_evidence_sentences", [])
    raw_bridge = a.get("bridge_sentence_ids", [])
    if not isinstance(raw_req, list):
        raw_req = []
    if not isinstance(raw_bridge, list):
        raw_bridge = []

    # Clamp to valid sentence indices
    req_ids = sorted(set(int(x) for x in raw_req if isinstance(x, (int, float)) and int(x) in valid_ids))
    bridge_ids = sorted(set(int(x) for x in raw_bridge if isinstance(x, (int, float)) and int(x) in valid_ids and int(x) in req_ids))
    a["required_evidence_sentences"] = req_ids
    a["bridge_sentence_ids"] = bridge_ids
    a["num_required_sentences"] = len(req_ids)

    # Validate enums
    for key in ("answer_sentence_alone_sufficient", "section_evidence_sufficient", "full_context_needed"):
        if a.get(key) not in _VALID_ABILITY:
            a[key] = "partial"
    if a.get("bridge_removal_effect") not in _VALID_REMOVAL:
        a["bridge_removal_effect"] = "harder"
    if a.get("necessity_type") not in _VALID_NEC_TYPE:
        a["necessity_type"] = "answer_identification"
    if a.get("reasoning_operation") not in _VALID_REASONING_OP:
        a["reasoning_operation"] = "explicit_lookup"
    if not isinstance(a.get("evidence_necessity_reason"), str):
        a["evidence_necessity_reason"] = ""

    # Fix contradictions
    num_req = a["num_required_sentences"]
    suff = a["answer_sentence_alone_sufficient"]

    # Contradiction: sufficiency=yes but has multiple required sentences
    if suff == "yes" and num_req > 1:
        contradictions += 1
        a["required_evidence_sentences"] = req_ids[:1] if req_ids else []
        a["bridge_sentence_ids"] = []
        a["num_required_sentences"] = len(a["required_evidence_sentences"])

    # Contradiction: num_required=1 but sufficiency != yes
    if a["num_required_sentences"] == 1 and a["answer_sentence_alone_sufficient"] != "yes":
        contradictions += 1
        a["answer_sentence_alone_sufficient"] = "yes"

    return a, contradictions


def classify_difficulty(assessment):
    """Classify Easy/Medium/Hard from the evidence assessment.

    Definitions (aligned with FINAL_PROPOSAL.md):
      Easy:   answer_sentence_alone_sufficient=yes AND num_required_sentences=1
      Medium: num_required_sentences=2
      Hard:   num_required_sentences>=3
    """
    suff = assessment.get("answer_sentence_alone_sufficient", "yes")
    num_req = assessment.get("num_required_sentences", 1)

    # Easy: exactly 1 required sentence AND that sentence alone suffices
    if suff == "yes" and num_req == 1:
        return "Easy"

    # Hard: 3 or more required sentences
    if num_req >= 3:
        return "Hard"

    # Medium: exactly 2 required sentences (or num_req=1 but ASA=no)
    return "Medium"


class FairytaleEvidenceAuditor:
    """Audit evidence difficulty for FairytaleQA QA pairs."""

    def __init__(self, batch_size=10, model=None, max_retries=2, timeout=120):
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout

        cfg = get_api_config()
        self.api_url = cfg["SILICONFLOW_API_URL"]
        self.api_key = cfg["SILICONFLOW_API_KEY"]
        self.model = model or cfg["JUDGE_MODEL"]

    def audit_batch(self, qa_records):
        """Audit a batch of QA pairs.

        Returns list of candidate dicts with evidence assessments.
        """
        prompt, raw_resp, assessments, parse_ok = self._assess(qa_records)

        results = []
        for i, rec in enumerate(qa_records):
            sentences = _split_sentences(rec["story_section"])
            num_sents = len(sentences)

            candidate = {
                "story_name": rec.get("story_name", ""),
                "story_section": rec.get("story_section", ""),
                "question": rec.get("question", ""),
                "answer": rec.get("answer1", ""),
                "answer1": rec.get("answer1", ""),
                "answer2": rec.get("answer2", ""),
                "local_or_sum": rec.get("local_or_sum", ""),
                "attribute": rec.get("attribute", ""),
                "ex_or_im": rec.get("ex_or_im", ""),
                "ex_or_im2": rec.get("ex_or_im2", ""),
                "split": rec.get("split", ""),
                "num_sentences_in_section": num_sents,
                # Trace fields
                "fairytale_evidence_prompt": prompt,
                "fairytale_evidence_raw": raw_resp or "",
                "fairytale_evidence_parse_ok": parse_ok,
                "fairytale_evidence_model": self.model,
            }

            if assessments and i < len(assessments):
                a, n_contra = _validate_assessment(assessments[i], num_sents)
                candidate.update({
                    "answer_sentence_alone_sufficient": a["answer_sentence_alone_sufficient"],
                    "section_evidence_sufficient": a["section_evidence_sufficient"],
                    "full_context_needed": a["full_context_needed"],
                    "required_evidence_sentences": a["required_evidence_sentences"],
                    "bridge_sentence_ids": a["bridge_sentence_ids"],
                    "num_required_sentences": a["num_required_sentences"],
                    "reasoning_operation": a["reasoning_operation"],
                    "bridge_removal_effect": a["bridge_removal_effect"],
                    "necessity_type": a["necessity_type"],
                    "evidence_necessity_reason": a["evidence_necessity_reason"],
                    "assessment_status": "ok",
                    "fairytale_evidence_status": "ok",
                    "contradiction_count": n_contra,
                })
            else:
                candidate.update({
                    "answer_sentence_alone_sufficient": "partial",
                    "section_evidence_sufficient": "partial",
                    "full_context_needed": "partial",
                    "required_evidence_sentences": [],
                    "bridge_sentence_ids": [],
                    "num_required_sentences": 0,
                    "reasoning_operation": "explicit_lookup",
                    "bridge_removal_effect": "harder",
                    "necessity_type": "background_context",
                    "evidence_necessity_reason": "assessment_failed",
                    "assessment_status": "llm_error",
                    "fairytale_evidence_status": "llm_error",
                    "contradiction_count": 0,
                })

            candidate["evidence_difficulty"] = classify_difficulty(candidate)
            results.append(candidate)

        return results

    def _assess(self, qa_records):
        """Call LLM to assess evidence necessity.

        Returns (prompt, raw_response, assessments_or_None, parse_ok).
        """
        prompt = _build_evidence_prompt(qa_records)

        for attempt in range(self.max_retries + 1):
            try:
                resp = call_openai_compatible(
                    prompt,
                    api_url=self.api_url,
                    api_key=self.api_key,
                    model=self.model,
                    temperature=0.0,
                    max_tokens=4000,
                    json_mode=True,
                    system="You are a precise evidence analyst for narrative QA. Return only valid JSON.",
                    timeout=self.timeout,
                )
                assessments = _parse_assessments(resp, len(qa_records))
                if assessments:
                    return (prompt, resp, assessments, True)
                if attempt == self.max_retries:
                    return (prompt, resp or "", None, False)
            except Exception as e:
                if attempt == self.max_retries:
                    return (prompt, f"ERROR: {e}", None, False)

        return (prompt, "", None, False)
