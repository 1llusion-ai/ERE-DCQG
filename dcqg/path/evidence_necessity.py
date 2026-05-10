"""Evidence Necessity Miner.

For each event in a document, assess whether the answer sentence alone is
sufficient to identify the answer, or whether anchor/bridge sentences are
required.  This determines true difficulty: Easy (single sentence), Medium
(two sentences), Hard (3+ necessary evidence steps).

Includes an ablation probe to distinguish:
  - context_needed_for_understanding (background context)
  - evidence_needed_for_answer_identification (true difficulty signal)

Phase 1: audit only — no question generation.
"""
import json
import re

from dcqg.graph.event_graph import EventGraph
from dcqg.path.answer_extraction import extract_answer_phrase_local
from dcqg.utils.api_client import call_openai_compatible
from dcqg.utils.config import get_api_config


GENERIC_TRIGGERS = {
    "occurred", "happened", "took place", "made", "did", "was", "were",
    "is", "are", "had", "has", "have", "said", "told", "asked",
    "went", "came", "got", "gave", "took", "put", "set",
}


def _is_valid_answer_event(trigger):
    """Reject generic or very short triggers that make poor answer targets."""
    t = trigger.lower().strip()
    if t in GENERIC_TRIGGERS:
        return False
    if len(t) <= 2:
        return False
    return True


def _format_context_window(graph, center_sent_id, window=5):
    """Return list of (sent_id, sentence_text) around center sentence."""
    n = len(graph.sentences)
    lo = max(0, center_sent_id - window)
    hi = min(n, center_sent_id + window + 1)
    return [(i, graph.get_sentence(i)) for i in range(lo, hi)]


def _build_assessment_prompt(doc_title, candidates):
    """Build a single LLM prompt that assesses evidence necessity for
    multiple answer candidates from the same document.

    Each candidate dict has:
      - candidate_id, trigger, event_type, answer_phrase
      - answer_sent_id, answer_sentence
      - context_sentences: [(sent_id, text), ...]
    """
    parts = []
    for c in candidates:
        ctx_lines = "\n".join(
            f"  [S{sid}] {text}" for sid, text in c["context_sentences"]
        )
        parts.append(
            f'--- Candidate {c["candidate_id"]} ---\n'
            f'Answer event trigger: "{c["trigger"]}" (type: {c["event_type"]})\n'
            f'Answer phrase: "{c["answer_phrase"]}"\n'
            f'Answer sentence [S{c["answer_sent_id"]}]: "{c["answer_sentence"]}"\n'
            f'Context sentences:\n{ctx_lines}'
        )

    candidates_block = "\n\n".join(parts)

    return f"""You are an expert evidence analyst for reading comprehension.

Document: "{doc_title}"

For each candidate answer event below, determine the MINIMUM set of sentences
a reader MUST read to CONFIDENTLY IDENTIFY the answer phrase. The reader does
NOT know which sentence contains the answer — they must find it from context.

## CRITICAL DISTINCTION

There are two different reasons extra sentences might be needed:

1. BACKGROUND CONTEXT: Extra sentences help the reader UNDERSTAND the topic,
   the entities involved, or the historical situation. The answer sentence
   itself still directly states the answer, but the reader benefits from
   knowing the background.

2. ANSWER IDENTIFICATION: The answer sentence alone does NOT give enough
   information to determine which phrase IS the answer. The reader literally
   cannot identify the correct answer without the extra sentences.

Example of BACKGROUND CONTEXT (NOT Hard):
  Answer sentence: "The treaty was signed in Paris in 1783."
  Background: "The American Revolution was a war between Britain and colonies."
  -> A reader seeing only the answer sentence CAN extract "signed in Paris in
     1783" as the answer. Background helps understanding but is NOT needed to
     identify the answer phrase. Set answer_only_can_identify_answer=yes.

Example of ANSWER IDENTIFICATION (Hard):
  Answer sentence: "He then led the final assault."
  Without context: Who is "he"? Which assault? The sentence alone is ambiguous.
  With context: S3 introduces "General Washington", S5 describes the siege.
  -> A reader seeing only the answer sentence CANNOT identify the answer.
     Set answer_only_can_identify_answer=no.

## Evidence role definitions

- answer_sentence_id: The sentence that contains the answer phrase.
- anchor_sentence_ids: Sentence(s) the reader needs FIRST to understand what
  the question is about.  Empty list if none required.
- bridge_sentence_ids: Sentence(s) that connect the anchor to the answer
  through intermediate information.  Empty list if none.

## Ablation probe fields

For each candidate, you MUST answer these ablation questions:

1. answer_only_can_identify_answer:
   If a reader sees ONLY the answer sentence (no other sentences), can they
   identify the answer phrase?
   - "yes": the answer phrase is directly stated and identifiable
   - "partial": the reader can guess but not be confident
   - "no": the sentence is ambiguous; the reader cannot determine the answer

2. anchor_answer_can_identify_answer:
   If a reader sees the answer sentence + anchor sentence(s) only (no bridge),
   can they identify the answer phrase?
   - "yes" / "partial" / "no"

3. full_evidence_can_identify_answer:
   If a reader sees ALL evidence sentences (anchor + bridge + answer), can they
   identify the answer phrase?
   - "yes" / "partial" / "no"

4. bridge_removal_effect:
   If the bridge sentence(s) are removed, what happens?
   - "none": the answer is still identifiable without bridge
   - "harder": the answer is harder to find but still possible
   - "ambiguous": the answer becomes ambiguous (multiple candidates)
   - "unanswerable": the answer cannot be determined at all

5. necessity_type:
   Why are extra sentences needed? Choose the most specific type:
   - "background_context": extra sentences provide background understanding,
     but the answer sentence alone still gives the answer directly
   - "answer_identification": extra sentences are needed to determine which
     phrase in the answer sentence IS the correct answer
   - "disambiguation": extra sentences resolve ambiguity (e.g., "he" could
     refer to multiple people)
   - "causal_bridge": extra sentences provide causal chain needed to
     understand what caused the answer event
   - "temporal_bridge": extra sentences establish temporal sequence needed
     to identify the correct event

6. ablation_reason: One sentence explaining WHY the ablation result holds.

## Decision criteria (existing)

answer_sentence_alone_sufficient:
  - "yes"    = a reader seeing ONLY the answer sentence can extract the answer
  - "partial"= the answer sentence gives a hint but needs one other sentence
  - "no"     = the answer sentence alone is ambiguous or unidentifiable

evidence_necessity:
  - "strong" = removing any required sentence makes the answer unidentifiable
  - "partial"= removing a required sentence makes it harder but not impossible
  - "weak"  = the answer is mostly findable from the answer sentence alone

answer_locality:
  - "single_sentence" = only the answer sentence is needed
  - "two_sentence"    = answer sentence + exactly one other
  - "multi_sentence"  = 3+ sentences required

## Candidates to assess

{candidates_block}

## Output

Return a JSON object with key "assessments" containing a list, one entry per
candidate.  Each entry MUST include ALL fields:
{{
  "candidate_id": 1,
  "answer_sentence_id": 5,
  "anchor_sentence_ids": [3],
  "bridge_sentence_ids": [4],
  "evidence_span": [3, 4, 5],
  "num_required_sentences": 3,
  "answer_locality": "multi_sentence",
  "reasoning_operation": "bridge",
  "answer_sentence_alone_sufficient": "no",
  "evidence_necessity": "strong",
  "evidence_necessity_reason": "...",
  "answer_only_can_identify_answer": "no",
  "anchor_answer_can_identify_answer": "partial",
  "full_evidence_can_identify_answer": "yes",
  "bridge_removal_effect": "ambiguous",
  "necessity_type": "disambiguation",
  "ablation_reason": "Without S4, 'he' could refer to either commander."
}}

Important:
- answer_sentence_id MUST equal the answer sentence ID shown above.
- anchor_sentence_ids and bridge_sentence_ids must ONLY contain IDs from the
  context sentences listed above.
- Do NOT mark Hard merely because background context helps understand the
  answer.  Mark Hard only if the answer sentence alone is INSUFFICIENT to
  identify the answer phrase.
- If answer_only_can_identify_answer=yes, the candidate is NOT Hard regardless
  of how many context sentences exist.

Return ONLY the JSON object, no other text."""


def _parse_assessments(resp, num_candidates):
    """Parse LLM response into a list of assessment dicts."""
    if not resp:
        return None

    # Try direct JSON parse
    try:
        data = json.loads(resp)
        if isinstance(data, dict) and "assessments" in data:
            assessments = data["assessments"]
            if isinstance(assessments, list) and len(assessments) == num_candidates:
                return assessments
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    try:
        s_idx = resp.index("{")
        e_idx = resp.rindex("}") + 1
        data = json.loads(resp[s_idx:e_idx])
        if isinstance(data, dict) and "assessments" in data:
            assessments = data["assessments"]
            if isinstance(assessments, list) and len(assessments) == num_candidates:
                return assessments
    except (ValueError, json.JSONDecodeError):
        pass

    # Try line-by-line JSON extraction
    cleaned = resp.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "assessments" in data:
            assessments = data["assessments"]
            if isinstance(assessments, list) and len(assessments) == num_candidates:
                return assessments
    except json.JSONDecodeError:
        pass

    return None


# Valid values for ablation fields
_VALID_ABILITY = {"yes", "partial", "no"}
_VALID_REMOVAL = {"none", "harder", "ambiguous", "unanswerable"}
_VALID_NECESSITY_TYPE = {
    "background_context", "answer_identification", "disambiguation",
    "causal_bridge", "temporal_bridge",
}


def _validate_assessment(a, answer_sent_id, context_window_ids):
    """Validate and normalize a single assessment dict.

    Args:
        a: raw assessment dict from LLM
        answer_sent_id: the true answer sentence ID (forced)
        context_window_ids: set of valid sentence IDs in the context window

    Returns:
        (assessment_dict, contradiction_count)
    """
    contradictions = 0
    valid_locality = {"single_sentence", "two_sentence", "multi_sentence"}
    valid_sufficiency = {"yes", "partial", "no"}
    valid_necessity = {"weak", "partial", "strong"}
    valid_operations = {
        "bridge", "contrast", "temporal_order", "causal_chain",
        "disambiguation", "comparison",
    }

    # --- Force answer_sentence_id ---
    a["answer_sentence_id"] = answer_sent_id

    # --- Clamp anchor/bridge IDs to context window ---
    raw_anchor = a.get("anchor_sentence_ids", [])
    raw_bridge = a.get("bridge_sentence_ids", [])
    if not isinstance(raw_anchor, list):
        raw_anchor = []
    if not isinstance(raw_bridge, list):
        raw_bridge = []

    anchor_ids = sorted(set(
        int(x) for x in raw_anchor
        if isinstance(x, (int, float)) and int(x) in context_window_ids
        and int(x) != answer_sent_id
    ))
    bridge_ids = sorted(set(
        int(x) for x in raw_bridge
        if isinstance(x, (int, float)) and int(x) in context_window_ids
        and int(x) != answer_sent_id
        and int(x) not in anchor_ids
    ))
    a["anchor_sentence_ids"] = anchor_ids
    a["bridge_sentence_ids"] = bridge_ids

    # --- Recompute evidence_span and num_required_sentences ---
    evidence_span = sorted(set([answer_sent_id] + anchor_ids + bridge_ids))
    a["evidence_span"] = evidence_span
    a["num_required_sentences"] = len(evidence_span)

    # --- Validate existing enum fields ---
    if a.get("answer_locality") not in valid_locality:
        a["answer_locality"] = "single_sentence"
    if a.get("answer_sentence_alone_sufficient") not in valid_sufficiency:
        a["answer_sentence_alone_sufficient"] = "partial"
    if a.get("evidence_necessity") not in valid_necessity:
        a["evidence_necessity"] = "partial"
    if a.get("reasoning_operation") not in valid_operations:
        a["reasoning_operation"] = "bridge"
    if not isinstance(a.get("evidence_necessity_reason"), str):
        a["evidence_necessity_reason"] = ""

    # --- Validate ablation fields ---
    for key in ("answer_only_can_identify_answer",
                "anchor_answer_can_identify_answer",
                "full_evidence_can_identify_answer"):
        if a.get(key) not in _VALID_ABILITY:
            a[key] = "partial"

    if a.get("bridge_removal_effect") not in _VALID_REMOVAL:
        a["bridge_removal_effect"] = "harder"

    if a.get("necessity_type") not in _VALID_NECESSITY_TYPE:
        a["necessity_type"] = "answer_identification"

    if not isinstance(a.get("ablation_reason"), str):
        a["ablation_reason"] = ""

    # --- Fix contradictions ---
    sufficiency = a["answer_sentence_alone_sufficient"]
    num_req = a["num_required_sentences"]

    # Contradiction 1: sufficiency=yes but has anchors/bridges
    if sufficiency == "yes" and num_req > 1:
        contradictions += 1
        a["anchor_sentence_ids"] = []
        a["bridge_sentence_ids"] = []
        a["evidence_span"] = [answer_sent_id]
        a["num_required_sentences"] = 1
        a["answer_locality"] = "single_sentence"
        a["evidence_necessity"] = "weak"

    # Contradiction 2: num_required=1 but sufficiency != yes
    if a["num_required_sentences"] == 1 and a["answer_sentence_alone_sufficient"] != "yes":
        contradictions += 1
        a["answer_sentence_alone_sufficient"] = "yes"
        a["answer_locality"] = "single_sentence"
        a["evidence_necessity"] = "weak"

    # Contradiction 3: locality mismatch with num_required
    num_req = a["num_required_sentences"]
    if num_req == 1 and a["answer_locality"] != "single_sentence":
        contradictions += 1
        a["answer_locality"] = "single_sentence"
    elif num_req == 2 and a["answer_locality"] != "two_sentence":
        contradictions += 1
        a["answer_locality"] = "two_sentence"
    elif num_req >= 3 and a["answer_locality"] != "multi_sentence":
        contradictions += 1
        a["answer_locality"] = "multi_sentence"

    # Contradiction 4: answer_only=yes but answer_sentence_alone_sufficient=no
    if (a.get("answer_only_can_identify_answer") == "yes"
            and a["answer_sentence_alone_sufficient"] == "no"):
        contradictions += 1
        a["answer_sentence_alone_sufficient"] = "yes"
        a["anchor_sentence_ids"] = []
        a["bridge_sentence_ids"] = []
        a["evidence_span"] = [answer_sent_id]
        a["num_required_sentences"] = 1
        a["answer_locality"] = "single_sentence"
        a["evidence_necessity"] = "weak"

    return a, contradictions


def classify_evidence_difficulty(assessment):
    """Classify potential Easy/Medium/Hard from evidence necessity.

    Conservative: Hard requires ALL of:
      - answer_sentence_alone_sufficient == "no"
      - answer_locality == "multi_sentence"
      - num_required_sentences >= 3
      - evidence_necessity == "strong"

    Everything else falls to Easy or Medium.
    """
    sufficiency = assessment.get("answer_sentence_alone_sufficient", "yes")
    locality = assessment.get("answer_locality", "single_sentence")
    num_req = assessment.get("num_required_sentences", 1)
    necessity = assessment.get("evidence_necessity", "weak")

    if sufficiency == "yes" or num_req <= 1:
        return "Easy"

    if (sufficiency == "no"
            and locality == "multi_sentence"
            and num_req >= 3
            and necessity == "strong"):
        return "Hard"

    return "Medium"


def classify_verified_difficulty(assessment):
    """Classify verified difficulty with ablation probe.

    Verified Hard requires ALL of:
      - potential difficulty is Hard (the four structural conditions)
      - answer_only_can_identify_answer == "no"
      - full_evidence_can_identify_answer == "yes"
      - bridge_removal_effect in {ambiguous, unanswerable}
      - necessity_type in {answer_identification, disambiguation,
                           causal_bridge, temporal_bridge}

    If necessity_type == background_context, always downgrade to Medium.
    """
    potential = assessment.get("potential_evidence_difficulty",
                               classify_evidence_difficulty(assessment))
    answer_only = assessment.get("answer_only_can_identify_answer", "yes")
    full_evidence = assessment.get("full_evidence_can_identify_answer", "yes")
    removal = assessment.get("bridge_removal_effect", "none")
    nec_type = assessment.get("necessity_type", "background_context")

    # If potential is not Hard, verified cannot be Hard either
    if potential != "Hard":
        if potential == "Easy":
            return "Easy"
        return "Medium"

    # background_context always downgrades
    if nec_type == "background_context":
        return "Medium"

    # All five conditions must hold for verified Hard
    if (answer_only == "no"
            and full_evidence == "yes"
            and removal in ("ambiguous", "unanswerable")
            and nec_type in ("answer_identification", "disambiguation",
                             "causal_bridge", "temporal_bridge")):
        return "Hard"

    # Potential Hard but ablation shows it's not truly Hard
    return "Medium"


class EvidenceNecessityMiner:
    """Mine evidence necessity for answer candidates from documents."""

    def __init__(self, context_window=5, max_candidates_per_doc=15,
                 model=None, max_retries=2):
        self.context_window = context_window
        self.max_candidates_per_doc = max_candidates_per_doc
        self.max_retries = max_retries

        cfg = get_api_config()
        self.api_url = cfg["SILICONFLOW_API_URL"]
        self.api_key = cfg["SILICONFLOW_API_KEY"]
        self.model = model or cfg["JUDGE_MODEL"]

    def mine_document(self, doc):
        """Mine evidence necessity for all event candidates in a document.

        Returns list of candidate dicts with evidence assessments and traces.
        """
        graph = EventGraph(doc)
        if not graph.events:
            return []

        # Extract candidates
        raw_candidates = self._extract_candidates(graph)
        if not raw_candidates:
            return []

        # Add candidate_id and batch into LLM call
        batch = raw_candidates[:self.max_candidates_per_doc]
        for idx, rc in enumerate(batch):
            rc["candidate_id"] = idx + 1

        prompt, raw_resp, assessments, parse_ok = self._assess_batch(
            graph.title, batch
        )

        candidates = []
        for i, rc in enumerate(batch):
            candidate = {
                "doc_id": graph.doc_id,
                "title": graph.title,
                "answer_event_id": rc["event_id"],
                "answer_trigger": rc["trigger"],
                "answer_event_type": rc["event_type"],
                "answer_phrase": rc["answer_phrase"],
                "answer_phrase_status": rc["answer_phrase_status"],
                "answer_sent_id": rc["answer_sent_id"],
                "answer_sentence": rc["answer_sentence"],
                "context_sentences": rc["context_sentences"],
                # Trace fields
                "evidence_assessment_prompt": prompt,
                "evidence_assessment_raw": raw_resp or "",
                "evidence_assessment_parse_ok": parse_ok,
                "evidence_assessment_model": self.model,
            }

            context_ids = {sid for sid, _ in rc["context_sentences"]}

            if assessments and i < len(assessments):
                a, n_contra = _validate_assessment(
                    assessments[i], rc["answer_sent_id"], context_ids
                )
                candidate.update({
                    "answer_sentence_id": a["answer_sentence_id"],
                    "anchor_sentence_ids": a["anchor_sentence_ids"],
                    "bridge_sentence_ids": a["bridge_sentence_ids"],
                    "evidence_span": a["evidence_span"],
                    "num_required_sentences": a["num_required_sentences"],
                    "answer_locality": a["answer_locality"],
                    "reasoning_operation": a["reasoning_operation"],
                    "answer_sentence_alone_sufficient": a["answer_sentence_alone_sufficient"],
                    "evidence_necessity": a["evidence_necessity"],
                    "evidence_necessity_reason": a["evidence_necessity_reason"],
                    # Ablation fields
                    "answer_only_can_identify_answer": a["answer_only_can_identify_answer"],
                    "anchor_answer_can_identify_answer": a["anchor_answer_can_identify_answer"],
                    "full_evidence_can_identify_answer": a["full_evidence_can_identify_answer"],
                    "bridge_removal_effect": a["bridge_removal_effect"],
                    "necessity_type": a["necessity_type"],
                    "ablation_reason": a["ablation_reason"],
                    # Status
                    "assessment_status": "ok",
                    "contradiction_count": n_contra,
                })
            else:
                candidate.update({
                    "answer_sentence_id": rc["answer_sent_id"],
                    "anchor_sentence_ids": [],
                    "bridge_sentence_ids": [],
                    "evidence_span": [rc["answer_sent_id"]],
                    "num_required_sentences": 1,
                    "answer_locality": "single_sentence",
                    "reasoning_operation": "bridge",
                    "answer_sentence_alone_sufficient": "yes",
                    "evidence_necessity": "weak",
                    "evidence_necessity_reason": "assessment_failed",
                    # Ablation defaults — easy
                    "answer_only_can_identify_answer": "yes",
                    "anchor_answer_can_identify_answer": "yes",
                    "full_evidence_can_identify_answer": "yes",
                    "bridge_removal_effect": "none",
                    "necessity_type": "background_context",
                    "ablation_reason": "assessment_failed",
                    # Status
                    "assessment_status": "llm_error",
                    "contradiction_count": 0,
                })

            candidate["potential_evidence_difficulty"] = classify_evidence_difficulty(candidate)
            candidate["verified_evidence_difficulty"] = classify_verified_difficulty(candidate)
            candidates.append(candidate)

        return candidates

    def _extract_candidates(self, graph):
        """Extract answer candidates from document events."""
        candidates = []
        seen_triggers = set()

        for eid, event in graph.events.items():
            mentions = event.get("mention", [])
            if not mentions:
                continue

            m = mentions[0]
            trigger = m.get("trigger_word", "")
            sent_id = m.get("sent_id", 0)

            if not _is_valid_answer_event(trigger):
                continue

            # Deduplicate by (trigger, sent_id)
            dedup_key = (trigger.lower(), sent_id)
            if dedup_key in seen_triggers:
                continue
            seen_triggers.add(dedup_key)

            answer_sentence = graph.get_sentence(sent_id)
            if not answer_sentence:
                continue

            answer_phrase, phrase_status = extract_answer_phrase_local(
                answer_sentence, trigger
            )
            if phrase_status == "invalid":
                continue

            context_sentences = _format_context_window(
                graph, sent_id, window=self.context_window
            )

            candidates.append({
                "event_id": eid,
                "trigger": trigger,
                "event_type": event.get("type", "N/A"),
                "answer_sent_id": sent_id,
                "answer_sentence": answer_sentence,
                "answer_phrase": answer_phrase,
                "answer_phrase_status": phrase_status,
                "context_sentences": context_sentences,
            })

        return candidates

    def _assess_batch(self, doc_title, candidates):
        """Call LLM to assess evidence necessity for a batch of candidates.

        Returns (prompt, raw_response, assessments_or_None, parse_ok).
        """
        if not candidates:
            return ("", "", None, False)

        prompt = _build_assessment_prompt(doc_title, candidates)

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
                    system="You are a precise evidence analyst. Return only valid JSON.",
                    timeout=120,
                )
                assessments = _parse_assessments(resp, len(candidates))
                if assessments:
                    return (prompt, resp, assessments, True)
                # Parse failed — save raw for trace, retry
                if attempt == self.max_retries:
                    return (prompt, resp or "", None, False)
            except Exception as e:
                if attempt == self.max_retries:
                    return (prompt, f"ERROR: {e}", None, False)

        return (prompt, "", None, False)
