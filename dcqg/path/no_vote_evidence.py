"""No-vote evidence selection and blind verification for FairytaleQA.

Stage 1 selects a complete minimal evidence set from the full section.
Stage 2 verifies sufficiency using only selected evidence. By default, this
verification is an annotation-priority signal, not a hard filter.

All prompts use frozen canonical text — do not paraphrase, shorten, or rewrite.
"""

import json
import re

from dcqg.path.fairytale_evidence_audit import _split_story_sentences, _check_split_anomalies
from dcqg.utils.api_client import call_openai_compatible
from dcqg.utils.config import get_api_config


# ═══════════════════════════════════════════════════════════════════
# Frozen prompt components — do not edit text
# ═══════════════════════════════════════════════════════════════════

SELECTOR_FEW_SHOTS = """Few-shot examples:

Example 1: single-sentence explicit
Context:
  [S0] then she bade him mount her palfrey again , and they rode on .
  [S1] the ferny road was not so bonnie all the way as it had been at first , however .
  [S2] for they had not ridden along it very far before it led them into a narrow ravine .
  [S3] there was a sound of rushing water everywhere .
  [S4] his courage had been slowly ebbing .
  [S5] he fell forward in a kind of swoon ; and , if it had not been that he had tight hold of the fairy's ash-grey gown , i warrant he had fallen from his seat , and had been drowned .
QA:
Question: what did thomas do to keep himself from falling off the palfrey ?
Target answer: held onto the fairy's ash-grey gown .
Output:
{"section_sufficient":"yes","selected_evidence_sentences":[5],"answer_directly_found":"yes","reasoning_level":"direct","evidence_reason":"S5 directly states that Thomas kept from falling because he had tight hold of the fairy's gown."}

Example 2: single-sentence implicit
Context:
  [S0] when the door creaked open , tom covered his eyes and trembled under the blanket .
  [S1] the candle burned low beside the bed .
  [S2] outside , the wind shook the shutters .
QA:
Question: how did tom feel when the door creaked open ?
Target answer: frightened .
Output:
{"section_sufficient":"yes","selected_evidence_sentences":[0],"answer_directly_found":"no","reasoning_level":"simple","evidence_reason":"S0 does not directly say Tom was frightened, but covering his eyes and trembling supports that simple inference."}

Example 3: multi-sentence direct synthesis
Context:
  [S0] the old woman gave the boy a silver key .
  [S1] the gardener gave the boy a red apple .
  [S2] the queen gave the boy a warm cloak .
  [S3] then the boy left the palace before sunset .
QA:
Question: what things was the boy given before he left the palace ?
Target answer: a silver key , a red apple , and a warm cloak .
Output:
{"section_sufficient":"yes","selected_evidence_sentences":[0,1,2,3],"answer_directly_found":"yes","reasoning_level":"simple","evidence_reason":"S0, S1, and S2 directly state the three things he was given, and S3 establishes that these gifts occurred before he left the palace."}

Example 4: multi-sentence implicit
Context:
  [S0] the king promised that whoever saved the princess could marry her .
  [S1] the youngest brother heard the promise but said nothing .
  [S2] that night he crossed the black river and fought the dragon .
  [S3] by morning the princess was safe again .
QA:
Question: why did the youngest brother cross the black river ?
Target answer: to save the princess so he could marry her .
Output:
{"section_sufficient":"yes","selected_evidence_sentences":[0,1,2,3],"answer_directly_found":"no","reasoning_level":"complex","evidence_reason":"The answer is not directly stated. S0 gives the reward, S1 links the brother to the promise, S2 gives his action, and S3 confirms the rescue outcome."}

Example 5: quoted speech boundaries
Context:
  [S0] the boy said , " i lost the brass key . "
  [S1] " i searched the garden until sunset . "
  [S2] " then you cannot open the gate , " replied the queen .
  [S3] the gate remained shut until morning .
QA:
Question: why could the boy not open the gate ?
Target answer: he had lost the brass key .
Output:
{"section_sufficient":"yes","selected_evidence_sentences":[0],"answer_directly_found":"yes","reasoning_level":"direct","evidence_reason":"S0 is a complete quoted sentence with local attribution and directly states that the boy lost the brass key. S1 is a separate quoted sentence from the same speech turn, and S2 is a different speaker's reply; neither is necessary for the target answer."}

Example 6: insufficient context
Context:
  [S0] silverwhite went through the city and asked why everyone was unhappy .
  [S1] they said the oldest princess would be taken away in the morning .
  [S2] this news pleased silverwhite , for he saw an opportunity for fame .
QA:
Question: what will silverwhite do next ?
Target answer: save the oldest princess .
Output:
{"section_sufficient":"no","selected_evidence_sentences":[],"answer_directly_found":"no","reasoning_level":"unknown","evidence_reason":"The context suggests an opportunity but does not contain enough evidence that Silverwhite will save the princess."}"""


SELECTOR_REQUIREMENTS = """Requirements:
1. Use only the numbered sentences in Context.
2. Treat the target answer as a claim to verify, not as evidence.
3. Select all and only the necessary evidence sentences.
4. Do not include background sentences unless removing them would make the answer unsupported, ambiguous, or require guessing.
5. Set answer_directly_found="yes" only when the target answer or a close paraphrase is directly present in the selected evidence.
6. Set reasoning_level as:
   - "direct": no substantive inference is needed.
   - "simple": one simple inference or simple synthesis is needed.
   - "complex": multi-step reasoning, aggregation, comparison, causal/temporal chaining, or tracking multiple entities is needed.
   - "unknown": use only when section_sufficient="no".
7. If Context is insufficient, set section_sufficient="no" and selected_evidence_sentences=[].
8. Return only valid JSON."""


BLIND_VERIFIER_FEW_SHOTS = """Few-shot examples:

Example A: sufficient, direct
Context:
  [S5] he fell forward in a kind of swoon ; and , if it had not been that he had tight hold of the fairy's ash-grey gown , i warrant he had fallen from his seat , and had been drowned .
QA:
Question: what did thomas do to keep himself from falling off the palfrey ?
Target answer: held onto the fairy's ash-grey gown .
Output:
{"sufficient":"yes","answer_directly_found":"yes","reasoning_level":"direct","reasoning":"S5 directly states that Thomas avoided falling because he had tight hold of the fairy's gown."}

Example B: sufficient, single-sentence simple inference
Context:
  [S0] when the door creaked open , tom covered his eyes and trembled under the blanket .
QA:
Question: how did tom feel when the door creaked open ?
Target answer: frightened .
Output:
{"sufficient":"yes","answer_directly_found":"no","reasoning_level":"simple","reasoning":"S0 does not directly state 'frightened', but covering his eyes and trembling supports that simple inference."}

Example C: sufficient, multi-sentence direct synthesis
Context:
  [S0] the old woman gave the boy a silver key .
  [S1] the gardener gave the boy a red apple .
  [S2] the queen gave the boy a warm cloak .
  [S3] then the boy left the palace before sunset .
QA:
Question: what things was the boy given before he left the palace ?
Target answer: a silver key , a red apple , and a warm cloak .
Output:
{"sufficient":"yes","answer_directly_found":"yes","reasoning_level":"simple","reasoning":"S0, S1, and S2 directly state the three things he was given, and S3 establishes that these gifts occurred before he left the palace."}

Example D: sufficient, multi-sentence implicit
Context:
  [S0] the king promised that whoever saved the princess could marry her .
  [S1] the youngest brother heard the promise but said nothing .
  [S2] that night he crossed the black river and fought the dragon .
  [S3] by morning the princess was safe again .
QA:
Question: why did the youngest brother cross the black river ?
Target answer: to save the princess so he could marry her .
Output:
{"sufficient":"yes","answer_directly_found":"no","reasoning_level":"complex","reasoning":"The target answer is not directly stated. S0 gives the reward, S1 links the brother to the promise, S2 gives his action, and S3 confirms the rescue outcome."}

Example E: sufficient, quoted speech
Context:
  [S0] the boy said , " i lost the brass key . "
QA:
Question: why could the boy not open the gate ?
Target answer: he had lost the brass key .
Output:
{"sufficient":"yes","answer_directly_found":"yes","reasoning_level":"direct","reasoning":"S0 is a complete quoted sentence with local attribution and directly states that the boy lost the brass key."}

Example F: not sufficient
Context:
  [S2] but this they did not know , and hence sold the ring for a small sum .
QA:
Question: why did the man and his wife sell the ring for a small sum ?
Target answer: they did not know that it was a lucky ring .
Output:
{"sufficient":"no","answer_directly_found":"no","reasoning_level":"unknown","reasoning":"S2 says they did not know this, but without the sentence explaining that the ring was lucky, the target answer is unsupported."}"""


BLIND_VERIFIER_REQUIREMENTS = """Requirements:
1. Use only the evidence sentences in Context.
2. The full story is not available.
3. Treat the target answer as a claim to verify, not as evidence.
4. If the evidence is enough to justify the target answer, output sufficient="yes".
5. If key information is missing, ambiguous, or only guessable, output sufficient="no".
6. If sufficient="yes", judge answer_directly_found and reasoning_level using only the provided evidence.
7. If sufficient="no", set answer_directly_found="no" and reasoning_level="unknown".
8. Return only valid JSON."""


REMOVAL_VERIFIER_FEW_SHOTS = """Few-shot examples:

Example R1: not sufficient after removing direct answer evidence
Context:
  [NO EVIDENCE SENTENCES PROVIDED]
QA:
Question: what did thomas do to keep himself from falling off the palfrey ?
Target answer: held onto the fairy's ash-grey gown .
Output:
{"sufficient":"no","answer_directly_found":"no","reasoning_level":"unknown","reasoning":"The direct evidence sentence is removed, so there is no support for the target answer."}

Example R2: not sufficient after removing a range/constraint sentence
Context:
  [S0] the old woman gave the boy a silver key .
  [S1] the gardener gave the boy a red apple .
  [S2] the queen gave the boy a warm cloak .
QA:
Question: what things was the boy given before he left the palace ?
Target answer: a silver key , a red apple , and a warm cloak .
Output:
{"sufficient":"no","answer_directly_found":"no","reasoning_level":"unknown","reasoning":"The gifts are stated, but the remaining evidence does not establish that they were given before he left the palace."}

Example R3: not sufficient after removing a bridge/motivation sentence
Context:
  [S1] the youngest brother heard the promise but said nothing .
  [S2] that night he crossed the black river and fought the dragon .
  [S3] by morning the princess was safe again .
QA:
Question: why did the youngest brother cross the black river ?
Target answer: to save the princess so he could marry her .
Output:
{"sufficient":"no","answer_directly_found":"no","reasoning_level":"unknown","reasoning":"The remaining evidence shows the brother acted and the princess became safe, but without the promised reward it does not support the target reason."}

Example R4: sufficient after removing non-essential background
Context:
  [S0] when the door creaked open , tom covered his eyes and trembled under the blanket .
QA:
Question: how did tom feel when the door creaked open ?
Target answer: frightened .
Output:
{"sufficient":"yes","answer_directly_found":"no","reasoning_level":"simple","reasoning":"The remaining sentence still supports the simple inference that Tom was frightened."}"""


REMOVAL_VERIFIER_REQUIREMENTS = """Requirements:
1. Use only the remaining evidence sentences in Context.
2. The full story is not available.
3. The removed sentence is not available.
4. Treat the target answer as a claim to verify, not as evidence.
5. If the remaining evidence is still enough to justify the target answer, output sufficient="yes".
6. If key information is missing, ambiguous, or only guessable, output sufficient="no".
7. If sufficient="yes", judge answer_directly_found and reasoning_level using only the remaining evidence.
8. If sufficient="no", set answer_directly_found="no" and reasoning_level="unknown".
9. Return only valid JSON."""


# ═══════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════

def _extract_json_object(raw):
    if not raw:
        return None
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[-1].strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError, TypeError):
        return None


def _normalize_yes_no(value, default="no"):
    if isinstance(value, str) and value.strip().lower() in {"yes", "no"}:
        return value.strip().lower()
    return default


def _normalize_reasoning_level(value, default="unknown"):
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"direct", "simple", "complex", "unknown"}:
            return value
    return default


def _safe_sentence_ids(raw_ids, num_sentences):
    if not isinstance(raw_ids, list):
        return []
    valid = set(range(num_sentences))
    ids = []
    for value in raw_ids:
        if isinstance(value, (int, float)) and int(value) in valid:
            ids.append(int(value))
    return sorted(set(ids))


def _answer_text(record):
    answer = record.get("answer1", "") or record.get("answer", "")
    if record.get("answer2"):
        answer = f"{answer} / {record['answer2']}"
    return answer


def _difficulty_from_ids(sentence_ids, answer_directly_found=None):
    """Label mapping consistent with canonical difficulty definitions.

    Easy  = answer_directly_found == "yes" and num_required_sentences == 1
    Medium = answer_directly_found == "no"  and num_required_sentences == 1
           OR answer_directly_found == "yes" and num_required_sentences >= 2
    Hard  = answer_directly_found == "no"  and num_required_sentences >= 2
    """
    direct = _normalize_yes_no(answer_directly_found, default=None)
    if direct is None:
        direct = "yes" if len(sentence_ids) == 1 else "no"
    num_req = len(sentence_ids)

    if num_req <= 0:
        return "Invalid"
    if direct == "yes" and num_req == 1:
        return "Easy"
    if direct == "no" and num_req == 1:
        return "Medium"
    if direct == "yes" and num_req >= 2:
        return "Medium"
    if direct == "no" and num_req >= 2:
        return "Hard"
    return "Invalid"


# ═══════════════════════════════════════════════════════════════════
# Prompt builders
# ═══════════════════════════════════════════════════════════════════

def build_selector_prompt(records):
    """Build SELECTOR prompt with frozen text.

    Structure: {few_shot} / {context} / {QA} / {requirements}
    """
    parts = []
    for idx, record in enumerate(records, start=1):
        sentences = _split_story_sentences(record.get("story_section", ""))
        sent_lines = "\n".join(
            f"  [S{i}] {sentence}" for i, sentence in enumerate(sentences)
        )
        parts.append(
            f"--- QA Pair {idx} ---\n"
            f"Context:\n{sent_lines}\n"
            "\n"
            f"QA:\n"
            f"Question: {record.get('question', '')}\n"
            f"Target answer: {_answer_text(record)}"
        )

    qa_block = "\n\n".join(parts)

    return f"""You are an expert evidence selector for narrative reading comprehension.
Your task is to select the smallest complete set of necessary evidence sentences that justifies the target answer.

A necessary evidence sentence is a sentence that must be available for a reader to justify the target answer. If removing the sentence would make the answer unsupported, ambiguous, or require guessing, the sentence is necessary.

{SELECTOR_FEW_SHOTS}

{qa_block}

{SELECTOR_REQUIREMENTS}

Return ONLY a JSON object:
{{
  "assessments": [
    {{
      "qa_id": 1,
      "section_sufficient": "yes",
      "selected_evidence_sentences": [1, 2],
      "answer_directly_found": "yes",
      "reasoning_level": "simple",
      "evidence_reason": "short explanation"
    }}
  ]
}}"""


def parse_selector_response(raw, records):
    data = _extract_json_object(raw)
    if not isinstance(data, dict):
        return None
    assessments = data.get("assessments")
    if not isinstance(assessments, list) or len(assessments) != len(records):
        return None
    return assessments


def build_blind_verifier_prompt(question, answer, evidence_items):
    """Build BLIND VERIFIER prompt with frozen text.

    Structure: {few_shot} / {context} / {QA} / {requirements}
    """
    if evidence_items:
        context_block = "\n".join(
            f"  [S{sid}] {sentence}" for sid, sentence in evidence_items
        )
    else:
        context_block = "  [NO EVIDENCE SENTENCES PROVIDED]"

    return f"""You are a blind evidence verifier for narrative reading comprehension.
Your task is to decide whether the provided evidence sentences alone are sufficient to justify the target answer.

A necessary evidence sentence is a sentence that must be available for a reader to justify the target answer. If removing the sentence would make the answer unsupported, ambiguous, or require guessing, the sentence is necessary.

{BLIND_VERIFIER_FEW_SHOTS}

Context:
{context_block}

QA:
Question: {question}
Target answer: {answer}

{BLIND_VERIFIER_REQUIREMENTS}

Return ONLY a JSON object:
{{
  "sufficient": "yes",
  "answer_directly_found": "yes",
  "reasoning_level": "simple",
  "reasoning": "one short explanation"
}}"""


def build_removal_verifier_prompt(question, answer, evidence_items):
    """Build REMOVAL VERIFIER prompt with frozen text.

    Structure: {few_shot} / {context} / {QA} / {requirements}
    """
    if evidence_items:
        context_block = "\n".join(
            f"  [S{sid}] {sentence}" for sid, sentence in evidence_items
        )
    else:
        context_block = "  [NO EVIDENCE SENTENCES PROVIDED]"

    return f"""You are a blind evidence verifier for narrative reading comprehension.
Your task is to decide whether the remaining evidence sentences are still sufficient to justify the target answer after one candidate evidence sentence has been removed.

A necessary evidence sentence is a sentence that must be available for a reader to justify the target answer. If removing the sentence would make the answer unsupported, ambiguous, or require guessing, the sentence is necessary.

{REMOVAL_VERIFIER_FEW_SHOTS}

Context:
{context_block}

QA:
Question: {question}
Target answer: {answer}

{REMOVAL_VERIFIER_REQUIREMENTS}

Return ONLY a JSON object:
{{
  "sufficient": "yes",
  "answer_directly_found": "yes",
  "reasoning_level": "simple",
  "reasoning": "one short explanation"
}}"""


def parse_sufficiency_response(raw):
    data = _extract_json_object(raw)
    if not isinstance(data, dict):
        return {
            "sufficient": "no",
            "answer_directly_found": "no",
            "reasoning_level": "unknown",
            "reasoning": "parse_failed",
            "parse_ok": False,
        }
    return {
        "sufficient": _normalize_yes_no(data.get("sufficient"), default="no"),
        "answer_directly_found": _normalize_yes_no(
            data.get("answer_directly_found"), default="no"
        ),
        "reasoning_level": _normalize_reasoning_level(
            data.get("reasoning_level"), default="unknown"
        ),
        "reasoning": data.get("reasoning", "") if isinstance(data.get("reasoning"), str) else "",
        "parse_ok": True,
    }


# ═══════════════════════════════════════════════════════════════════
# Auditor
# ═══════════════════════════════════════════════════════════════════

class NoVoteEvidenceAuditor:
    """Single-selector plus blind verifier with no self-consistency voting."""

    def __init__(self, batch_size=10, model=None, timeout=120,
                 selector_retries=1, use_removal_verifier=False,
                 blind_filter=False, selector_enable_thinking=None,
                 verifier_enable_thinking=None, thinking_budget=None):
        cfg = get_api_config()
        self.api_url = cfg["SILICONFLOW_API_URL"]
        self.api_key = cfg["SILICONFLOW_API_KEY"]
        self.model = model or cfg["JUDGE_MODEL"]
        self.batch_size = batch_size
        self.timeout = timeout
        self.selector_retries = selector_retries
        self.use_removal_verifier = use_removal_verifier
        self.blind_filter = blind_filter
        self.selector_enable_thinking = selector_enable_thinking
        self.verifier_enable_thinking = verifier_enable_thinking
        self.thinking_budget = thinking_budget
        self.selector_calls = 0
        self.sufficiency_calls = 0
        self.removal_calls = 0

    def select_batch(self, records):
        prompt = build_selector_prompt(records)
        raw = ""
        assessments = None
        parse_ok = False

        for _ in range(self.selector_retries + 1):
            self.selector_calls += 1
            try:
                raw = call_openai_compatible(
                    prompt,
                    api_url=self.api_url,
                    api_key=self.api_key,
                    model=self.model,
                    temperature=0.0,
                    max_tokens=6000,
                    json_mode=False,
                    system=(
                        "/think\n"
                        "You are an expert evidence selector. Return only JSON."
                    ),
                    enable_thinking=self.selector_enable_thinking,
                    thinking_budget=self.thinking_budget,
                    timeout=self.timeout,
                )
                assessments = parse_selector_response(raw, records)
                parse_ok = assessments is not None
                if parse_ok:
                    break
            except Exception as exc:
                raw = f"ERROR: {exc}"

        results = []
        for idx, record in enumerate(records):
            sentences = _split_story_sentences(record.get("story_section", ""))
            split_issues = _check_split_anomalies(sentences)
            candidate = {
                "story_name": record.get("story_name", ""),
                "story_section": record.get("story_section", ""),
                "question": record.get("question", ""),
                "answer": record.get("answer1", ""),
                "answer1": record.get("answer1", ""),
                "answer2": record.get("answer2", ""),
                "local_or_sum": record.get("local_or_sum", ""),
                "attribute": record.get("attribute", ""),
                "ex_or_im": record.get("ex_or_im", ""),
                "split": record.get("split", ""),
                "num_sentences_in_section": len(sentences),
                "split_anomalies": split_issues,
                "needs_manual_check": bool(split_issues),
                "selector_raw": raw,
                "selector_parse_ok": parse_ok,
                "selector_model": self.model,
            }
            if assessments and idx < len(assessments):
                assessment = assessments[idx]
                selected = _safe_sentence_ids(
                    assessment.get("selected_evidence_sentences", []),
                    len(sentences),
                )
                section_sufficient = _normalize_yes_no(
                    assessment.get("section_sufficient"), default="no"
                )
                if section_sufficient == "yes" and not selected:
                    section_sufficient = "no"
                answer_directly_found = _normalize_yes_no(
                    assessment.get("answer_directly_found"), default="no"
                )
                candidate.update({
                    "section_sufficient": section_sufficient,
                    "selected_evidence_sentences": selected,
                    "num_selected_sentences": len(selected),
                    "answer_directly_found": answer_directly_found,
                    "reasoning_level": _normalize_reasoning_level(
                        assessment.get("reasoning_level"), default="unknown"
                    ),
                    "evidence_reason": assessment.get("evidence_reason", ""),
                    "selector_status": "ok",
                    "selector_difficulty": (
                        _difficulty_from_ids(selected, answer_directly_found)
                        if section_sufficient == "yes" else "Invalid"
                    ),
                })
            else:
                candidate.update({
                    "section_sufficient": "no",
                    "selected_evidence_sentences": [],
                    "num_selected_sentences": 0,
                    "answer_directly_found": "no",
                    "reasoning_level": "unknown",
                    "evidence_reason": "selector_parse_failed",
                    "selector_status": "parse_failed",
                    "selector_difficulty": "Invalid",
                })
            results.append(candidate)
        return results

    def _check_blind_sufficiency(self, question, answer, evidence_items):
        prompt = build_blind_verifier_prompt(question, answer, evidence_items)
        self.sufficiency_calls += 1
        try:
            raw = call_openai_compatible(
                prompt,
                api_url=self.api_url,
                api_key=self.api_key,
                model=self.model,
                temperature=0.0,
                max_tokens=300,
                json_mode=False,
                system=(
                    "/no_think\n"
                    "You are a blind evidence verifier. Return only JSON."
                ),
                enable_thinking=self.verifier_enable_thinking,
                timeout=self.timeout,
            )
        except Exception as exc:
            raw = f"ERROR: {exc}"
        parsed = parse_sufficiency_response(raw)
        parsed["raw"] = raw
        return parsed

    def _check_removal_sufficiency(self, question, answer, evidence_items):
        prompt = build_removal_verifier_prompt(question, answer, evidence_items)
        self.removal_calls += 1
        try:
            raw = call_openai_compatible(
                prompt,
                api_url=self.api_url,
                api_key=self.api_key,
                model=self.model,
                temperature=0.0,
                max_tokens=300,
                json_mode=False,
                system=(
                    "/no_think\n"
                    "You are a blind evidence verifier. Return only JSON."
                ),
                enable_thinking=self.verifier_enable_thinking,
                timeout=self.timeout,
            )
        except Exception as exc:
            raw = f"ERROR: {exc}"
        parsed = parse_sufficiency_response(raw)
        parsed["raw"] = raw
        return parsed

    def verify_candidate(self, candidate):
        sentences = _split_story_sentences(candidate.get("story_section", ""))
        selected = list(candidate.get("selected_evidence_sentences", []))
        answer = _answer_text(candidate)
        question = candidate.get("question", "")

        result = dict(candidate)
        result["source"] = "implicit"

        if candidate.get("section_sufficient") != "yes" or not selected:
            result.update({
                "annotation_priority": "discard",
                "evidence_set_sufficient": "no",
                "sufficiency_reason": "selector_invalid_or_empty",
                "removal_checks": [],
                "required_evidence_sentences": [],
                "num_required_sentences": 0,
                "final_answer_directly_found": "no",
                "final_reasoning_level": "unknown",
                "suggested_difficulty_label": "Invalid",
                "difficulty_label": "Invalid",
                "verification_status": "selector_invalid_or_empty",
            })
            return result

        evidence_items = [
            (sid, sentences[sid]) for sid in selected if 0 <= sid < len(sentences)
        ]
        suff = self._check_blind_sufficiency(question, answer, evidence_items)
        result.update({
            "evidence_set_sufficient": suff["sufficient"],
            "verified_answer_directly_found": suff["answer_directly_found"],
            "verified_reasoning_level": suff["reasoning_level"],
            "sufficiency_reason": suff["reasoning"],
            "sufficiency_parse_ok": suff["parse_ok"],
            "sufficiency_raw": suff["raw"],
        })

        if suff["sufficient"] != "yes" and self.blind_filter:
            result.update({
                "annotation_priority": "discard",
                "removal_checks": [],
                "required_evidence_sentences": [],
                "num_required_sentences": 0,
                "final_answer_directly_found": "no",
                "final_reasoning_level": "unknown",
                "suggested_difficulty_label": "Invalid",
                "difficulty_label": "Invalid",
                "verification_status": "insufficient_selected_evidence",
            })
            return result

        if not self.use_removal_verifier:
            if suff["sufficient"] == "yes":
                current_direct = suff["answer_directly_found"]
                current_reasoning_level = suff["reasoning_level"]
                annotation_priority = "high"
                verification_status = "blind_sufficient"
            else:
                current_direct = candidate.get("answer_directly_found", "no")
                current_reasoning_level = candidate.get(
                    "reasoning_level", "unknown"
                )
                annotation_priority = "repair"
                verification_status = "needs_human_repair"

            difficulty = _difficulty_from_ids(selected, current_direct)
            result.update({
                "annotation_priority": annotation_priority,
                "removal_checks": [],
                "required_evidence_sentences": selected,
                "num_required_sentences": len(selected),
                "final_answer_directly_found": current_direct,
                "final_reasoning_level": current_reasoning_level,
                "suggested_difficulty_label": difficulty,
                "difficulty_label": difficulty,
                "verification_status": verification_status,
            })
            return result

        current = list(selected)
        current_direct = suff["answer_directly_found"]
        current_reasoning_level = suff["reasoning_level"]
        removal_checks = []
        for sid in selected:
            if sid not in current:
                continue
            remaining = [x for x in current if x != sid]
            if not remaining:
                removal_checks.append({
                    "sentence_id": sid,
                    "remaining_sentence_ids": [],
                    "can_still_answer": "no",
                    "answer_directly_found": "no",
                    "reasoning_level": "unknown",
                    "reasoning": "no_evidence_remaining",
                    "parse_ok": True,
                    "raw": "",
                    "decision": "keep",
                })
                continue
            remaining_items = [
                (rid, sentences[rid])
                for rid in remaining if 0 <= rid < len(sentences)
            ]
            removal = self._check_removal_sufficiency(
                question, answer, remaining_items
            )
            can_still = removal["sufficient"]
            decision = "drop" if can_still == "yes" else "keep"
            if decision == "drop":
                current = remaining
                current_direct = removal["answer_directly_found"]
                current_reasoning_level = removal["reasoning_level"]
            removal_checks.append({
                "sentence_id": sid,
                "remaining_sentence_ids": remaining,
                "can_still_answer": can_still,
                "answer_directly_found": removal["answer_directly_found"],
                "reasoning_level": removal["reasoning_level"],
                "reasoning": removal["reasoning"],
                "parse_ok": removal["parse_ok"],
                "raw": removal["raw"],
                "decision": decision,
            })

        difficulty = _difficulty_from_ids(current, current_direct)
        result.update({
            "annotation_priority": "high",
            "removal_checks": removal_checks,
            "required_evidence_sentences": current,
            "num_required_sentences": len(current),
            "final_answer_directly_found": current_direct,
            "final_reasoning_level": current_reasoning_level,
            "suggested_difficulty_label": difficulty,
            "difficulty_label": difficulty,
            "verification_status": "ok" if difficulty != "Invalid" else "empty_after_removal",
        })
        return result

    def call_counts(self):
        return {
            "selector_calls": self.selector_calls,
            "sufficiency_calls": self.sufficiency_calls,
            "removal_calls": self.removal_calls,
            "total_calls": (
                self.selector_calls + self.sufficiency_calls + self.removal_calls
            ),
        }
