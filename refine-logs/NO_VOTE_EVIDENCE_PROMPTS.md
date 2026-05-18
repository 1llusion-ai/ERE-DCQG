# No-Vote Evidence Audit — Frozen Prompt Snapshot

Source: `dcqg/path/no_vote_evidence.py`
Date: 2026-05-16

Pipeline:

1. Selector: full-context evidence selection.
2. Blind Verifier: selected-evidence-only sufficiency check.
3. Removal Verifier: leave-one-out sufficiency check with one candidate sentence removed.

Difficulty labels are NOT provided to the Selector or Verifiers. Labels are assigned
by `_difficulty_from_ids()` after evidence selection:

- Easy   = answer_directly_found == "yes" and num_required_sentences == 1
- Medium = answer_directly_found == "no"  and num_required_sentences == 1
         OR answer_directly_found == "yes" and num_required_sentences >= 2
- Hard   = answer_directly_found == "no"  and num_required_sentences >= 2
- Invalid = section_sufficient == "no", or no evidence selected, or parse failure

---

## PROMPT 1: SELECTOR

```text
You are an expert evidence selector for narrative reading comprehension.
Your task is to select the smallest complete set of necessary evidence sentences that justifies the target answer.

A necessary evidence sentence is a sentence that must be available for a reader to justify the target answer. If removing the sentence would make the answer unsupported, ambiguous, or require guessing, the sentence is necessary.

{few_shot}

{context}

{QA}

{requirements}

Return ONLY a JSON object:
{
  "assessments": [
    {
      "qa_id": 1,
      "section_sufficient": "yes",
      "selected_evidence_sentences": [1, 2],
      "answer_directly_found": "yes",
      "reasoning_level": "simple",
      "evidence_reason": "short explanation"
    }
  ]
}
```

### SELECTOR FEW-SHOT

```text
Few-shot examples:

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
{"section_sufficient":"no","selected_evidence_sentences":[],"answer_directly_found":"no","reasoning_level":"unknown","evidence_reason":"The context suggests an opportunity but does not contain enough evidence that Silverwhite will save the princess."}
```

### SELECTOR REQUIREMENTS

```text
Requirements:
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
8. Return only valid JSON.
```

---

## PROMPT 2: BLIND VERIFIER

```text
You are a blind evidence verifier for narrative reading comprehension.
Your task is to decide whether the provided evidence sentences alone are sufficient to justify the target answer.

A necessary evidence sentence is a sentence that must be available for a reader to justify the target answer. If removing the sentence would make the answer unsupported, ambiguous, or require guessing, the sentence is necessary.

{few_shot}

{context}

{QA}

{requirements}

Return ONLY a JSON object:
{
  "sufficient": "yes",
  "answer_directly_found": "yes",
  "reasoning_level": "simple",
  "reasoning": "one short explanation"
}
```

### BLIND VERIFIER FEW-SHOT

```text
Few-shot examples:

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
{"sufficient":"no","answer_directly_found":"no","reasoning_level":"unknown","reasoning":"S2 says they did not know this, but without the sentence explaining that the ring was lucky, the target answer is unsupported."}
```

### BLIND VERIFIER REQUIREMENTS

```text
Requirements:
1. Use only the evidence sentences in Context.
2. The full story is not available.
3. Treat the target answer as a claim to verify, not as evidence.
4. If the evidence is enough to justify the target answer, output sufficient="yes".
5. If key information is missing, ambiguous, or only guessable, output sufficient="no".
6. If sufficient="yes", judge answer_directly_found and reasoning_level using only the provided evidence.
7. If sufficient="no", set answer_directly_found="no" and reasoning_level="unknown".
8. Return only valid JSON.
```

---

## PROMPT 3: REMOVAL VERIFIER

```text
You are a blind evidence verifier for narrative reading comprehension.
Your task is to decide whether the remaining evidence sentences are still sufficient to justify the target answer after one candidate evidence sentence has been removed.

A necessary evidence sentence is a sentence that must be available for a reader to justify the target answer. If removing the sentence would make the answer unsupported, ambiguous, or require guessing, the sentence is necessary.

{few_shot}

{context}

{QA}

{requirements}

Return ONLY a JSON object:
{
  "sufficient": "yes",
  "answer_directly_found": "yes",
  "reasoning_level": "simple",
  "reasoning": "one short explanation"
}
```

### REMOVAL VERIFIER FEW-SHOT

```text
Few-shot examples:

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
{"sufficient":"yes","answer_directly_found":"no","reasoning_level":"simple","reasoning":"The remaining sentence still supports the simple inference that Tom was frightened."}
```

### REMOVAL VERIFIER REQUIREMENTS

```text
Requirements:
1. Use only the remaining evidence sentences in Context.
2. The full story is not available.
3. The removed sentence is not available.
4. Treat the target answer as a claim to verify, not as evidence.
5. If the remaining evidence is still enough to justify the target answer, output sufficient="yes".
6. If key information is missing, ambiguous, or only guessable, output sufficient="no".
7. If sufficient="yes", judge answer_directly_found and reasoning_level using only the remaining evidence.
8. If sufficient="no", set answer_directly_found="no" and reasoning_level="unknown".
9. Return only valid JSON.
```

---

## Label Mapping

```
Easy   = answer_directly_found == "yes" and num_required_sentences == 1
Medium = answer_directly_found == "no"  and num_required_sentences == 1
       OR answer_directly_found == "yes" and num_required_sentences >= 2
Hard   = answer_directly_found == "no"  and num_required_sentences >= 2
```

Implemented in `_difficulty_from_ids()` in `dcqg/path/no_vote_evidence.py`.
