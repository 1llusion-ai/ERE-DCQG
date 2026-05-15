# Answer-Grounded Evidence Planning Audit

Generated: 2026-05-14 12:30:20

## 1. Run Summary

| Metric | Value |
|---|---|
| Total candidates | 318 |
| Model | Qwen/Qwen2.5-32B-Instruct |
| Parse OK | 302/318 (95.0%) |
| Evidence plan valid | 297/318 (93.4%) |
| Plans with contradictions | 98/318 (30.8%) |
| Total contradiction count | 245 |

## 2. Prompt Leakage Audit

| Metric | Value |
|---|---|
| original_question_present_in_prompt=True | 0 |
| Status | **PASS** |

## 3. Parse and Validity Stats

| Difficulty | Total | Parse OK | Pct | Plan Valid | Pct |
|---|---:|---:|---:|---:|---:|
| Easy | 106 | 100 | 94.3% | 95 | 89.6% |
| Medium | 106 | 101 | 95.3% | 101 | 95.3% |
| Hard | 106 | 101 | 95.3% | 101 | 95.3% |

## 4. Evidence Plan Feasibility by Target Difficulty

| Difficulty | yes | partial | no | ? |
|---|---:|---:|---:|---:|
| Easy | 95 | 0 | 5 | 0 |
| Medium | 98 | 3 | 0 | 0 |
| Hard | 100 | 1 | 0 | 0 |

## 5. Required Evidence Count Distribution

| Count | Easy | Medium | Hard |
|---|---:|---:|---:|
| 0 | 6 | 5 | 5 |
| 1 | 100 | 4 | 0 |
| 2 | 0 | 97 | 2 |
| 3 | 0 | 0 | 84 |
| 4 | 0 | 0 | 10 |
| 5 | 0 | 0 | 2 |
| 6 | 0 | 0 | 1 |
| 8 | 0 | 0 | 1 |
| 14 | 0 | 0 | 1 |

## 6. Easy Diagnostics

| Metric | Count | Pct | Target |
|---|---:|---:|---:|
| num_req=1 | 100/106 | 94.3% | >=70% |
| ASA=yes | 95/106 | 89.6% | >=70% |

### Infeasible Easy Examples

| # | Story | Answer | Num Req | ASA | Feasible | Reason |
|---|---|---|---|---|---|
| 1 | lasse-my-thrall | read the words outloud . | 1 | no | no | The answer sentence alone is not sufficient, and a bridge se |
| 2 | story-of-princess-hase | angry . | 1 | no | no | The story section does not contain any evidence that the cha |
| 3 | tale-of-johnny-town-mouse | scared . | 1 | no | no | The story section does not contain evidence that Timmy Willi |
| 4 | the-elfin-knight | it went well for a few hours . | 1 | no | no | The target answer 'it went well for a few hours' is not dire |
| 5 | the-enchanted-moccasins | he placed the moccasins near t | 1 | no | no | The story section does not contain a sentence that directly  |

## 7. Medium Diagnostics

| Metric | Count | Pct | Target |
|---|---:|---:|---:|
| num_req=2 | 97/106 | 91.5% | >=50% |
| necessity one_relation/answer_local | 91/106 | 85.8% | — |

### Medium Necessity Types

| Type | Count |
|---|---:|
| one_relation | 72 |
| answer_local | 19 |
| causal_bridge | 8 |
| None | 5 |
| motivation_bridge | 1 |
| disambiguation | 1 |

## 8. Hard Diagnostics

| Metric | Count | Pct | Target |
|---|---:|---:|---:|
| num_req>=3 | 99/106 | 93.4% | >=45% |
| bridge_required=yes | 100/106 | 94.3% | >=45% |
| causal/motivation/summary | 101/106 | 95.3% | — |
| feasible yes/partial | 101/106 | 95.3% | >=60% |

### Hard Reasoning Operations

| Operation | Count |
|---|---:|
| causal_chain | 63 |
| motivation_chain | 31 |
| summary_synthesis | 7 |
| None | 5 |

## 9. Comparison to Old Question-Conditioned Evidence

| Metric | Value |
|---|---|
| Answer sentence ID match (in old required) | 236/318 (74.2%) |
| Avg Jaccard (required evidence) | 0.619 |
| Avg Jaccard (bridge) | 0.151 |

### Required Evidence Jaccard Distribution

| Range | Count |
|---|---:|
| 0.0 (disjoint) | 44 |
| 0.01-0.29 | 20 |
| 0.30-0.49 | 48 |
| 0.50-0.69 | 43 |
| 0.70-0.99 | 8 |
| 1.0 (identical) | 138 |

## 10. Examples per Difficulty

### Easy

**Example 1: Snow-man**

- Target answer: "a young girl and a young man ."
- Original question (not in prompt): "who came into the garden ?"
- Answer sentence: S0
- Required evidence: [0]
- Anchor: []
- Bridge: []
- ASA: yes
- Bridge required: no
- Reasoning: explicit_lookup
- Necessity: answer_local
- Feasible: yes
- Reason: The answer is directly stated in the first sentence, making it suitable for an easy difficulty level.

**Example 2: a-lost-paradise**

- Target answer: "they wanted plenty of food to eat ."
- Original question (not in prompt): "why did the couple want to find a job ?"
- Answer sentence: S13
- Required evidence: [13]
- Anchor: []
- Bridge: []
- ASA: yes
- Bridge required: no
- Reasoning: explicit_lookup
- Necessity: answer_local
- Feasible: yes
- Reason: The answer sentence directly states the desire for plenty of food to eat.

**Example 3: bokwewa-the-humpback**

- Target answer: "sad ."
- Original question (not in prompt): "how will the wife feel about kwasynd abandoning his quest to find her ?"
- Answer sentence: S4
- Required evidence: [4]
- Anchor: [3]
- Bridge: []
- ASA: yes
- Bridge required: no
- Reasoning: local_inference
- Necessity: answer_local
- Feasible: yes
- Reason: The answer sentence alone is sufficient to infer the target answer.

**Example 4: brave-tin-soldier**

- Target answer: "they ran out of melted tin ."
- Original question (not in prompt): "why did one of the tin soldiers only have one leg ?"
- Answer sentence: S5
- Required evidence: [5]
- Anchor: [4]
- Bridge: []
- ASA: yes
- Bridge required: no
- Reasoning: local_inference
- Necessity: answer_local
- Feasible: yes
- Reason: The answer is directly implied in sentence [S5] without needing additional sentences.

**Example 5: brother-sister**

- Target answer: "their stepmother was mean to them ."
- Original question (not in prompt): "why did the two decide to leave ?"
- Answer sentence: S0
- Required evidence: [0]
- Anchor: []
- Bridge: []
- ASA: yes
- Bridge required: no
- Reasoning: explicit_lookup
- Necessity: answer_local
- Feasible: yes
- Reason: The answer is directly stated in the first sentence, making it sufficient for an easy difficulty level.

### Medium

**Example 1: Snow-man**

- Target answer: "move away from that place ."
- Original question (not in prompt): "what did the snow man wish he could do ?"
- Answer sentence: S3
- Required evidence: [3, 4]
- Anchor: []
- Bridge: [4]
- ASA: partial
- Bridge required: yes
- Reasoning: local_inference
- Necessity: one_relation
- Feasible: yes
- Reason: Sentence [S3] directly mentions the desire to move, while [S4] provides context on the action of moving on ice, supporting the answer.

**Example 2: a-legend-of-knockmany**

- Target answer: "he wanted to give finn a considerable beating ."
- Original question (not in prompt): "why did far rua swear he would never rest ?"
- Answer sentence: S10
- Required evidence: [10, 11]
- Anchor: []
- Bridge: []
- ASA: partial
- Bridge required: no
- Reasoning: local_inference
- Necessity: one_relation
- Feasible: yes
- Reason: Sentence S10 provides the direct evidence, while S11 adds the motivation, making it a medium difficulty question.

**Example 3: a-lost-paradise**

- Target answer: "they could not find work ."
- Original question (not in prompt): "why was the couple unhappy ?"
- Answer sentence: S3
- Required evidence: [3, 4]
- Anchor: [2]
- Bridge: [4]
- ASA: partial
- Bridge required: yes
- Reasoning: local_inference
- Necessity: one_relation
- Feasible: yes
- Reason: Sentence S3 provides the direct answer that they grew poorer, and S4 provides the context of frequent hunger, indicating they could not find work.

**Example 4: bokwewa-the-humpback**

- Target answer: "kwasynd did not save her ."
- Original question (not in prompt): "why did the young woman pay no attention to kwasynd ?"
- Answer sentence: S1
- Required evidence: [1, 2]
- Anchor: [0]
- Bridge: [0]
- ASA: partial
- Bridge required: yes
- Reasoning: local_inference
- Necessity: one_relation
- Feasible: yes
- Reason: Sentence S1 provides the context that the woman pays no heed to kwasynd, and S2 shows the immediate action of her sitting by kwasynd, implying he did not save her.

**Example 5: brave-tin-soldier**

- Target answer: "the fish that swallowed him was caught and sold ."
- Original question (not in prompt): "how did the soldier return home ?"
- Answer sentence: S5
- Required evidence: [4, 5]
- Anchor: [3]
- Bridge: [4]
- ASA: partial
- Bridge required: yes
- Reasoning: local_inference
- Necessity: one_relation
- Feasible: yes
- Reason: Sentence [S4] provides the context that the fish was caught, which is necessary to understand the event described in [S5].

### Hard

**Example 1: Snow-man**

- Target answer: "he had to ."
- Original question (not in prompt): "why did the yard-dog give up such a comfortable place ?"
- Answer sentence: S16
- Required evidence: [13, 14, 15, 16]
- Anchor: [13, 14]
- Bridge: [15]
- ASA: no
- Bridge required: yes
- Reasoning: causal_chain
- Necessity: causal_bridge
- Feasible: yes
- Reason: The plan requires understanding the causal chain of events leading to the dog's action, which matches the hard difficulty.

**Example 2: a-legend-of-knockmany**

- Target answer: "happy ."
- Original question (not in prompt): "how will oonagh feel when finn comes home ?"
- Answer sentence: S7
- Required evidence: [5, 7, 8]
- Anchor: [4]
- Bridge: [5]
- ASA: no
- Bridge required: yes
- Reasoning: motivation_chain
- Necessity: motivation_bridge
- Feasible: yes
- Reason: The plan requires understanding Finn's sudden affection for his wife as a motivation for his return, which is a hard-level reasoning task.

**Example 3: bokwewa-the-humpback**

- Target answer: "she was dead ."
- Original question (not in prompt): "why did kwasynd ask bokwewa to restore the beautiful young woman ?"
- Answer sentence: S9
- Required evidence: [3, 9, 10]
- Anchor: [3]
- Bridge: []
- ASA: no
- Bridge required: yes
- Reasoning: causal_chain
- Necessity: causal_bridge
- Feasible: yes
- Reason: The plan requires understanding the context of the dead woman and the brother's attempt to restore her, forming a causal chain.

**Example 4: brave-tin-soldier**

- Target answer: "she fluttered into the stove and burned ."
- Original question (not in prompt): "what happened to the little dancer ?"
- Answer sentence: S0
- Required evidence: [0, 1, 2]
- Anchor: []
- Bridge: [1]
- ASA: no
- Bridge required: yes
- Reasoning: causal_chain
- Necessity: causal_bridge
- Feasible: yes
- Reason: The plan requires understanding the initial event, the outcome of the tin soldier, and the final state of the dancer to fully grasp the causal chain.

**Example 5: canonbie-dick-and-thomas-of-ercildoune**

- Target answer: "the stranger wanted to reveal his identity ."
- Original question (not in prompt): "why did the stranger ask canonbie dick if he knew thomas the rhymer ?"
- Answer sentence: S5
- Required evidence: [5, 6, 7]
- Anchor: [1, 2, 3]
- Bridge: [4]
- ASA: no
- Bridge required: yes
- Reasoning: motivation_chain
- Necessity: motivation_bridge
- Feasible: yes
- Reason: The plan requires understanding the stranger's identity and his motivation for revealing it, which spans multiple sentences.

## 11. Success Criteria Summary

| Criterion | Actual | Target | Status |
|---|---:|---:|---|
| original_question_present_in_prompt=0 | 0/318 | 0/318 | **PASS** |
| parse_ok >= 95% | 95.0% | >=95% | **FAIL** |
| evidence_plan_valid >= 85% | 93.4% | >=85% | **PASS** |
| Easy len=1 >= 70% | 94.3% | >=70% | **PASS** |
| Easy ASA=yes >= 70% | 89.6% | >=70% | **PASS** |
| Medium len=2 >= 50% | 91.5% | >=50% | **PASS** |
| Hard len>=3 >= 45% | 93.4% | >=45% | **PASS** |
| Hard bridge_required >= 45% | 94.3% | >=45% | **PASS** |
| Hard feasible yes/partial >= 60% | 95.3% | >=60% | **PASS** |

