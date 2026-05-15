# FairytaleQA Hard QG Pilot Report

Generated: 2026-05-11 14:51:34

## 1. Run Summary

| Field | Value |
|---|---|
| Methods | Direct, ICL, SelfRefine, Ours |
| Requested limit | 71 |
| Graph total | 75 |
| Graph valid | 71 |
| Selected candidates | 71 |
| Total generations | 284 |
| Target difficulty | Hard |

### Parse success by method

| Method | parse_ok | Total | Pct |
|---|---:|---:|---:|
| Direct | 60 | 71 | 84.5% |
| ICL | 68 | 71 | 95.8% |
| SelfRefine | 42 | 71 | 59.2% |
| Ours | 66 | 71 | 93.0% |

### 1b. Generation Robustness by Method

| Method | degenerate | repair_attempted | repair_success | quality_pass |
|---|---:|---:|---:|---:|
| Direct | 11 | 0 | 0 | 37 |
| ICL | 4 | 0 | 0 | 47 |
| SelfRefine | 0 | 0 | 0 | 25 |
| Ours | 20 | 37 | 6 | 50 |

## 2. Quality Pass by Method

| Method | quality_pass | strict_quality_pass | Total | Pct (loose) | Pct (strict) |
|---|---:|---:|---:|---:|---:|
| Direct | 37 | 5 | 71 | 52.1% | 7.0% |
| ICL | 47 | 5 | 71 | 66.2% | 7.0% |
| SelfRefine | 25 | 2 | 71 | 35.2% | 2.8% |
| Ours | 50 | 8 | 71 | 70.4% | 11.3% |

## 3. Blind Difficulty Distribution (quality-pass only)

| Method | Easy | Medium | Hard | JudgeError | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 4 | 32 | 1 | 0 | 37 |
| ICL | 5 | 38 | 4 | 0 | 47 |
| SelfRefine | 1 | 22 | 2 | 0 | 25 |
| Ours | 1 | 41 | 8 | 0 | 50 |

### 3b. Difficulty Judge Status by Method

| Method | judge_ok | judge_error | Total | Error rate |
|---|---:|---:|---:|---:|
| Direct | 37 | 0 | 37 | 0.0% |
| ICL | 47 | 0 | 47 | 0.0% |
| SelfRefine | 25 | 0 | 25 | 0.0% |
| Ours | 50 | 0 | 50 | 0.0% |

## 4. Hard Hit Rate by Method

Denominator: quality-pass AND difficulty_judge_status=ok

| Method | Hard hit | Wilson 95% CI |
|---|---|---|
| Direct | 1/37 (2.7%, 95%CI [0.5, 13.8%]) | |
| ICL | 4/47 (8.5%, 95%CI [3.4, 19.9%]) | |
| SelfRefine | 2/25 (8.0%, 95%CI [2.2, 25.0%]) | |
| Ours | 8/50 (16.0%, 95%CI [8.3, 28.5%]) | |

## 5. Evidence Dependency by Method (quality-pass, judge-ok only)

| Method | alone_sufficient=no | bridge_required=yes | removal in {ambiguous,unanswerable} | Total |
|---|---:|---:|---:|---:|
| Direct | 14 | 33 | 29 | 37 |
| ICL | 18 | 43 | 36 | 47 |
| SelfRefine | 7 | 24 | 20 | 25 |
| Ours | 22 | 49 | 40 | 50 |

### 5b. Target Evidence Coverage by Method (quality-pass, judge-ok only)

| Method | mean coverage | coverage>=0.67 | uses_all_target | Total |
|---|---:|---:|---:|---:|
| Direct | 0.351 | 0 | 0 | 37 |
| ICL | 0.388 | 1 | 1 | 47 |
| SelfRefine | 0.404 | 0 | 0 | 25 |
| Ours | 0.510 | 3 | 3 | 50 |

### 5c. Hard Realization Pass by Method (exact-id diagnostic)

Hard realization (legacy) = judge_ok AND num_judge_used>=3 AND uses_bridge in {yes,partial} AND coverage>=0.67 AND predicted=Hard

| Method | hard_realization_pass | quality-pass judge-ok | Rate |
|---|---:|---:|---:|
| Direct | 0 | 37 | 0.0% |
| ICL | 1 | 47 | 2.1% |
| SelfRefine | 0 | 25 | 0.0% |
| Ours | 3 | 50 | 6.0% |

### 5e. Hard Realization Pass v2 by Method

Denominator: quality-pass AND difficulty_judge_status=ok

hrp_v2 = predicted=Hard AND num_judge_used>=3 AND bridge_required=yes AND alone_sufficient=no AND semantic_evidence_match in {yes,partial}

| Method | hrp_v2 | Wilson 95% CI |
|---|---|---|
| Direct | 1/37 (2.7%, 95%CI [0.5, 13.8%]) | |
| ICL | 4/47 (8.5%, 95%CI [3.4, 19.9%]) | |
| SelfRefine | 2/25 (8.0%, 95%CI [2.2, 25.0%]) | |
| Ours | 8/50 (16.0%, 95%CI [8.3, 28.5%]) | |

### 5e2. Strict HRP-v2 by Method

strict_hrp_v2 = hard_realization_pass_v2=yes AND strict_quality_pass=true AND focus_match=yes

| Method | strict_hrp_v2 | Wilson 95% CI |
|---|---|---|
| Direct | 0/37 (0.0%, 95%CI [0.0, 9.4%]) | |
| ICL | 0/47 (0.0%, 95%CI [0.0, 7.6%]) | |
| SelfRefine | 0/25 (0.0%, 95%CI [0.0, 13.3%]) | |
| Ours | 0/50 (0.0%, 95%CI [0.0, 7.1%]) | |

### 5f. Semantic Evidence Match by Method (quality-pass, judge-ok)

| Method | yes | partial | no | judge_error | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 0 | 35 | 2 | 0 | 37 |
| ICL | 2 | 41 | 4 | 0 | 47 |
| SelfRefine | 0 | 25 | 0 | 0 | 25 |
| Ours | 4 | 45 | 1 | 0 | 50 |

### 5d. Answer Focus Diagnostics (Ours)

#### Question focus distribution

| Focus | Count | Pct |
|---|---:|---:|
| bridge | 23 | 46.0% |
| state | 12 | 24.0% |
| motivation | 9 | 18.0% |
| outcome | 6 | 12.0% |

#### Focus match rate

- focus_match=yes: 39 / 50
- focus_match=no: 11 / 50

#### Answer role -> question focus mapping

| answer_role | question_focus | count |
|---|---|---:|
| answer | bridge | 19 |
| answer | state | 12 |
| answer | motivation | 6 |
| answer | outcome | 4 |
| answer_bridge | bridge | 4 |
| answer_bridge | motivation | 3 |
| answer_bridge | outcome | 2 |

#### Focus mismatch examples

**Mismatch 1:**
- Story: lame-dog
- Question: What was the final consequence of the youngest princess's response to her sisters' question about wishing for a husband?
- Target answer: she did not care who she married .
- answer_role=answer, question_focus=outcome

**Mismatch 2:**
- Story: thomas-the-rhymer
- Question: Why was Thomas eventually able to eat the fruit despite the warning?
- Target answer: he could eat this fruit .
- answer_role=answer, question_focus=outcome

**Mismatch 3:**
- Story: evil-one-kitta-grau
- Question: How did the merchant come to be in a position where he could not sell Kitta Grau to anyone for three weeks?
- Target answer: the merchant was unable to sell kitta grau .
- answer_role=answer, question_focus=state

#### Per question_focus metrics (all Ours)

| Focus | N | quality_pass | Hard | hrp_v2 | strict_hrp_v2 | focus_match=yes |
|---|---:|---:|---:|---:|---:|---:|
| bridge | 32 | 23 | 5 | 5 | 0 | 22 |
| count | 1 | 0 | 0 | 0 | 0 | 1 |
| motivation | 16 | 9 | 1 | 1 | 0 | 9 |
| outcome | 7 | 6 | 2 | 2 | 0 | 4 |
| state | 15 | 12 | 0 | 0 | 0 | 10 |

#### Per answer_role metrics (all Ours)

| answer_role | N | quality_pass | Hard | hrp_v2 | strict_hrp_v2 |
|---|---:|---:|---:|---:|---:|
| answer | 59 | 41 | 6 | 6 | 0 |
| answer_bridge | 11 | 9 | 2 | 2 | 0 |
| count_pattern | 1 | 0 | 0 | 0 | 0 |

#### Per answer_node_type metrics (all Ours)

| node_type | N | quality_pass | Hard | hrp_v2 | strict_hrp_v2 |
|---|---:|---:|---:|---:|---:|
| action | 15 | 11 | 0 | 0 | 0 |
| belief | 3 | 3 | 0 | 0 | 0 |
| count | 1 | 0 | 0 | 0 | 0 |
| description | 15 | 11 | 4 | 4 | 0 |
| emotion | 5 | 4 | 1 | 1 | 0 |
| goal | 1 | 0 | 0 | 0 | 0 |
| motivation | 10 | 5 | 0 | 0 | 0 |
| outcome | 7 | 6 | 2 | 2 | 0 |
| problem | 1 | 1 | 1 | 1 | 0 |
| resolution | 1 | 1 | 0 | 0 | 0 |
| state | 12 | 8 | 0 | 0 | 0 |

### 5g. Unique Story Diversity and Cluster Diagnostic

#### Unique stories among predicted Hard (quality-pass, judge-ok)

| Method | unique stories | Hard count | stories |
|---|---:|---:|---|
| Direct | 1 | 1 | how-princess-pride-was-broken |
| ICL | 4 | 4 | per-gynt, soria-moria-castle, three-dogs, youth-who-was-to-serve-three-years-without-pay |
| SelfRefine | 1 | 2 | youth-who-was-to-serve-three-years-without-pay |
| Ours | 6 | 8 | little-lasse, master-girl, the-black-bull-of-norroway, the-fairies-of-merlin-crag, the-three-crowns, youth-who-was-to-serve-three-years-without-pay |

#### Unique stories among hrp_v2 (quality-pass, judge-ok)

| Method | unique stories | hrp_v2 count | stories |
|---|---:|---:|---|
| Direct | 1 | 1 | how-princess-pride-was-broken |
| ICL | 4 | 4 | per-gynt, soria-moria-castle, three-dogs, youth-who-was-to-serve-three-years-without-pay |
| SelfRefine | 1 | 2 | youth-who-was-to-serve-three-years-without-pay |
| Ours | 6 | 8 | little-lasse, master-girl, the-black-bull-of-norroway, the-fairies-of-merlin-crag, the-three-crowns, youth-who-was-to-serve-three-years-without-pay |

#### Unique stories among strict_hrp_v2 (quality-pass, judge-ok)

| Method | unique stories | strict_hrp_v2 count | stories |
|---|---:|---:|---|
| Direct | 0 | 0 |  |
| ICL | 0 | 0 |  |
| SelfRefine | 0 | 0 |  |
| Ours | 0 | 0 |  |

#### Cluster diagnostic: three-dogs concentration

| Method | three-dogs in Hard | total Hard | three-dogs in hrp_v2 | total hrp_v2 | three-dogs in strict_hrp_v2 | total strict_hrp_v2 |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 0 | 1 | 0 | 1 | 0 | 0 |
| ICL | 1 | 4 | 1 | 4 | 0 | 0 |
| SelfRefine | 0 | 2 | 0 | 2 | 0 | 0 |
| Ours | 0 | 8 | 0 | 8 | 0 | 0 |

## 6. Failure Reasons by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | not answerable | 21 |
| Direct | wrong answer | 9 |
| Direct | not fluent | 2 |
| Direct | other | 2 |
| ICL | not answerable | 11 |
| ICL | wrong answer | 10 |
| ICL | not fluent | 3 |
| SelfRefine | not answerable | 33 |
| SelfRefine | wrong answer | 7 |
| SelfRefine | other | 4 |
| SelfRefine | not fluent | 2 |
| Ours | not answerable | 9 |
| Ours | not fluent | 5 |
| Ours | wrong answer | 5 |
| Ours | gen error: degenerate output | 2 |

### 6b. Difficulty Judge Parse Failures

None.

## 7. Copy/Reference Diagnostics

| Method | Total | Copies source | Copy rate |
|---|---:|---:|---:|
| Direct | 71 | 0 | 0.0% |
| ICL | 71 | 0 | 0.0% |
| SelfRefine | 71 | 0 | 0.0% |
| Ours | 71 | 0 | 0.0% |

## 8. Examples

### Best Ours examples (quality-pass, predicted Hard)

**Example 1:**
- Story: master-girl
- Question: What was the king's son's state of mind as he decided to try the soup in the third room after his previous experiences?
- Target answer: intrigued .
- Quality: answerable=partial, asks_expected=partial, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 0.000, hard_realization=no, hrp_v2=yes
- Focus: answer_role=answer, question_focus=bridge, focus_match=yes
- Semantic match: partial - The judge-identified sentences [10, 12, 13] cover the king's son's actions and reactions to the third kettle, but they miss the earlier thought process and actions that led up to his decision to try the soup, which are covered in the target evidence sentences [2, 4, 6, 9, 11].

**Example 2:**
- Story: little-lasse
- Question: What development connected the situation where the oars were locked up to the difficulty of rowing to Asia?
- Target answer: he did not have oars to row with .
- Quality: answerable=yes, asks_expected=partial, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 1.000, hard_realization=yes, hrp_v2=yes
- Focus: answer_role=answer, question_focus=bridge, focus_match=yes
- Semantic match: yes - The judge-identified sentences [6, 7, 8] cover the same reasoning chain as the target evidence sentences, explaining the lack of oars and the resulting difficulty in rowing to Asia.

**Example 3:**
- Story: youth-who-was-to-serve-three-years-without-pay
- Question: What pivotal moment connected the youth's decision to speak with the prince and the prince's advice about choosing the ring?
- Target answer: the prince knew what was most valuable .
- Quality: answerable=yes, asks_expected=partial, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 0.250, hard_realization=no, hrp_v2=yes
- Focus: answer_role=answer, question_focus=bridge, focus_match=yes
- Semantic match: partial - The judge-identified sentences cover the advice about choosing the ring but miss the youth's decision to speak with the prince and the dialogue leading up to the advice, which are crucial for understanding the full reasoning chain.

### Hard realization pass v2 examples

**HRP-v2 Example 1 (ICL):**
- Story: three-dogs
- Question: Why did the youth agree to exchange his pigs for the old man's dogs each time they met?
- Target answer: three times .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial - The judge-identified sentences cover the final exchange but miss the earlier exchanges, which are key to understanding the repeated pattern of the youth's decision-making.

**HRP-v2 Example 2 (Direct):**
- Story: how-princess-pride-was-broken
- Question: Why was the princess so upset about the price the lad asked for his musical box?
- Target answer: annoyed .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial - The judge-identified sentences [5, 6, 7] cover the princess's reaction to the price and her refusal but miss the initial expression of her internal conflict in sentence [4], which is part of the target evidence.

**HRP-v2 Example 3 (Ours):**
- Story: master-girl
- Question: What was the king's son's state of mind as he decided to try the soup in the third room after his previous experiences?
- Target answer: intrigued .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial - The judge-identified sentences [10, 12, 13] cover the king's son's actions and reactions to the third kettle, but they miss the earlier thought process and actions that led up to his decision to try the soup, which are covered in the target evidence sentences [2, 4, 6, 9, 11].

**HRP-v2 Example 4 (Ours):**
- Story: little-lasse
- Question: What development connected the situation where the oars were locked up to the difficulty of rowing to Asia?
- Target answer: he did not have oars to row with .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: yes - The judge-identified sentences [6, 7, 8] cover the same reasoning chain as the target evidence sentences, explaining the lack of oars and the resulting difficulty in rowing to Asia.

**HRP-v2 Example 5 (Ours):**
- Story: youth-who-was-to-serve-three-years-without-pay
- Question: What pivotal moment connected the youth's decision to speak with the prince and the prince's advice about choosing the ring?
- Target answer: the prince knew what was most valuable .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial - The judge-identified sentences cover the advice about choosing the ring but miss the youth's decision to speak with the prince and the dialogue leading up to the advice, which are crucial for understanding the full reasoning chain.

### Ours focus-match examples (quality-pass, focus_match=yes)

**Focus Example 1:**
- Story: three-dogs
- Question: Why did the princes take counsel together to get the better of the youth and win power and glory for themselves?
- Target answer: they were jealous of the youth .
- answer_role=answer, question_focus=bridge, node_type=action

**Focus Example 2:**
- Story: youth-who-wanted-to-win-daughter-of-mother-in-corner
- Question: How did the mother feel as the son continued to sing and dance all day and night?
- Target answer: unhappy .
- answer_role=answer, question_focus=state, node_type=state

**Focus Example 3:**
- Story: youth-who-wanted-to-win-daughter-of-mother-in-corner
- Question: How did the mother come to feel that she could no longer support her son's lifestyle?
- Target answer: she could not afford to care for him anymore .
- answer_role=answer, question_focus=state, node_type=belief

### Best baseline examples (quality-pass)

**Direct Example:**
- Story: three-dogs
- Question: Why did the princes decide to strangle the youth who had saved the princesses?
- Target answer: they were jealous of the youth .
- Predicted difficulty: Medium

**ICL Example:**
- Story: three-dogs
- Question: Why did the youth agree to exchange his pigs for the old man's dogs each time they met?
- Target answer: three times .
- Predicted difficulty: Hard

**SelfRefine Example:**
- Story: youth-who-wanted-to-win-daughter-of-mother-in-corner
- Question: Why did the mother eventually tell her son to get to work, considering his increasing needs and the impact on their resources?
- Target answer: she could not afford to care for him anymore .
- Predicted difficulty: Medium

### Ours failure cases (grouped)

| Failure category | Count |
|---|---:|
| degenerate / parse failure | 10 |
| not fluent | 4 |
| not answerable | 4 |
| answer mismatch | 3 |

#### Ours failure examples

**Failure 1:**
- Story: three-dogs
- Question: How many did the old man propose the exchange of the dog for one of the young ma
- Reason: The question is not fluent due to the repetition of 'in the story' and the phrase 'multiple times in in the story story' is awkward and redundant.

**Failure 2:**
- Story: master-girl
- Question: Why why did the girl not not want to kill him .
- Reason: The question is not fully answerable from the story as it does not explicitly state the girl's intentions. The question is also not fluent due to repetition of 'why' and 'not not'. The expected answer is partially aligned with the story's implication but not directly stated.

**Failure 3:**
- Story: evil-one-kitta-grau
- Question: Why did the devil give up after three attempts to sow dissension between the new
- Reason: The question is answerable from the story, but it does not naturally lead to the expected answer about Kitta Grau's actions. The story indicates the devil's failure and Kitta Grau's success, but the question focuses on the devil's actions.

### Baseline failure cases

**Failure 1 (Direct):**
- Story: three-dogs
- Question: Who did the old man want to take with the dog and why did the youth do with the 
- Reason: The question is unclear and does not accurately reflect the story's content. It does not ask about the number of exchanges, which is what 'three times' would imply.

**Failure 2 (SelfRefine):**
- Story: three-dogs
- Question: 
- Reason: empty question

**Failure 3 (ICL):**
- Story: three-dogs
- Question: 
- Reason: empty question

## 8b. Pairwise Difference Table (Ours - Baseline)

| Metric | Ours | Direct | ICL | SelfRefine | Ours-Direct | Ours-ICL | Ours-SelfRefine |
|---|---|---:|---:|---:|---|---|---|
| quality_pass | 70.4% (50/71) | 52.1% (37/71) | 66.2% (47/71) | 35.2% (25/71) | +18.3pp | +4.2pp | +35.2pp |
| Hard hit | 16.0% (8/50) | 2.7% (1/37) | 8.5% (4/47) | 8.0% (2/25) | +13.3pp | +7.5pp | +8.0pp |
| HRP-v2 | 16.0% (8/50) | 2.7% (1/37) | 8.5% (4/47) | 8.0% (2/25) | +13.3pp | +7.5pp | +8.0pp |
| unique HRP-v2 stories | 6 | 1 | 4 | 1 | +5 | +2 | +5 |

## 9. Success Criteria

| Criterion | Status | Value |
|---|---|---|
| Ours quality_pass >= 65% | PASS | 70.4% (50/71 (70.4%, 95%CI [59.0, 79.8%])) |
| Ours predicted Hard >= 25% | FAIL | 16.0% (8/50 (16.0%, 95%CI [8.3, 28.5%])) |
| Ours eval_hrp_v2 >= 25% (quality-pass, judge-ok) | FAIL | 16.0% (8/50 (16.0%, 95%CI [8.3, 28.5%])) |
| Ours strict_hrp_v2 >= 10% | FAIL | 0.0% (0/50 (0.0%, 95%CI [0.0, 7.1%])) |
| Ours unique HRP-v2 stories > each baseline | PASS | Ours=6, Direct=1, ICL=4, SelfRefine=1 |
| Ours Hard hit >= Direct/ICL/SelfRefine | PASS | Ours=0.16, Direct=0.03, ICL=0.09, SelfRefine=0.08 |

**Overall: SOME CRITERIA FAILED**
