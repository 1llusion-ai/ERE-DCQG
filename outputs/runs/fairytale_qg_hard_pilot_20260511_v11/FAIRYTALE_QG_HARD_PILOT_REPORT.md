# FairytaleQA Hard QG Pilot Report

Generated: 2026-05-11 09:51:43

## 1. Run Summary

| Field | Value |
|---|---|
| Methods | Direct, ICL, SelfRefine, Ours |
| Requested limit | 50 |
| Graph total | 75 |
| Graph valid | 69 |
| Selected candidates | 50 |
| Total generations | 200 |
| Target difficulty | Hard |

### Parse success by method

| Method | parse_ok | Total | Pct |
|---|---:|---:|---:|
| Direct | 48 | 50 | 96.0% |
| ICL | 45 | 50 | 90.0% |
| SelfRefine | 27 | 50 | 54.0% |
| Ours | 48 | 50 | 96.0% |

### 1b. Generation Robustness by Method

| Method | degenerate | repair_attempted | repair_success | quality_pass |
|---|---:|---:|---:|---:|
| Direct | 2 | 0 | 0 | 31 |
| ICL | 6 | 0 | 0 | 29 |
| SelfRefine | 0 | 0 | 0 | 19 |
| Ours | 12 | 23 | 4 | 34 |

## 2. Quality Pass by Method

| Method | quality_pass | strict_quality_pass | Total | Pct (loose) | Pct (strict) |
|---|---:|---:|---:|---:|---:|
| Direct | 31 | 7 | 50 | 62.0% | 14.0% |
| ICL | 29 | 5 | 50 | 58.0% | 10.0% |
| SelfRefine | 19 | 0 | 50 | 38.0% | 0.0% |
| Ours | 34 | 8 | 50 | 68.0% | 16.0% |

## 3. Blind Difficulty Distribution (quality-pass only)

| Method | Easy | Medium | Hard | JudgeError | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 1 | 28 | 2 | 0 | 31 |
| ICL | 2 | 23 | 4 | 0 | 29 |
| SelfRefine | 2 | 14 | 3 | 0 | 19 |
| Ours | 3 | 23 | 8 | 0 | 34 |

### 3b. Difficulty Judge Status by Method

| Method | judge_ok | judge_error | Total | Error rate |
|---|---:|---:|---:|---:|
| Direct | 31 | 0 | 31 | 0.0% |
| ICL | 29 | 0 | 29 | 0.0% |
| SelfRefine | 19 | 0 | 19 | 0.0% |
| Ours | 34 | 0 | 34 | 0.0% |

## 4. Hard Hit Rate by Method

Denominator: quality-pass AND difficulty_judge_status=ok

| Method | Hard hit | Wilson 95% CI |
|---|---|---|
| Direct | 2/31 (6.5%, 95%CI [1.8, 20.7%]) | |
| ICL | 4/29 (13.8%, 95%CI [5.5, 30.6%]) | |
| SelfRefine | 3/19 (15.8%, 95%CI [5.5, 37.6%]) | |
| Ours | 8/34 (23.5%, 95%CI [12.4, 40.0%]) | |

## 5. Evidence Dependency by Method (quality-pass, judge-ok only)

| Method | alone_sufficient=no | bridge_required=yes | removal in {ambiguous,unanswerable} | Total |
|---|---:|---:|---:|---:|
| Direct | 11 | 30 | 26 | 31 |
| ICL | 12 | 27 | 22 | 29 |
| SelfRefine | 8 | 17 | 15 | 19 |
| Ours | 15 | 31 | 25 | 34 |

### 5b. Target Evidence Coverage by Method (quality-pass, judge-ok only)

| Method | mean coverage | coverage>=0.67 | uses_all_target | Total |
|---|---:|---:|---:|---:|
| Direct | 0.367 | 0 | 0 | 31 |
| ICL | 0.327 | 0 | 0 | 29 |
| SelfRefine | 0.377 | 0 | 0 | 19 |
| Ours | 0.446 | 2 | 1 | 34 |

### 5c. Hard Realization Pass by Method (exact-id diagnostic)

Hard realization (legacy) = judge_ok AND num_judge_used>=3 AND uses_bridge in {yes,partial} AND coverage>=0.67 AND predicted=Hard

| Method | hard_realization_pass | quality-pass judge-ok | Rate |
|---|---:|---:|---:|
| Direct | 0 | 31 | 0.0% |
| ICL | 0 | 29 | 0.0% |
| SelfRefine | 0 | 19 | 0.0% |
| Ours | 2 | 34 | 5.9% |

### 5e. Hard Realization Pass v2 by Method

Denominator: quality-pass AND difficulty_judge_status=ok

hrp_v2 = predicted=Hard AND num_judge_used>=3 AND bridge_required=yes AND alone_sufficient=no AND semantic_evidence_match in {yes,partial}

| Method | hrp_v2 | Wilson 95% CI |
|---|---|---|
| Direct | 2/31 (6.5%, 95%CI [1.8, 20.7%]) | |
| ICL | 4/29 (13.8%, 95%CI [5.5, 30.6%]) | |
| SelfRefine | 2/19 (10.5%, 95%CI [2.9, 31.4%]) | |
| Ours | 7/34 (20.6%, 95%CI [10.3, 36.8%]) | |

### 5e2. Strict HRP-v2 by Method

strict_hrp_v2 = hard_realization_pass_v2=yes AND strict_quality_pass=true AND focus_match=yes

| Method | strict_hrp_v2 | Wilson 95% CI |
|---|---|---|
| Direct | 0/31 (0.0%, 95%CI [0.0, 11.0%]) | |
| ICL | 0/29 (0.0%, 95%CI [0.0, 11.7%]) | |
| SelfRefine | 0/19 (0.0%, 95%CI [0.0, 16.8%]) | |
| Ours | 2/34 (5.9%, 95%CI [1.6, 19.1%]) | |

### 5f. Semantic Evidence Match by Method (quality-pass, judge-ok)

| Method | yes | partial | no | judge_error | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 0 | 29 | 2 | 0 | 31 |
| ICL | 0 | 27 | 2 | 0 | 29 |
| SelfRefine | 0 | 18 | 1 | 0 | 19 |
| Ours | 1 | 31 | 2 | 0 | 34 |

### 5d. Answer Focus Diagnostics (Ours)

#### Question focus distribution

| Focus | Count | Pct |
|---|---:|---:|
| bridge | 18 | 52.9% |
| state | 12 | 35.3% |
| motivation | 3 | 8.8% |
| count | 1 | 2.9% |

#### Focus match rate

- focus_match=yes: 25 / 34
- focus_match=no: 9 / 34

#### Answer role -> question focus mapping

| answer_role | question_focus | count |
|---|---|---:|
| answer | bridge | 15 |
| answer | state | 12 |
| answer | motivation | 2 |
| answer_bridge | bridge | 3 |
| answer_bridge | motivation | 1 |
| count_pattern | count | 1 |

#### Focus mismatch examples

**Mismatch 1:**
- Story: thomas-the-rhymer
- Question: Why did Thomas have to ride with the fairy queen?
- Target answer: he needed to pay for the price of kissing her .
- answer_role=answer, question_focus=state

**Mismatch 2:**
- Story: thomas-the-rhymer
- Question: Why did Thomas have to restrain himself from eating the fruit in the orchard?
- Target answer: it was not safe to eat .
- answer_role=answer_bridge, question_focus=bridge

**Mismatch 3:**
- Story: master-girl
- Question: What motivated the king's son to continue exploring the rooms despite not finding anything alarming in the first one?
- Target answer: intrigued .
- answer_role=answer, question_focus=bridge

### 5g. Unique Story Diversity and Cluster Diagnostic

#### Unique stories among predicted Hard (quality-pass, judge-ok)

| Method | unique stories | Hard count | stories |
|---|---:|---:|---|
| Direct | 2 | 2 | three-dogs, youth-who-was-to-serve-three-years-without-pay |
| ICL | 4 | 4 | evil-one-kitta-grau, how-princess-pride-was-broken, three-dogs, youth-who-was-to-serve-three-years-without-pay |
| SelfRefine | 3 | 3 | how-princess-pride-was-broken, master-girl, the-one-handed-girl |
| Ours | 6 | 8 | evil-one-kitta-grau, little-lasse, the-fairies-of-merlin-crag, the-fire-plume, the-one-handed-girl, three-dogs |

#### Unique stories among hrp_v2 (quality-pass, judge-ok)

| Method | unique stories | hrp_v2 count | stories |
|---|---:|---:|---|
| Direct | 2 | 2 | three-dogs, youth-who-was-to-serve-three-years-without-pay |
| ICL | 4 | 4 | evil-one-kitta-grau, how-princess-pride-was-broken, three-dogs, youth-who-was-to-serve-three-years-without-pay |
| SelfRefine | 2 | 2 | how-princess-pride-was-broken, the-one-handed-girl |
| Ours | 6 | 7 | evil-one-kitta-grau, little-lasse, the-fairies-of-merlin-crag, the-fire-plume, the-one-handed-girl, three-dogs |

#### Unique stories among strict_hrp_v2 (quality-pass, judge-ok)

| Method | unique stories | strict_hrp_v2 count | stories |
|---|---:|---:|---|
| Direct | 0 | 0 |  |
| ICL | 0 | 0 |  |
| SelfRefine | 0 | 0 |  |
| Ours | 2 | 2 | evil-one-kitta-grau, three-dogs |

#### Cluster diagnostic: three-dogs concentration

| Method | three-dogs in Hard | total Hard | three-dogs in hrp_v2 | total hrp_v2 | three-dogs in strict_hrp_v2 | total strict_hrp_v2 |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 1 | 2 | 1 | 2 | 0 | 0 |
| ICL | 1 | 4 | 1 | 4 | 0 | 0 |
| SelfRefine | 0 | 3 | 0 | 2 | 0 | 0 |
| Ours | 1 | 8 | 1 | 7 | 1 | 2 |

## 6. Failure Reasons by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | wrong answer | 10 |
| Direct | not answerable | 7 |
| Direct | not fluent | 2 |
| ICL | not answerable | 13 |
| ICL | wrong answer | 5 |
| ICL | not fluent | 2 |
| ICL | other | 1 |
| SelfRefine | not answerable | 26 |
| SelfRefine | wrong answer | 5 |
| Ours | not answerable | 8 |
| Ours | not fluent | 5 |
| Ours | wrong answer | 3 |

### 6b. Difficulty Judge Parse Failures

None.

## 7. Copy/Reference Diagnostics

| Method | Total | Copies source | Copy rate |
|---|---:|---:|---:|
| Direct | 50 | 0 | 0.0% |
| ICL | 50 | 0 | 0.0% |
| SelfRefine | 50 | 0 | 0.0% |
| Ours | 50 | 0 | 0.0% |

## 8. Examples

### Best Ours examples (quality-pass, predicted Hard)

**Example 1:**
- Story: three-dogs
- Question: How many times did the old man offer to exchange his dog for one of the youth's pigs?
- Target answer: three times .
- Quality: answerable=yes, asks_expected=yes, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 0.333, hard_realization=no, hrp_v2=yes
- Focus: answer_role=count_pattern, question_focus=count, focus_match=yes
- Semantic match: partial — The judge-identified sentences [17, 31, 32] cover the third exchange but miss the first and second exchanges, which are covered by the target evidence sentences [4, 16, 31].

**Example 2:**
- Story: little-lasse
- Question: What development connected the situation where the oars were locked up in the boat-house to the difficulty of rowing to Asia?
- Target answer: he did not have oars to row with .
- Quality: answerable=yes, asks_expected=partial, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 1.000, hard_realization=yes, hrp_v2=yes
- Focus: answer_role=answer, question_focus=bridge, focus_match=yes
- Semantic match: yes — The judge-identified sentences [6, 7, 8] cover the same causal chain as the target evidence sentences, explaining that the oars were locked up, Lasse did not have oars to row with, and it is difficult to row to Asia without oars.

**Example 3:**
- Story: the-one-handed-girl
- Question: What development connected the girl's decision to follow the snake's instructions with the discovery of her baby?
- Target answer: her unwounded arm .
- Quality: answerable=yes, asks_expected=partial, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 0.333, hard_realization=no, hrp_v2=no
- Focus: answer_role=answer, question_focus=bridge, focus_match=yes
- Semantic match: partial — The judge-identified sentences [5, 8] cover part of the reasoning chain, specifically the action of using the other arm and the discovery of the baby, but miss the initial detailed action of searching with the whole hand as described in sentence [2].

### Hard realization pass v2 examples

**HRP-v2 Example 1 (Direct):**
- Story: three-dogs
- Question: Why did the young swineherd agree to exchange his pigs for dogs three times?
- Target answer: three times .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences cover the final exchange but miss the earlier exchanges that establish the pattern, which are crucial for the full reasoning chain.

**HRP-v2 Example 2 (ICL):**
- Story: three-dogs
- Question: How many times did the youth exchange his pigs for dogs from the old man?
- Target answer: three times .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences [17, 31, 32] cover the final exchange but miss the first and second exchanges, which are covered in the target evidence sentences [4, 16, 31].

**HRP-v2 Example 3 (Ours):**
- Story: three-dogs
- Question: How many times did the old man offer to exchange his dog for one of the youth's pigs?
- Target answer: three times .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences [17, 31, 32] cover the third exchange but miss the first and second exchanges, which are covered by the target evidence sentences [4, 16, 31].

**HRP-v2 Example 4 (Ours):**
- Story: little-lasse
- Question: What development connected the situation where the oars were locked up in the boat-house to the difficulty of rowing to Asia?
- Target answer: he did not have oars to row with .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: yes — The judge-identified sentences [6, 7, 8] cover the same causal chain as the target evidence sentences, explaining that the oars were locked up, Lasse did not have oars to row with, and it is difficult to row to Asia without oars.

**HRP-v2 Example 5 (ICL):**
- Story: evil-one-kitta-grau
- Question: Why did the evil one recognize Kitta Grau and not want to buy her from the merchant?
- Target answer: the merchant was unable to sell kitta grau .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences [11, 12, 13] explain the recognition of Kitta Grau by the evil one and his decision not to buy her, but they do not cover the merchant's inability to sell Kitta Grau over the three weeks, which is a key part of the reasoning chain covered by the target evidence sentences [7, 9, 15].

### Ours focus-match examples (quality-pass, focus_match=yes)

**Focus Example 1:**
- Story: three-dogs
- Question: How many times did the old man offer to exchange his dog for one of the youth's pigs?
- Target answer: three times .
- answer_role=count_pattern, question_focus=count, node_type=count

**Focus Example 2:**
- Story: three-dogs
- Question: What motivated the princes to plot against the youth and ultimately lead to his death?
- Target answer: they were jealous of the youth .
- answer_role=answer, question_focus=bridge, node_type=action

**Focus Example 3:**
- Story: youth-who-wanted-to-win-daughter-of-mother-in-corner
- Question: How did the mother come to the decision that she should make the youth work instead of continuing to support him?
- Target answer: she could not afford to care for him anymore .
- answer_role=answer, question_focus=state, node_type=belief

### Best baseline examples (quality-pass)

**Direct Example:**
- Story: three-dogs
- Question: Why did the young swineherd agree to exchange his pigs for dogs three times?
- Target answer: three times .
- Predicted difficulty: Hard

**ICL Example:**
- Story: three-dogs
- Question: How many times did the youth exchange his pigs for dogs from the old man?
- Target answer: three times .
- Predicted difficulty: Hard

**SelfRefine Example:**
- Story: silverwhite-lillwacker
- Question: Why did the courtier force the princess to promise that he, and none other, had rescued her, and how did this action compare to the actual rescuer's behavior?
- Target answer: he pretended to have saved the two princesses .
- Predicted difficulty: Medium

### Ours failure cases (grouped)

| Failure category | Count |
|---|---:|
| degenerate / parse failure | 5 |
| not answerable | 5 |
| other | 4 |
| focus mismatch | 2 |

#### Ours failure examples

**Failure 1:**
- Story: werewolf
- Question: How did the princess come to feel so lonely and sad that she did not even notice
- Reason: The question is answerable from the story as it describes the princess's feelings of loneliness and sadness. However, the expected answer does not match the target answer 'help her.' The story does not indicate that the princess needed help, and the target answer is not consistent with the story's content.

**Failure 2:**
- Story: silverwhite-lillwacker
- Question: What what was the ultimate consequence of the court way 's journey?
- Reason: The question contains a grammatical error ('What what') and is not clear about which journey it refers to. The story does not mention two princesses being rescued, only one. The courtier's action of pretending to have saved the princess is implied but not directly stated.

**Failure 3:**
- Story: torre-jeppe
- Question: Why Why did the girl finally agree herself to go . and how the tailorsors offere
- Reason: The question is not clear and contains grammatical errors, making it difficult to understand. However, the story does provide a reason related to her bravery and the offer of a dress, which partially answers the question.

### Baseline failure cases

**Failure 1 (SelfRefine):**
- Story: three-dogs
- Question: 
- Reason: empty question

**Failure 2 (Direct):**
- Story: three-dogs
- Question: 
- Reason: empty question

**Failure 3 (SelfRefine):**
- Story: three-dogs
- Question: 
- Reason: empty question

## 9. Success Criteria

| Criterion | Status | Value |
|---|---|---|
| Ours quality_pass >= 65% | PASS | 68.0% (34/50 (68.0%, 95%CI [54.2, 79.2%])) |
| Ours predicted Hard >= 25% | FAIL | 23.5% (8/34 (23.5%, 95%CI [12.4, 40.0%])) |
| Ours eval_hrp_v2 >= 25% (quality-pass, judge-ok) | FAIL | 20.6% (7/34 (20.6%, 95%CI [10.3, 36.8%])) |
| Ours strict_hrp_v2 >= 10% | FAIL | 5.9% (2/34 (5.9%, 95%CI [1.6, 19.1%])) |
| Ours unique HRP-v2 stories > each baseline | PASS | Ours=6, Direct=2, ICL=4, SelfRefine=2 |
| Ours Hard hit >= Direct/ICL/SelfRefine | PASS | Ours=0.24, Direct=0.06, ICL=0.14, SelfRefine=0.16 |

**Overall: SOME CRITERIA FAILED**
