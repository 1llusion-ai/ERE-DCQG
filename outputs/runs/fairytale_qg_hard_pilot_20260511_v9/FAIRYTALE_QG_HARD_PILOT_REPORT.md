# FairytaleQA Hard QG Pilot Report

Generated: 2026-05-11 01:17:08

## 1. Run Summary

| Field | Value |
|---|---|
| Methods | Direct, ICL, SelfRefine, Ours |
| Total candidates | 19 |
| Total generations | 76 |
| Target difficulty | Hard |

### Parse success by method

| Method | parse_ok | Total | Pct |
|---|---:|---:|---:|
| Direct | 14 | 19 | 73.7% |
| ICL | 17 | 19 | 89.5% |
| SelfRefine | 7 | 19 | 36.8% |
| Ours | 12 | 19 | 63.2% |

## 2. Quality Pass by Method

| Method | quality_pass | strict_quality_pass | Total | Pct (loose) | Pct (strict) |
|---|---:|---:|---:|---:|---:|
| Direct | 12 | 3 | 19 | 63.2% | 15.8% |
| ICL | 13 | 2 | 19 | 68.4% | 10.5% |
| SelfRefine | 5 | 1 | 19 | 26.3% | 5.3% |
| Ours | 11 | 3 | 19 | 57.9% | 15.8% |

## 3. Blind Difficulty Distribution (quality-pass only)

| Method | Easy | Medium | Hard | JudgeError | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 0 | 11 | 1 | 0 | 12 |
| ICL | 0 | 11 | 2 | 0 | 13 |
| SelfRefine | 0 | 4 | 1 | 0 | 5 |
| Ours | 1 | 7 | 3 | 0 | 11 |

### 3b. Difficulty Judge Status by Method

| Method | judge_ok | judge_error | Total | Error rate |
|---|---:|---:|---:|---:|
| Direct | 12 | 0 | 12 | 0.0% |
| ICL | 13 | 0 | 13 | 0.0% |
| SelfRefine | 5 | 0 | 5 | 0.0% |
| Ours | 11 | 0 | 11 | 0.0% |

## 4. Hard Hit Rate by Method

Denominator: quality-pass AND difficulty_judge_status=ok

| Method | Hard hit | Wilson 95% CI |
|---|---|---|
| Direct | 1/12 (8.3%, 95%CI [1.5–35.4%]) | |
| ICL | 2/13 (15.4%, 95%CI [4.3–42.2%]) | |
| SelfRefine | 1/5 (20.0%, 95%CI [3.6–62.4%]) | |
| Ours | 3/11 (27.3%, 95%CI [9.7–56.6%]) | |

## 5. Evidence Dependency by Method (quality-pass, judge-ok only)

| Method | alone_sufficient=no | bridge_required=yes | removal in {ambiguous,unanswerable} | Total |
|---|---:|---:|---:|---:|
| Direct | 4 | 12 | 11 | 12 |
| ICL | 6 | 11 | 10 | 13 |
| SelfRefine | 2 | 5 | 3 | 5 |
| Ours | 6 | 10 | 7 | 11 |

### 5b. Target Evidence Coverage by Method (quality-pass, judge-ok only)

| Method | mean coverage | coverage>=0.67 | uses_all_target | Total |
|---|---:|---:|---:|---:|
| Direct | 0.340 | 0 | 0 | 12 |
| ICL | 0.333 | 0 | 0 | 13 |
| SelfRefine | 0.417 | 0 | 0 | 5 |
| Ours | 0.488 | 0 | 0 | 11 |

### 5c. Hard Realization Pass by Method (exact-id diagnostic)

Hard realization (legacy) = judge_ok AND num_judge_used>=3 AND uses_bridge in {yes,partial} AND coverage>=0.67 AND predicted=Hard

| Method | hard_realization_pass | quality-pass judge-ok | Rate |
|---|---:|---:|---:|
| Direct | 0 | 12 | 0.0% |
| ICL | 1 | 13 | 7.7% |
| SelfRefine | 0 | 5 | 0.0% |
| Ours | 1 | 11 | 9.1% |

### 5e. Hard Realization Pass v2 by Method

hrp_v2 = predicted=Hard AND num_judge_used>=3 AND bridge_required=yes AND alone_sufficient=no AND semantic_evidence_match in {yes,partial}

| Method | hrp_v2 | Wilson 95% CI |
|---|---|---|
| Direct | 1/12 (8.3%, 95%CI [1.5–35.4%]) | |
| ICL | 2/13 (15.4%, 95%CI [4.3–42.2%]) | |
| SelfRefine | 0/5 (0.0%, 95%CI [0.0–43.4%]) | |
| Ours | 3/11 (27.3%, 95%CI [9.7–56.6%]) | |

### 5e2. Strict HRP-v2 by Method

strict_hrp_v2 = hard_realization_pass_v2=yes AND strict_quality_pass=true AND focus_match=yes

| Method | strict_hrp_v2 | Wilson 95% CI |
|---|---|---|
| Direct | 0/12 (0.0%, 95%CI [0.0–24.3%]) | |
| ICL | 0/13 (0.0%, 95%CI [0.0–22.8%]) | |
| SelfRefine | 0/5 (0.0%, 95%CI [0.0–43.4%]) | |
| Ours | 2/11 (18.2%, 95%CI [5.1–47.7%]) | |

### 5f. Semantic Evidence Match by Method (quality-pass, judge-ok)

| Method | yes | partial | no | judge_error | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 0 | 11 | 1 | 0 | 12 |
| ICL | 0 | 13 | 0 | 0 | 13 |
| SelfRefine | 0 | 4 | 1 | 0 | 5 |
| Ours | 0 | 10 | 1 | 0 | 11 |

### 5d. Answer Focus Diagnostics (Ours)

#### Question focus distribution

| Focus | Count | Pct |
|---|---:|---:|
| bridge | 5 | 45.5% |
| state | 3 | 27.3% |
| motivation | 2 | 18.2% |
| count | 1 | 9.1% |

#### Focus match rate

- focus_match=yes: 10 / 11
- focus_match=no: 1 / 11

#### Answer role -> question focus mapping

| answer_role | question_focus | count |
|---|---|---:|
| answer | bridge | 4 |
| answer | state | 3 |
| answer | motivation | 2 |
| answer_bridge | bridge | 1 |
| count_pattern | count | 1 |

#### Focus mismatch examples

**Mismatch 1:**
- Story: three-dogs
- Question: What development connected the princes' jealousy to the youth's death?
- Target answer: they were jealous of the youth .
- answer_role=answer, question_focus=bridge

### 5g. Unique Story Diversity and Cluster Diagnostic

#### Unique stories among predicted Hard (quality-pass, judge-ok)

| Method | unique stories | Hard count | stories |
|---|---:|---:|---|
| Direct | 1 | 1 | three-dogs |
| ICL | 2 | 2 | evil-one-kitta-grau, the-crane-that-crossed-the-river |
| SelfRefine | 1 | 1 | master-girl |
| Ours | 3 | 3 | evil-one-kitta-grau, master-girl, three-dogs |

#### Unique stories among hrp_v2 (quality-pass, judge-ok)

| Method | unique stories | hrp_v2 count | stories |
|---|---:|---:|---|
| Direct | 1 | 1 | three-dogs |
| ICL | 2 | 2 | evil-one-kitta-grau, the-crane-that-crossed-the-river |
| SelfRefine | 0 | 0 |  |
| Ours | 3 | 3 | evil-one-kitta-grau, master-girl, three-dogs |

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
| Direct | 1 | 1 | 1 | 1 | 0 | 0 |
| ICL | 0 | 2 | 0 | 2 | 0 | 0 |
| SelfRefine | 0 | 1 | 0 | 0 | 0 | 0 |
| Ours | 1 | 3 | 1 | 3 | 1 | 2 |

## 6. Failure Reasons by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | not answerable | 5 |
| Direct | wrong answer | 2 |
| ICL | not answerable | 3 |
| ICL | wrong answer | 2 |
| ICL | not fluent | 1 |
| SelfRefine | not answerable | 12 |
| SelfRefine | wrong answer | 2 |
| Ours | not answerable | 7 |
| Ours | wrong answer | 1 |

### 6b. Difficulty Judge Parse Failures

None.

## 7. Copy/Reference Diagnostics

| Method | Total | Copies source | Copy rate |
|---|---:|---:|---:|
| Direct | 19 | 0 | 0.0% |
| ICL | 19 | 0 | 0.0% |
| SelfRefine | 19 | 0 | 0.0% |
| Ours | 19 | 0 | 0.0% |

## 8. Examples

### Best Ours examples (quality-pass, predicted Hard)

**Example 1:**
- Story: three-dogs
- Question: How many times did the old man propose exchanging his dog for one of the youth's pigs?
- Target answer: three times .
- Quality: answerable=yes, asks_expected=yes, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 0.333, hard_realization=no, hrp_v2=yes
- Focus: answer_role=count_pattern, question_focus=count, focus_match=yes
- Semantic match: partial — The judge-identified sentences [17, 31, 32] cover the final exchange and the agreement to the exchange, but miss the initial proposal in the first interaction, which is covered in the target evidence sentence [4].

**Example 2:**
- Story: master-girl
- Question: What was the king's son's state of mind as he decided to try the soup in the third room after his discoveries in the previous ones?
- Target answer: intrigued .
- Quality: answerable=yes, asks_expected=partial, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 0.200, hard_realization=no, hrp_v2=yes
- Focus: answer_role=answer, question_focus=bridge, focus_match=yes
- Semantic match: partial — The judge-identified sentences [5, 9, 12] cover part of the king's son's curiosity and actions regarding the kettles but miss the initial thoughts and motivations expressed in sentences [2, 4, 6] which are crucial for understanding his state of mind fully.

**Example 3:**
- Story: evil-one-kitta-grau
- Question: What was the ultimate goal of Kitta Grau's actions that led to the evil one being afraid to give her a reward on the opposite side of a stream?
- Target answer: kitta grau managed to sow dissension between the couple .
- Quality: answerable=yes, asks_expected=yes, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 0.667, hard_realization=yes, hrp_v2=yes
- Focus: answer_role=answer, question_focus=motivation, focus_match=yes
- Semantic match: partial — The judge-identified sentences include [11] which introduces Kitta Grau's action but misses [15] which directly links to the reward and the evil one's fear, making the chain less complete.

### Hard realization pass v2 examples

**HRP-v2 Example 1 (Direct):**
- Story: three-dogs
- Question: Why did the young swineherd agree to exchange his pigs for dogs three times?
- Target answer: three times .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences cover the final exchange but miss the earlier exchanges' motivations, which are included in the target evidence sentences.

**HRP-v2 Example 2 (Ours):**
- Story: three-dogs
- Question: How many times did the old man propose exchanging his dog for one of the youth's pigs?
- Target answer: three times .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences [17, 31, 32] cover the final exchange and the agreement to the exchange, but miss the initial proposal in the first interaction, which is covered in the target evidence sentence [4].

**HRP-v2 Example 3 (ICL):**
- Story: master-girl
- Question: Why did the king's son decide to dip his hair in the third kettle and what did he expect to happen?
- Target answer: the second kettle turned his hair into silver .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences [9, 12, 13] cover the action of dipping hair in the third kettle and the result of it turning into gold, but they miss the comparison with the second kettle turning hair into silver, which is a key part of the reasoning chain in the target evidence sentences [7, 9, 13].

**HRP-v2 Example 4 (ICL):**
- Story: little-lasse
- Question: Why was Little Lasse frightened and begin to cry as his boat drifted out to sea?
- Target answer: he did not have oars to row with .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences [6, 10, 11] cover the situation of Lasse not having oars and the boat drifting out to sea, which led to him being frightened. However, they miss the detail about the oars being locked up in the boat-house, which is provided in sentence [7] of the target evidence sentences.

**HRP-v2 Example 5 (Ours):**
- Story: master-girl
- Question: What was the king's son's state of mind as he decided to try the soup in the third room after his discoveries in the previous ones?
- Target answer: intrigued .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences [5, 9, 12] cover part of the king's son's curiosity and actions regarding the kettles but miss the initial thoughts and motivations expressed in sentences [2, 4, 6] which are crucial for understanding his state of mind fully.

### Ours focus-match examples (quality-pass, focus_match=yes)

**Focus Example 1:**
- Story: three-dogs
- Question: How many times did the old man propose exchanging his dog for one of the youth's pigs?
- Target answer: three times .
- answer_role=count_pattern, question_focus=count, node_type=count

**Focus Example 2:**
- Story: youth-who-wanted-to-win-daughter-of-mother-in-corner
- Question: How did the mother come to feel that she could no longer support her son's lifestyle?
- Target answer: she could not afford to care for him anymore .
- answer_role=answer, question_focus=state, node_type=belief

**Focus Example 3:**
- Story: werewolf
- Question: How did the princess feel after the king departed and the queen showed her true nature?
- Target answer: poorly .
- answer_role=answer, question_focus=state, node_type=state

### Best baseline examples (quality-pass)

**Direct Example:**
- Story: three-dogs
- Question: Why did the young swineherd agree to exchange his pigs for dogs three times?
- Target answer: three times .
- Predicted difficulty: Hard

**ICL Example:**
- Story: youth-who-wanted-to-win-daughter-of-mother-in-corner
- Question: Why did the mother eventually decide that her son should start working instead of just singing and dancing all day?
- Target answer: she could not afford to care for him anymore .
- Predicted difficulty: Medium

**SelfRefine Example:**
- Story: three-dogs
- Question: Why did the princes take counsel together to get the better of the youth and win power and glory for themselves, and how did they plan to carry out their evil plot?
- Target answer: they were jealous of the youth .
- Predicted difficulty: Medium

### Ours failure cases (grouped)

| Failure category | Count |
|---|---:|
| empty / parse failure | 7 |
| other | 1 |

#### Ours failure examples

**Failure 1:**
- Story: thomas-the-rhymer
- Question: (empty)
- Reason: empty question

**Failure 2:**
- Story: silverwhite-lillwacker
- Question: (empty)
- Reason: empty question

**Failure 3:**
- Story: master-girl
- Question: (empty)
- Reason: empty question

### Baseline failure cases

**Failure 1 (ICL):**
- Story: three-dogs
- Question: 
- Reason: empty question

**Failure 2 (SelfRefine):**
- Story: three-dogs
- Question: 
- Reason: empty question

**Failure 3 (Direct):**
- Story: three-dogs
- Question: 
- Reason: empty question

## 9. Success Criteria

| Criterion | Status | Value |
|---|---|---|
| Ours quality_pass >= 65% | FAIL | 57.9% (11/19 (57.9%, 95%CI [36.3–76.9%])) |
| Ours predicted Hard >= 25% | PASS | 27.3% (3/11 (27.3%, 95%CI [9.7–56.6%])) |
| Ours hrp_v2 >= 25% | PASS | 27.3% (3/11 (27.3%, 95%CI [9.7–56.6%])) |
| Ours strict_hrp_v2 >= 10% | PASS | 18.2% (2/11 (18.2%, 95%CI [5.1–47.7%])) |
| Ours unique HRP-v2 stories > each baseline | PASS | Ours=3, Direct=1, ICL=2, SelfRefine=0 |
| Ours Hard hit >= Direct/ICL/SelfRefine | PASS | Ours=0.27, Direct=0.08, ICL=0.15, SelfRefine=0.20 |

**Overall: SOME CRITERIA FAILED**
