# FairytaleQA Hard QG Pilot Report

Generated: 2026-05-11 00:38:18

## 1. Run Summary

| Field | Value |
|---|---|
| Methods | Direct, ICL, SelfRefine, Ours |
| Total candidates | 15 |
| Total generations | 60 |
| Target difficulty | Hard |

### Parse success by method

| Method | parse_ok | Total | Pct |
|---|---:|---:|---:|
| Direct | 15 | 15 | 100.0% |
| ICL | 11 | 15 | 73.3% |
| SelfRefine | 5 | 15 | 33.3% |
| Ours | 12 | 15 | 80.0% |

## 2. Quality Pass by Method

| Method | quality_pass | strict_quality_pass | Total | Pct (loose) | Pct (strict) |
|---|---:|---:|---:|---:|---:|
| Direct | 12 | 4 | 15 | 80.0% | 26.7% |
| ICL | 8 | 2 | 15 | 53.3% | 13.3% |
| SelfRefine | 4 | 1 | 15 | 26.7% | 6.7% |
| Ours | 11 | 3 | 15 | 73.3% | 20.0% |

## 3. Blind Difficulty Distribution (quality-pass only)

| Method | Easy | Medium | Hard | JudgeError | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 1 | 10 | 1 | 0 | 12 |
| ICL | 2 | 5 | 1 | 0 | 8 |
| SelfRefine | 0 | 3 | 1 | 0 | 4 |
| Ours | 0 | 7 | 4 | 0 | 11 |

### 3b. Difficulty Judge Status by Method

| Method | judge_ok | judge_error | Total | Error rate |
|---|---:|---:|---:|---:|
| Direct | 12 | 0 | 12 | 0.0% |
| ICL | 8 | 0 | 8 | 0.0% |
| SelfRefine | 4 | 0 | 4 | 0.0% |
| Ours | 11 | 0 | 11 | 0.0% |

## 4. Hard Hit Rate by Method

Denominator: quality-pass AND difficulty_judge_status=ok

| Method | Hard | judge-ok QP | Hard hit rate |
|---|---:|---:|---:|
| Direct | 1 | 12 | 8.3% |
| ICL | 1 | 8 | 12.5% |
| SelfRefine | 1 | 4 | 25.0% |
| Ours | 4 | 11 | 36.4% |

## 5. Evidence Dependency by Method (quality-pass, judge-ok only)

| Method | alone_sufficient=no | bridge_required=yes | removal in {ambiguous,unanswerable} | Total |
|---|---:|---:|---:|---:|
| Direct | 6 | 10 | 9 | 12 |
| ICL | 3 | 7 | 4 | 8 |
| SelfRefine | 3 | 4 | 4 | 4 |
| Ours | 9 | 11 | 9 | 11 |

### 5b. Target Evidence Coverage by Method (quality-pass, judge-ok only)

| Method | mean coverage | coverage>=0.67 | uses_all_target | Total |
|---|---:|---:|---:|---:|
| Direct | 0.326 | 0 | 0 | 12 |
| ICL | 0.365 | 0 | 0 | 8 |
| SelfRefine | 0.333 | 0 | 0 | 4 |
| Ours | 0.503 | 1 | 1 | 11 |

### 5c. Hard Realization Pass by Method (exact-id diagnostic)

Hard realization (legacy) = judge_ok AND num_judge_used>=3 AND uses_bridge in {yes,partial} AND coverage>=0.67 AND predicted=Hard

| Method | hard_realization_pass | quality-pass judge-ok | Rate |
|---|---:|---:|---:|
| Direct | 0 | 12 | 0.0% |
| ICL | 0 | 8 | 0.0% |
| SelfRefine | 0 | 4 | 0.0% |
| Ours | 1 | 11 | 9.1% |

### 5e. Hard Realization Pass v2 by Method

hrp_v2 = predicted=Hard AND num_judge_used>=3 AND bridge_required=yes AND alone_sufficient=no AND semantic_evidence_match in {yes,partial}

| Method | hrp_v2 | quality-pass judge-ok | Rate |
|---|---:|---:|---:|
| Direct | 1 | 12 | 8.3% |
| ICL | 1 | 8 | 12.5% |
| SelfRefine | 1 | 4 | 25.0% |
| Ours | 4 | 11 | 36.4% |

### 5f. Semantic Evidence Match by Method (quality-pass, judge-ok)

| Method | yes | partial | no | judge_error | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 0 | 10 | 2 | 0 | 12 |
| ICL | 0 | 7 | 1 | 0 | 8 |
| SelfRefine | 0 | 4 | 0 | 0 | 4 |
| Ours | 1 | 10 | 0 | 0 | 11 |

### 5d. Answer Focus Diagnostics (Ours)

#### Question focus distribution

| Focus | Count | Pct |
|---|---:|---:|
| bridge | 5 | 45.5% |
| state | 4 | 36.4% |
| count | 1 | 9.1% |
| motivation | 1 | 9.1% |

#### Focus match rate

- focus_match=yes: 10 / 11
- focus_match=no: 1 / 11

#### Answer role -> question focus mapping

| answer_role | question_focus | count |
|---|---|---:|
| answer | state | 4 |
| answer | bridge | 4 |
| answer | motivation | 1 |
| answer_bridge | bridge | 1 |
| count_pattern | count | 1 |

#### Focus mismatch examples

**Mismatch 1:**
- Story: the-one-handed-girl
- Question: What development connected the girl's decision to use her wounded arm to the recovery of her baby?
- Target answer: her unwounded arm .
- answer_role=answer, question_focus=bridge

## 6. Failure Reasons by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | other | 1 |
| Direct | wrong answer | 1 |
| Direct | not answerable | 1 |
| ICL | not answerable | 5 |
| ICL | wrong answer | 1 |
| ICL | not fluent | 1 |
| SelfRefine | not answerable | 11 |
| Ours | not answerable | 4 |

### 6b. Difficulty Judge Parse Failures

None.

## 7. Copy/Reference Diagnostics

| Method | Total | Copies source | Copy rate |
|---|---:|---:|---:|
| Direct | 15 | 0 | 0.0% |
| ICL | 15 | 0 | 0.0% |
| SelfRefine | 15 | 0 | 0.0% |
| Ours | 15 | 0 | 0.0% |

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
- Semantic match: partial — The judge-identified sentences [17, 31, 32] cover the third exchange but miss the first and second exchanges, which are captured in the target evidence sentences [4, 16, 31].

**Example 2:**
- Story: little-lasse
- Question: What development connected the situation where the boat was empty and the difficulty in rowing to Asia?
- Target answer: he did not have oars to row with .
- Quality: answerable=yes, asks_expected=yes, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 1.000, hard_realization=yes, hrp_v2=yes
- Focus: answer_role=answer, question_focus=bridge, focus_match=yes
- Semantic match: yes — The judge-identified sentences [6, 7, 8] cover the same reasoning chain as the target evidence sentences, explaining the absence of oars and the resulting difficulty in rowing to Asia.

**Example 3:**
- Story: the-one-handed-girl
- Question: What development connected the girl's decision to use her wounded arm to the recovery of her baby?
- Target answer: her unwounded arm .
- Quality: answerable=yes, asks_expected=partial, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 0.333, hard_realization=no, hrp_v2=yes
- Focus: answer_role=answer, question_focus=bridge, focus_match=no
- Semantic match: partial — The judge-identified sentences [6, 8, 14] cover the use of the wounded arm and the recovery of the baby but miss the initial reluctance and the act of putting her fingers into the tiniest crannies, which are captured in the target evidence sentences [2, 7, 8].

### Hard realization pass v2 examples

**HRP-v2 Example 1 (Direct):**
- Story: three-dogs
- Question: Why did the young swineherd agree to exchange his pigs for dogs three times?
- Target answer: three times .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences cover the final exchange but miss the earlier exchanges' motivations, which are included in the target evidence sentences.

**HRP-v2 Example 2 (ICL):**
- Story: three-dogs
- Question: How many times did the youth exchange his pigs for dogs from the old man?
- Target answer: three times .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences cover part of the chain but miss the first exchange, which is a key link in the reasoning chain.

**HRP-v2 Example 3 (SelfRefine):**
- Story: three-dogs
- Question: Why did the young swineherd agree to exchange his pigs for dogs three times?
- Target answer: three times .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences cover the final exchange but miss the earlier exchanges that establish the pattern, which are crucial for understanding the full reasoning chain.

**HRP-v2 Example 4 (Ours):**
- Story: three-dogs
- Question: How many times did the old man propose exchanging his dog for one of the youth's pigs?
- Target answer: three times .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences [17, 31, 32] cover the third exchange but miss the first and second exchanges, which are captured in the target evidence sentences [4, 16, 31].

**HRP-v2 Example 5 (ICL):**
- Story: little-lasse
- Question: Why was Little Lasse frightened and begin to cry as his boat drifted out to sea?
- Target answer: he did not have oars to row with .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences [6, 10, 11] cover the lack of oars and the boat drifting out to sea, which are key points. However, they miss the specific mention of the oars being locked up in the boat-house, which is a detail provided in the target evidence sentence [7].

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
- Question: How did the princess feel after the king left for war and the queen showed her true nature?
- Target answer: poorly .
- answer_role=answer, question_focus=state, node_type=state

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
- Story: three-dogs
- Question: Why did the young swineherd agree to exchange his pigs for dogs three times?
- Target answer: three times .
- Predicted difficulty: Hard

### Failure cases

**Failure 1 (Ours):**
- Story: three-dogs
- Question: 
- Reason: empty question

**Failure 2 (ICL):**
- Story: youth-who-wanted-to-win-daughter-of-mother-in-corner
- Question: 
- Reason: empty question

**Failure 3 (SelfRefine):**
- Story: youth-who-wanted-to-win-daughter-of-mother-in-corner
- Question: 
- Reason: empty question

## 9. Success Criteria

| Criterion | Status | Value |
|---|---|---|
| Ours quality-pass >= 60% | PASS | 73.3% |
| Ours strict quality-pass >= 40% | FAIL | 20.0% |
| Ours focus_match=yes >= 50% | PASS | 90.9% |
| Ours predicted Hard among quality-pass >= 20% | PASS | 36.4% |
| Ours hard_realization_pass (exact-id) >= 15% | FAIL | 9.1% |
| Ours hard_realization_pass_v2 >= 15% | PASS | 36.4% |
| Ours bridge_required=yes >= 50% | PASS | 100.0% |
| Ours Hard hit > Direct/ICL/SelfRefine | PASS | Ours=0.36, Direct=0.08, ICL=0.12, SelfRefine=0.25 |
| No method copies source question | PASS | pass |

**Overall: SOME CRITERIA FAILED**
