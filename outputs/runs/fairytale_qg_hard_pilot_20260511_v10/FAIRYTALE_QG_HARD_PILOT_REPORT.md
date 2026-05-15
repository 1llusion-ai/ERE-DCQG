# FairytaleQA Hard QG Pilot Report

Generated: 2026-05-11 02:58:45

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
| Direct | 43 | 50 | 86.0% |
| ICL | 40 | 50 | 80.0% |
| SelfRefine | 21 | 50 | 42.0% |
| Ours | 29 | 50 | 58.0% |

## 2. Quality Pass by Method

| Method | quality_pass | strict_quality_pass | Total | Pct (loose) | Pct (strict) |
|---|---:|---:|---:|---:|---:|
| Direct | 27 | 7 | 50 | 54.0% | 14.0% |
| ICL | 25 | 5 | 50 | 50.0% | 10.0% |
| SelfRefine | 11 | 2 | 50 | 22.0% | 4.0% |
| Ours | 24 | 6 | 50 | 48.0% | 12.0% |

## 3. Blind Difficulty Distribution (quality-pass only)

| Method | Easy | Medium | Hard | JudgeError | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 1 | 24 | 2 | 0 | 27 |
| ICL | 3 | 20 | 2 | 0 | 25 |
| SelfRefine | 0 | 10 | 1 | 0 | 11 |
| Ours | 2 | 17 | 5 | 0 | 24 |

### 3b. Difficulty Judge Status by Method

| Method | judge_ok | judge_error | Total | Error rate |
|---|---:|---:|---:|---:|
| Direct | 27 | 0 | 27 | 0.0% |
| ICL | 25 | 0 | 25 | 0.0% |
| SelfRefine | 11 | 0 | 11 | 0.0% |
| Ours | 24 | 0 | 24 | 0.0% |

## 4. Hard Hit Rate by Method

Denominator: quality-pass AND difficulty_judge_status=ok

| Method | Hard hit | Wilson 95% CI |
|---|---|---|
| Direct | 2/27 (7.4%, 95%CI [2.1, 23.4%]) | |
| ICL | 2/25 (8.0%, 95%CI [2.2, 25.0%]) | |
| SelfRefine | 1/11 (9.1%, 95%CI [1.6, 37.7%]) | |
| Ours | 5/24 (20.8%, 95%CI [9.2, 40.5%]) | |

## 5. Evidence Dependency by Method (quality-pass, judge-ok only)

| Method | alone_sufficient=no | bridge_required=yes | removal in {ambiguous,unanswerable} | Total |
|---|---:|---:|---:|---:|
| Direct | 12 | 26 | 22 | 27 |
| ICL | 10 | 22 | 17 | 25 |
| SelfRefine | 6 | 11 | 10 | 11 |
| Ours | 11 | 22 | 17 | 24 |

### 5b. Target Evidence Coverage by Method (quality-pass, judge-ok only)

| Method | mean coverage | coverage>=0.67 | uses_all_target | Total |
|---|---:|---:|---:|---:|
| Direct | 0.401 | 0 | 0 | 27 |
| ICL | 0.379 | 0 | 0 | 25 |
| SelfRefine | 0.417 | 0 | 0 | 11 |
| Ours | 0.399 | 1 | 1 | 24 |

### 5c. Hard Realization Pass by Method (exact-id diagnostic)

Hard realization (legacy) = judge_ok AND num_judge_used>=3 AND uses_bridge in {yes,partial} AND coverage>=0.67 AND predicted=Hard

| Method | hard_realization_pass | quality-pass judge-ok | Rate |
|---|---:|---:|---:|
| Direct | 0 | 27 | 0.0% |
| ICL | 0 | 25 | 0.0% |
| SelfRefine | 0 | 11 | 0.0% |
| Ours | 1 | 24 | 4.2% |

### 5e. Hard Realization Pass v2 by Method

Denominator: quality-pass AND difficulty_judge_status=ok

hrp_v2 = predicted=Hard AND num_judge_used>=3 AND bridge_required=yes AND alone_sufficient=no AND semantic_evidence_match in {yes,partial}

| Method | hrp_v2 | Wilson 95% CI |
|---|---|---|
| Direct | 1/27 (3.7%, 95%CI [0.7, 18.3%]) | |
| ICL | 2/25 (8.0%, 95%CI [2.2, 25.0%]) | |
| SelfRefine | 1/11 (9.1%, 95%CI [1.6, 37.7%]) | |
| Ours | 5/24 (20.8%, 95%CI [9.2, 40.5%]) | |

### 5e2. Strict HRP-v2 by Method

strict_hrp_v2 = hard_realization_pass_v2=yes AND strict_quality_pass=true AND focus_match=yes

| Method | strict_hrp_v2 | Wilson 95% CI |
|---|---|---|
| Direct | 0/27 (0.0%, 95%CI [0.0, 12.5%]) | |
| ICL | 0/25 (0.0%, 95%CI [0.0, 13.3%]) | |
| SelfRefine | 0/11 (0.0%, 95%CI [0.0, 25.9%]) | |
| Ours | 1/24 (4.2%, 95%CI [0.7, 20.2%]) | |

### 5f. Semantic Evidence Match by Method (quality-pass, judge-ok)

| Method | yes | partial | no | judge_error | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 0 | 26 | 1 | 0 | 27 |
| ICL | 0 | 23 | 2 | 0 | 25 |
| SelfRefine | 0 | 11 | 0 | 0 | 11 |
| Ours | 1 | 20 | 3 | 0 | 24 |

### 5d. Answer Focus Diagnostics (Ours)

#### Question focus distribution

| Focus | Count | Pct |
|---|---:|---:|
| bridge | 11 | 45.8% |
| state | 8 | 33.3% |
| motivation | 3 | 12.5% |
| count | 1 | 4.2% |
| outcome | 1 | 4.2% |

#### Focus match rate

- focus_match=yes: 22 / 24
- focus_match=no: 2 / 24

#### Answer role -> question focus mapping

| answer_role | question_focus | count |
|---|---|---:|
| answer | bridge | 9 |
| answer | state | 8 |
| answer | motivation | 3 |
| answer | outcome | 1 |
| answer_bridge | bridge | 2 |
| count_pattern | count | 1 |

#### Focus mismatch examples

**Mismatch 1:**
- Story: master-girl
- Question: What motivated the king's son to continue exploring the rooms despite the warnings about the kettles?
- Target answer: intrigued .
- answer_role=answer, question_focus=bridge

**Mismatch 2:**
- Story: evil-one-kitta-grau
- Question: How did the merchant come to be in a situation where he had to show Kitta Grau to the evil one despite having not been able to sell her for three weeks?
- Target answer: the merchant was unable to sell kitta grau .
- answer_role=answer, question_focus=state

### 5g. Unique Story Diversity and Cluster Diagnostic

#### Unique stories among predicted Hard (quality-pass, judge-ok)

| Method | unique stories | Hard count | stories |
|---|---:|---:|---|
| Direct | 2 | 2 | how-princess-pride-was-broken, three-dogs |
| ICL | 2 | 2 | three-dogs, youth-who-was-to-serve-three-years-without-pay |
| SelfRefine | 1 | 1 | habetrot-the-spinstress |
| Ours | 5 | 5 | evil-one-kitta-grau, little-lasse, the-fire-plume, three-dogs, youth-who-was-to-serve-three-years-without-pay |

#### Unique stories among hrp_v2 (quality-pass, judge-ok)

| Method | unique stories | hrp_v2 count | stories |
|---|---:|---:|---|
| Direct | 1 | 1 | three-dogs |
| ICL | 2 | 2 | three-dogs, youth-who-was-to-serve-three-years-without-pay |
| SelfRefine | 1 | 1 | habetrot-the-spinstress |
| Ours | 5 | 5 | evil-one-kitta-grau, little-lasse, the-fire-plume, three-dogs, youth-who-was-to-serve-three-years-without-pay |

#### Unique stories among strict_hrp_v2 (quality-pass, judge-ok)

| Method | unique stories | strict_hrp_v2 count | stories |
|---|---:|---:|---|
| Direct | 0 | 0 |  |
| ICL | 0 | 0 |  |
| SelfRefine | 0 | 0 |  |
| Ours | 1 | 1 | three-dogs |

#### Cluster diagnostic: three-dogs concentration

| Method | three-dogs in Hard | total Hard | three-dogs in hrp_v2 | total hrp_v2 | three-dogs in strict_hrp_v2 | total strict_hrp_v2 |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 1 | 2 | 1 | 1 | 0 | 0 |
| ICL | 1 | 2 | 1 | 2 | 0 | 0 |
| SelfRefine | 0 | 1 | 0 | 1 | 0 | 0 |
| Ours | 1 | 5 | 1 | 5 | 1 | 1 |

## 6. Failure Reasons by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | not answerable | 10 |
| Direct | wrong answer | 10 |
| Direct | other | 3 |
| ICL | not answerable | 13 |
| ICL | wrong answer | 9 |
| ICL | not fluent | 2 |
| ICL | other | 1 |
| SelfRefine | not answerable | 33 |
| SelfRefine | wrong answer | 5 |
| SelfRefine | other | 1 |
| Ours | not answerable | 23 |
| Ours | not fluent | 1 |
| Ours | gen error: degenerate output | 1 |
| Ours | wrong answer | 1 |

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
- Question: How many times did the old man come to the youth to exchange his dog for a pig?
- Target answer: three times .
- Quality: answerable=yes, asks_expected=yes, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 0.000, hard_realization=no, hrp_v2=yes
- Focus: answer_role=count_pattern, question_focus=count, focus_match=yes
- Semantic match: partial — The judge-identified sentences [19, 32, 33] indicate the final exchange and the old man's departure but do not fully cover the initial and second exchanges as indicated by the target evidence sentences [4, 16, 31].

**Example 2:**
- Story: little-lasse
- Question: What development connected the situation where the oars were locked up in the boat-house to the difficulty of rowing to Asia?
- Target answer: he did not have oars to row with .
- Quality: answerable=yes, asks_expected=partial, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 1.000, hard_realization=yes, hrp_v2=yes
- Focus: answer_role=answer, question_focus=bridge, focus_match=yes
- Semantic match: yes — The judge-identified sentences [6, 7, 8] cover the same causal chain as the target evidence sentences, explaining that the oars were locked up, Lasse did not notice the boat was empty, and thus it was not easy to row to Asia without oars.

**Example 3:**
- Story: evil-one-kitta-grau
- Question: How did the merchant come to be in a situation where he had to show Kitta Grau to the evil one despite having not been able to sell her for three weeks?
- Target answer: the merchant was unable to sell kitta grau .
- Quality: answerable=yes, asks_expected=partial, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 0.333, hard_realization=no, hrp_v2=yes
- Focus: answer_role=answer, question_focus=state, focus_match=no
- Semantic match: partial — The judge-identified sentences [2, 9, 10] cover the merchant's pact and his inability to sell Kitta Grau, but miss the outcome where the evil one leaves after recognizing Kitta Grau, which is covered in sentence [15] of the target evidence sentences.

### Hard realization pass v2 examples

**HRP-v2 Example 1 (Direct):**
- Story: three-dogs
- Question: Why did the young swineherd agree to exchange his pigs for dogs three times?
- Target answer: three times .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences cover the third exchange but miss the first two exchanges, which are necessary to fully explain the repeated behavior of the swineherd.

**HRP-v2 Example 2 (ICL):**
- Story: three-dogs
- Question: How many times did the youth exchange his pigs for dogs from the old man?
- Target answer: three times .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences [17, 31, 32] cover the final exchange but miss the earlier exchanges indicated by the target evidence sentences [4, 16, 31], thus missing part of the chain.

**HRP-v2 Example 3 (Ours):**
- Story: three-dogs
- Question: How many times did the old man come to the youth to exchange his dog for a pig?
- Target answer: three times .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences [19, 32, 33] indicate the final exchange and the old man's departure but do not fully cover the initial and second exchanges as indicated by the target evidence sentences [4, 16, 31].

**HRP-v2 Example 4 (Ours):**
- Story: little-lasse
- Question: What development connected the situation where the oars were locked up in the boat-house to the difficulty of rowing to Asia?
- Target answer: he did not have oars to row with .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: yes — The judge-identified sentences [6, 7, 8] cover the same causal chain as the target evidence sentences, explaining that the oars were locked up, Lasse did not notice the boat was empty, and thus it was not easy to row to Asia without oars.

**HRP-v2 Example 5 (Ours):**
- Story: evil-one-kitta-grau
- Question: How did the merchant come to be in a situation where he had to show Kitta Grau to the evil one despite having not been able to sell her for three weeks?
- Target answer: the merchant was unable to sell kitta grau .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences [2, 9, 10] cover the merchant's pact and his inability to sell Kitta Grau, but miss the outcome where the evil one leaves after recognizing Kitta Grau, which is covered in sentence [15] of the target evidence sentences.

### Ours focus-match examples (quality-pass, focus_match=yes)

**Focus Example 1:**
- Story: three-dogs
- Question: How many times did the old man come to the youth to exchange his dog for a pig?
- Target answer: three times .
- answer_role=count_pattern, question_focus=count, node_type=count

**Focus Example 2:**
- Story: three-dogs
- Question: What motivated the princes to plot against the youth and ultimately lead to his death?
- Target answer: they were jealous of the youth .
- answer_role=answer, question_focus=bridge, node_type=action

**Focus Example 3:**
- Story: werewolf
- Question: How did the princess feel after the king left for war and her step-mother showed her true nature?
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
- Question: Why did the princes take counsel together to get the better of the youth and win power and glory for themselves?
- Target answer: they were jealous of the youth .
- Predicted difficulty: Medium

### Ours failure cases (grouped)

| Failure category | Count |
|---|---:|
| degenerate / parse failure | 21 |
| not answerable | 2 |
| other | 2 |
| focus mismatch | 1 |

#### Ours failure examples

**Failure 1:**
- Story: youth-who-wanted-to-win-daughter-of-mother-in-corner
- Question: (empty)
- Raw prefix: `{
  "question":": "How . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .`
- Reason: empty question

**Failure 2:**
- Story: werewolf
- Question: (empty)
- Raw prefix: `{"question": "How on on did way the princess feel so sad and alone . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .`
- Reason: empty question

**Failure 3:**
- Story: silverwhite-lillwacker
- Question: What Ultimately, what did the court court . way lead way .
- Reason: The generated question is incoherent and does not make sense, making it impossible to determine if it is answerable or if it asks for the expected answer. The question also does not align with the story's content, and it is not grammatically correct.

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
- Story: youth-who-wanted-to-win-daughter-of-mother-in-corner
- Question: 
- Reason: empty question

## 9. Success Criteria

| Criterion | Status | Value |
|---|---|---|
| Ours quality_pass >= 65% | FAIL | 48.0% (24/50 (48.0%, 95%CI [34.8, 61.5%])) |
| Ours predicted Hard >= 25% | FAIL | 20.8% (5/24 (20.8%, 95%CI [9.2, 40.5%])) |
| Ours eval_hrp_v2 >= 25% (quality-pass, judge-ok) | FAIL | 20.8% (5/24 (20.8%, 95%CI [9.2, 40.5%])) |
| Ours strict_hrp_v2 >= 10% | FAIL | 4.2% (1/24 (4.2%, 95%CI [0.7, 20.2%])) |
| Ours unique HRP-v2 stories > each baseline | PASS | Ours=5, Direct=1, ICL=2, SelfRefine=1 |
| Ours Hard hit >= Direct/ICL/SelfRefine | PASS | Ours=0.21, Direct=0.07, ICL=0.08, SelfRefine=0.09 |

**Overall: SOME CRITERIA FAILED**
