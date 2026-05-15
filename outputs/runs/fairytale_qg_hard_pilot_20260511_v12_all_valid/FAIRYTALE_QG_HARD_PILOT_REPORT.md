# FairytaleQA Hard QG Pilot Report

Generated: 2026-05-11 12:02:04

## 1. Run Summary

| Field | Value |
|---|---|
| Methods | Direct, ICL, SelfRefine, Ours |
| Requested limit | 69 |
| Graph total | 75 |
| Graph valid | 69 |
| Selected candidates | 69 |
| Total generations | 276 |
| Target difficulty | Hard |

### Parse success by method

| Method | parse_ok | Total | Pct |
|---|---:|---:|---:|
| Direct | 61 | 69 | 88.4% |
| ICL | 65 | 69 | 94.2% |
| SelfRefine | 32 | 69 | 46.4% |
| Ours | 68 | 69 | 98.6% |

### 1b. Generation Robustness by Method

| Method | degenerate | repair_attempted | repair_success | quality_pass |
|---|---:|---:|---:|---:|
| Direct | 9 | 0 | 0 | 41 |
| ICL | 8 | 0 | 0 | 45 |
| SelfRefine | 0 | 0 | 0 | 19 |
| Ours | 17 | 36 | 6 | 56 |

## 2. Quality Pass by Method

| Method | quality_pass | strict_quality_pass | Total | Pct (loose) | Pct (strict) |
|---|---:|---:|---:|---:|---:|
| Direct | 41 | 6 | 69 | 59.4% | 8.7% |
| ICL | 45 | 4 | 69 | 65.2% | 5.8% |
| SelfRefine | 19 | 1 | 69 | 27.5% | 1.4% |
| Ours | 56 | 10 | 69 | 81.2% | 14.5% |

## 3. Blind Difficulty Distribution (quality-pass only)

| Method | Easy | Medium | Hard | JudgeError | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 3 | 34 | 4 | 0 | 41 |
| ICL | 2 | 37 | 6 | 0 | 45 |
| SelfRefine | 1 | 14 | 4 | 0 | 19 |
| Ours | 2 | 41 | 13 | 0 | 56 |

### 3b. Difficulty Judge Status by Method

| Method | judge_ok | judge_error | Total | Error rate |
|---|---:|---:|---:|---:|
| Direct | 41 | 0 | 41 | 0.0% |
| ICL | 45 | 0 | 45 | 0.0% |
| SelfRefine | 19 | 0 | 19 | 0.0% |
| Ours | 56 | 0 | 56 | 0.0% |

## 4. Hard Hit Rate by Method

Denominator: quality-pass AND difficulty_judge_status=ok

| Method | Hard hit | Wilson 95% CI |
|---|---|---|
| Direct | 4/41 (9.8%, 95%CI [3.9, 22.5%]) | |
| ICL | 6/45 (13.3%, 95%CI [6.3, 26.2%]) | |
| SelfRefine | 4/19 (21.1%, 95%CI [8.5, 43.3%]) | |
| Ours | 13/56 (23.2%, 95%CI [14.1, 35.8%]) | |

## 5. Evidence Dependency by Method (quality-pass, judge-ok only)

| Method | alone_sufficient=no | bridge_required=yes | removal in {ambiguous,unanswerable} | Total |
|---|---:|---:|---:|---:|
| Direct | 18 | 38 | 36 | 41 |
| ICL | 22 | 41 | 34 | 45 |
| SelfRefine | 13 | 16 | 16 | 19 |
| Ours | 28 | 53 | 44 | 56 |

### 5b. Target Evidence Coverage by Method (quality-pass, judge-ok only)

| Method | mean coverage | coverage>=0.67 | uses_all_target | Total |
|---|---:|---:|---:|---:|
| Direct | 0.346 | 0 | 0 | 41 |
| ICL | 0.387 | 2 | 2 | 45 |
| SelfRefine | 0.405 | 1 | 1 | 19 |
| Ours | 0.445 | 3 | 3 | 56 |

### 5c. Hard Realization Pass by Method (exact-id diagnostic)

Hard realization (legacy) = judge_ok AND num_judge_used>=3 AND uses_bridge in {yes,partial} AND coverage>=0.67 AND predicted=Hard

| Method | hard_realization_pass | quality-pass judge-ok | Rate |
|---|---:|---:|---:|
| Direct | 0 | 41 | 0.0% |
| ICL | 2 | 45 | 4.4% |
| SelfRefine | 1 | 19 | 5.3% |
| Ours | 3 | 56 | 5.4% |

### 5e. Hard Realization Pass v2 by Method

Denominator: quality-pass AND difficulty_judge_status=ok

hrp_v2 = predicted=Hard AND num_judge_used>=3 AND bridge_required=yes AND alone_sufficient=no AND semantic_evidence_match in {yes,partial}

| Method | hrp_v2 | Wilson 95% CI |
|---|---|---|
| Direct | 3/41 (7.3%, 95%CI [2.5, 19.4%]) | |
| ICL | 6/45 (13.3%, 95%CI [6.3, 26.2%]) | |
| SelfRefine | 3/19 (15.8%, 95%CI [5.5, 37.6%]) | |
| Ours | 12/56 (21.4%, 95%CI [12.7, 33.8%]) | |

### 5e2. Strict HRP-v2 by Method

strict_hrp_v2 = hard_realization_pass_v2=yes AND strict_quality_pass=true AND focus_match=yes

| Method | strict_hrp_v2 | Wilson 95% CI |
|---|---|---|
| Direct | 0/41 (0.0%, 95%CI [0.0, 8.6%]) | |
| ICL | 0/45 (0.0%, 95%CI [0.0, 7.9%]) | |
| SelfRefine | 0/19 (0.0%, 95%CI [0.0, 16.8%]) | |
| Ours | 2/56 (3.6%, 95%CI [1.0, 12.1%]) | |

### 5f. Semantic Evidence Match by Method (quality-pass, judge-ok)

| Method | yes | partial | no | judge_error | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 0 | 35 | 6 | 0 | 41 |
| ICL | 2 | 43 | 0 | 0 | 45 |
| SelfRefine | 0 | 17 | 2 | 0 | 19 |
| Ours | 3 | 50 | 3 | 0 | 56 |

### 5d. Answer Focus Diagnostics (Ours)

#### Question focus distribution

| Focus | Count | Pct |
|---|---:|---:|
| bridge | 28 | 50.0% |
| state | 13 | 23.2% |
| motivation | 11 | 19.6% |
| outcome | 3 | 5.4% |
| count | 1 | 1.8% |

#### Focus match rate

- focus_match=yes: 38 / 56
- focus_match=no: 18 / 56

#### Answer role -> question focus mapping

| answer_role | question_focus | count |
|---|---|---:|
| answer | bridge | 24 |
| answer | state | 13 |
| answer | motivation | 8 |
| answer | outcome | 3 |
| answer_bridge | bridge | 4 |
| answer_bridge | motivation | 3 |
| count_pattern | count | 1 |

#### Focus mismatch examples

**Mismatch 1:**
- Story: torre-jeppe
- Question: What larger goal was the girl pursuing by agreeing to go to church and fetch back Torre Jeppe, given the tailors' request and her response?
- Target answer: she was brave .
- answer_role=answer, question_focus=motivation

**Mismatch 2:**
- Story: lame-dog
- Question: What was the final consequence of the youngest princess's response to her sisters' question about wishing for a husband?
- Target answer: she did not care who she married .
- answer_role=answer, question_focus=outcome

**Mismatch 3:**
- Story: thomas-the-rhymer
- Question: What development connected the Queen's change and Thomas's terror to his begging for mercy?
- Target answer: scared .
- answer_role=answer, question_focus=bridge

#### Per question_focus metrics (all Ours)

| Focus | N | quality_pass | Hard | hrp_v2 | strict_hrp_v2 | focus_match=yes |
|---|---:|---:|---:|---:|---:|---:|
| bridge | 32 | 28 | 7 | 6 | 0 | 19 |
| count | 1 | 1 | 1 | 1 | 1 | 1 |
| motivation | 15 | 11 | 2 | 2 | 0 | 11 |
| outcome | 6 | 3 | 0 | 0 | 0 | 2 |
| state | 15 | 13 | 3 | 3 | 1 | 11 |

#### Per answer_role metrics (all Ours)

| answer_role | N | quality_pass | Hard | hrp_v2 | strict_hrp_v2 |
|---|---:|---:|---:|---:|---:|
| answer | 60 | 48 | 12 | 11 | 1 |
| answer_bridge | 8 | 7 | 0 | 0 | 0 |
| count_pattern | 1 | 1 | 1 | 1 | 1 |

#### Per answer_node_type metrics (all Ours)

| node_type | N | quality_pass | Hard | hrp_v2 | strict_hrp_v2 |
|---|---:|---:|---:|---:|---:|
| action | 17 | 16 | 4 | 3 | 0 |
| belief | 3 | 2 | 0 | 0 | 0 |
| count | 1 | 1 | 1 | 1 | 1 |
| description | 14 | 11 | 2 | 2 | 0 |
| emotion | 5 | 5 | 1 | 1 | 0 |
| goal | 2 | 1 | 0 | 0 | 0 |
| motivation | 8 | 5 | 1 | 1 | 0 |
| outcome | 6 | 3 | 0 | 0 | 0 |
| problem | 1 | 1 | 1 | 1 | 0 |
| state | 12 | 11 | 3 | 3 | 1 |

### 5g. Unique Story Diversity and Cluster Diagnostic

#### Unique stories among predicted Hard (quality-pass, judge-ok)

| Method | unique stories | Hard count | stories |
|---|---:|---:|---|
| Direct | 3 | 4 | master-girl, three-dogs, youth-who-was-to-serve-three-years-without-pay |
| ICL | 6 | 6 | habetrot-the-spinstress, leelinau-the-lost-daughter, per-gynt, soria-moria-castle, the-fairies-of-merlin-crag, youth-who-was-to-serve-three-years-without-pay |
| SelfRefine | 3 | 4 | evil-one-kitta-grau, master-girl, youth-who-was-to-serve-three-years-without-pay |
| Ours | 11 | 13 | evil-one-kitta-grau, habetrot-the-spinstress, leelinau-the-lost-daughter, little-lasse, per-gynt, the-fairies-of-merlin-crag, the-fire-plume, thomas-the-rhymer, three-dogs, youth-who-wanted-to-win-daughter-of-mother-in-corner, youth-who-was-to-serve-three-years-without-pay |

#### Unique stories among hrp_v2 (quality-pass, judge-ok)

| Method | unique stories | hrp_v2 count | stories |
|---|---:|---:|---|
| Direct | 2 | 3 | three-dogs, youth-who-was-to-serve-three-years-without-pay |
| ICL | 6 | 6 | habetrot-the-spinstress, leelinau-the-lost-daughter, per-gynt, soria-moria-castle, the-fairies-of-merlin-crag, youth-who-was-to-serve-three-years-without-pay |
| SelfRefine | 2 | 3 | evil-one-kitta-grau, youth-who-was-to-serve-three-years-without-pay |
| Ours | 10 | 12 | evil-one-kitta-grau, habetrot-the-spinstress, leelinau-the-lost-daughter, little-lasse, per-gynt, the-fairies-of-merlin-crag, the-fire-plume, three-dogs, youth-who-wanted-to-win-daughter-of-mother-in-corner, youth-who-was-to-serve-three-years-without-pay |

#### Unique stories among strict_hrp_v2 (quality-pass, judge-ok)

| Method | unique stories | strict_hrp_v2 count | stories |
|---|---:|---:|---|
| Direct | 0 | 0 |  |
| ICL | 0 | 0 |  |
| SelfRefine | 0 | 0 |  |
| Ours | 2 | 2 | three-dogs, youth-who-wanted-to-win-daughter-of-mother-in-corner |

#### Cluster diagnostic: three-dogs concentration

| Method | three-dogs in Hard | total Hard | three-dogs in hrp_v2 | total hrp_v2 | three-dogs in strict_hrp_v2 | total strict_hrp_v2 |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 1 | 4 | 1 | 3 | 0 | 0 |
| ICL | 0 | 6 | 0 | 6 | 0 | 0 |
| SelfRefine | 0 | 4 | 0 | 3 | 0 | 0 |
| Ours | 1 | 13 | 1 | 12 | 1 | 2 |

## 6. Failure Reasons by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | not answerable | 15 |
| Direct | wrong answer | 10 |
| Direct | not fluent | 2 |
| Direct | other | 1 |
| ICL | not answerable | 14 |
| ICL | wrong answer | 6 |
| ICL | not fluent | 3 |
| ICL | other | 1 |
| SelfRefine | not answerable | 44 |
| SelfRefine | wrong answer | 3 |
| SelfRefine | other | 2 |
| SelfRefine | not fluent | 1 |
| Ours | not answerable | 5 |
| Ours | wrong answer | 4 |
| Ours | not fluent | 3 |
| Ours | gen error: degenerate output | 1 |

### 6b. Difficulty Judge Parse Failures

None.

## 7. Copy/Reference Diagnostics

| Method | Total | Copies source | Copy rate |
|---|---:|---:|---:|
| Direct | 69 | 0 | 0.0% |
| ICL | 69 | 0 | 0.0% |
| SelfRefine | 69 | 0 | 0.0% |
| Ours | 69 | 0 | 0.0% |

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
- Semantic match: partial — The judge-identified sentences [17, 31, 32] cover the final exchange and the agreement to the exchange, but miss the initial offer which is captured in the target evidence sentence [16].

**Example 2:**
- Story: youth-who-wanted-to-win-daughter-of-mother-in-corner
- Question: How did the mother feel as her son grew older and continued to skip and dance without working?
- Target answer: unhappy .
- Quality: answerable=yes, asks_expected=yes, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 0.667, hard_realization=yes, hrp_v2=yes
- Focus: answer_role=answer, question_focus=state, focus_match=yes
- Semantic match: partial — The judge-identified sentences [3, 5, 6] cover the worsening condition of the mother and the son's behavior, but miss the direct description of the son's activities from sentence [2], which is relevant to understanding the mother's unhappiness.

**Example 3:**
- Story: thomas-the-rhymer
- Question: What development connected the Queen's change and Thomas's terror to his begging for mercy?
- Target answer: scared .
- Quality: answerable=yes, asks_expected=partial, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 0.000, hard_realization=no, hrp_v2=no
- Focus: answer_role=answer, question_focus=bridge, focus_match=no
- Semantic match: no — no question or no judge-used sentences

### Hard realization pass v2 examples

**HRP-v2 Example 1 (Direct):**
- Story: three-dogs
- Question: Why did the young swineherd agree to exchange his pigs for dogs three times?
- Target answer: three times .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences cover the final exchange but miss the earlier exchanges' motivations, which are included in the target evidence sentences.

**HRP-v2 Example 2 (Ours):**
- Story: three-dogs
- Question: How many times did the old man offer to exchange his dog for one of the youth's pigs?
- Target answer: three times .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences [17, 31, 32] cover the final exchange and the agreement to the exchange, but miss the initial offer which is captured in the target evidence sentence [16].

**HRP-v2 Example 3 (Ours):**
- Story: youth-who-wanted-to-win-daughter-of-mother-in-corner
- Question: How did the mother feel as her son grew older and continued to skip and dance without working?
- Target answer: unhappy .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences [3, 5, 6] cover the worsening condition of the mother and the son's behavior, but miss the direct description of the son's activities from sentence [2], which is relevant to understanding the mother's unhappiness.

**HRP-v2 Example 4 (SelfRefine):**
- Story: evil-one-kitta-grau
- Question: Why was Kitta Grau unable to be sold by the merchant in three weeks despite being put in a glass cage and treated as a curious bird?
- Target answer: the merchant was unable to sell kitta grau .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences [5, 7, 15] cover the part where Kitta Grau is treated as a bird and the failure to sell her, but they miss the merchant's dialogue with the evil one in sentence [9], which explains the pact and the three-week condition, a key part of the reasoning chain.

**HRP-v2 Example 5 (Ours):**
- Story: evil-one-kitta-grau
- Question: How did the merchant come to be in a position where he had to show Kitta Grau to the evil one despite having patience?
- Target answer: the merchant was unable to sell kitta grau .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial — The judge-identified sentences [9, 10, 11] cover the moment when the merchant shows Kitta Grau to the evil one, which is part of the reasoning chain. However, they miss the context provided by sentence [7] about the passage of time and the lack of interest in Kitta Grau, which is crucial for understanding the merchant's situation. Sentence [15] also provides a conclusion to the reasoning chain that is missing in the judge-identified sentences.

### Ours focus-match examples (quality-pass, focus_match=yes)

**Focus Example 1:**
- Story: three-dogs
- Question: How many times did the old man offer to exchange his dog for one of the youth's pigs?
- Target answer: three times .
- answer_role=count_pattern, question_focus=count, node_type=count

**Focus Example 2:**
- Story: three-dogs
- Question: Why did the princes take counsel together to get the better of the youth and win power and glory for themselves?
- Target answer: they were jealous of the youth .
- answer_role=answer, question_focus=bridge, node_type=action

**Focus Example 3:**
- Story: youth-who-wanted-to-win-daughter-of-mother-in-corner
- Question: How did the mother feel as her son grew older and continued to skip and dance without working?
- Target answer: unhappy .
- answer_role=answer, question_focus=state, node_type=state

### Best baseline examples (quality-pass)

**Direct Example:**
- Story: three-dogs
- Question: Why did the young swineherd agree to exchange his pigs for dogs three times?
- Target answer: three times .
- Predicted difficulty: Hard

**ICL Example:**
- Story: three-dogs
- Question: Why did the princes decide to attack their comrade who had saved the king's daughters?
- Target answer: they were jealous of the youth .
- Predicted difficulty: Easy

**SelfRefine Example:**
- Story: werewolf
- Question: Why did the little old man appear and greet the princess? 
- Target answer: help her .
- Predicted difficulty: Medium

### Ours failure cases (grouped)

| Failure category | Count |
|---|---:|
| degenerate / parse failure | 5 |
| not fluent | 3 |
| answer mismatch | 3 |
| not answerable | 2 |

#### Ours failure examples

**Failure 1:**
- Story: werewolf
- Question: Why did the princess feel embarrassed and look around when she heard the voice?
- Reason: The question asks about the princess's reaction to hearing a voice, but the target answer 'help her.' does not match the content or context of the story. The story does not indicate that the princess's embarrassment or looking around was related to someone helping her.

**Failure 2:**
- Story: how-princess-pride-was-broken
- Question: How the princess react when she the gooseherd asked for five and twenty kisses ?
- Reason: The question is not fully grammatically correct and natural-sounding, as it should be 'reacted' instead of 'react' and 'when she saw the gooseherd asked' instead of 'when she the gooseherd asked'.

**Failure 3:**
- Story: master-girl
- Question: WhyS . 'S4'. 'S6'. 'S .'. 'S . . 'S . . 'S . . 'S . . 'S . . 'S . . 'S . . 'S . 
- Reason: The generated question is not coherent, contains numerous formatting errors, and does not form a clear or answerable question based on the story provided.

### Baseline failure cases

**Failure 1 (ICL):**
- Story: three-dogs
- Question: Why why Why did the youth agree not with the dog and the two pigs in the wood? .
- Reason: The question is not coherent and does not make sense, making it impossible to determine if it is answerable or if it asks for the expected answer. The fluency is also poor due to grammatical errors and lack of clarity.

**Failure 2 (SelfRefine):**
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
| Ours quality_pass >= 65% | PASS | 81.2% (56/69 (81.2%, 95%CI [70.4, 88.6%])) |
| Ours predicted Hard >= 25% | FAIL | 23.2% (13/56 (23.2%, 95%CI [14.1, 35.8%])) |
| Ours eval_hrp_v2 >= 25% (quality-pass, judge-ok) | FAIL | 21.4% (12/56 (21.4%, 95%CI [12.7, 33.8%])) |
| Ours strict_hrp_v2 >= 10% | FAIL | 3.6% (2/56 (3.6%, 95%CI [1.0, 12.1%])) |
| Ours unique HRP-v2 stories > each baseline | PASS | Ours=10, Direct=2, ICL=6, SelfRefine=2 |
| Ours Hard hit >= Direct/ICL/SelfRefine | PASS | Ours=0.23, Direct=0.10, ICL=0.13, SelfRefine=0.21 |

**Overall: SOME CRITERIA FAILED**
