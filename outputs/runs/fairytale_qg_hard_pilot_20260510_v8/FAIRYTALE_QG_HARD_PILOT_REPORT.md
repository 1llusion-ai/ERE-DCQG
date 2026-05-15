# FairytaleQA Hard QG Pilot Report

Generated: 2026-05-11 00:16:00

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
| Direct | 13 | 15 | 86.7% |
| ICL | 9 | 15 | 60.0% |
| SelfRefine | 9 | 15 | 60.0% |
| Ours | 13 | 15 | 86.7% |

## 2. Quality Pass by Method

| Method | quality_pass | strict_quality_pass | Total | Pct (loose) | Pct (strict) |
|---|---:|---:|---:|---:|---:|
| Direct | 12 | 3 | 15 | 80.0% | 20.0% |
| ICL | 7 | 0 | 15 | 46.7% | 0.0% |
| SelfRefine | 6 | 1 | 15 | 40.0% | 6.7% |
| Ours | 12 | 5 | 15 | 80.0% | 33.3% |

## 3. Blind Difficulty Distribution (quality-pass only)

| Method | Easy | Medium | Hard | JudgeError | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 0 | 10 | 2 | 0 | 12 |
| ICL | 1 | 5 | 1 | 0 | 7 |
| SelfRefine | 0 | 5 | 1 | 0 | 6 |
| Ours | 0 | 9 | 3 | 0 | 12 |

### 3b. Difficulty Judge Status by Method

| Method | judge_ok | judge_error | Total | Error rate |
|---|---:|---:|---:|---:|
| Direct | 12 | 0 | 12 | 0.0% |
| ICL | 7 | 0 | 7 | 0.0% |
| SelfRefine | 6 | 0 | 6 | 0.0% |
| Ours | 12 | 0 | 12 | 0.0% |

## 4. Hard Hit Rate by Method

Denominator: quality-pass AND difficulty_judge_status=ok

| Method | Hard | judge-ok QP | Hard hit rate |
|---|---:|---:|---:|
| Direct | 2 | 12 | 16.7% |
| ICL | 1 | 7 | 14.3% |
| SelfRefine | 1 | 6 | 16.7% |
| Ours | 3 | 12 | 25.0% |

## 5. Evidence Dependency by Method (quality-pass, judge-ok only)

| Method | alone_sufficient=no | bridge_required=yes | removal in {ambiguous,unanswerable} | Total |
|---|---:|---:|---:|---:|
| Direct | 4 | 12 | 10 | 12 |
| ICL | 3 | 6 | 3 | 7 |
| SelfRefine | 2 | 6 | 3 | 6 |
| Ours | 7 | 12 | 11 | 12 |

### 5b. Target Evidence Coverage by Method (quality-pass, judge-ok only)

| Method | mean coverage | coverage>=0.67 | uses_all_target | Total |
|---|---:|---:|---:|---:|
| Direct | 0.361 | 0 | 0 | 12 |
| ICL | 0.440 | 0 | 0 | 7 |
| SelfRefine | 0.250 | 0 | 0 | 6 |
| Ours | 0.496 | 0 | 0 | 12 |

### 5c. Hard Realization Pass by Method

Hard realization = judge_ok AND num_judge_used>=3 AND uses_bridge in {yes,partial} AND coverage>=0.67 AND predicted=Hard

| Method | hard_realization_pass | quality-pass judge-ok | Rate |
|---|---:|---:|---:|
| Direct | 0 | 12 | 0.0% |
| ICL | 0 | 7 | 0.0% |
| SelfRefine | 0 | 6 | 0.0% |
| Ours | 0 | 12 | 0.0% |

### 5d. Answer Focus Diagnostics (Ours)

#### Question focus distribution

| Focus | Count | Pct |
|---|---:|---:|
| bridge | 6 | 50.0% |
| state | 4 | 33.3% |
| count | 1 | 8.3% |
| motivation | 1 | 8.3% |

#### Focus match rate

- focus_match=yes: 11 / 12
- focus_match=no: 1 / 12

#### Answer role -> question focus mapping

| answer_role | question_focus | count |
|---|---|---:|
| answer | bridge | 4 |
| answer | state | 4 |
| answer | motivation | 1 |
| answer_bridge | bridge | 2 |
| count_pattern | count | 1 |

#### Focus mismatch examples

**Mismatch 1:**
- Story: three-dogs
- Question: What development connected the princes' jealousy to the youth's demise?
- Target answer: they were jealous of the youth .
- answer_role=answer, question_focus=bridge

## 6. Failure Reasons by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | not answerable | 2 |
| Direct | other | 1 |
| ICL | not answerable | 7 |
| ICL | other | 1 |
| SelfRefine | not answerable | 8 |
| SelfRefine | other | 1 |
| Ours | not answerable | 3 |

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
- Question: How many times did the old man propose exchanging his dog for one of the boy's pigs?
- Target answer: three times .
- Quality: answerable=yes, asks_expected=yes, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 0.333, hard_realization=no
- Focus: answer_role=count_pattern, question_focus=count, focus_match=yes

**Example 2:**
- Story: master-girl
- Question: What was the king's son's state of mind as he decided to try the soup in the third room after discovering the effects of the first two?
- Target answer: intrigued .
- Quality: answerable=yes, asks_expected=partial, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 0.200, hard_realization=no
- Focus: answer_role=answer, question_focus=bridge, focus_match=yes

**Example 3:**
- Story: evil-one-kitta-grau
- Question: What was the ultimate goal of Kitta Grau's actions as evidenced by the events in the story?
- Target answer: kitta grau managed to sow dissension between the couple .
- Quality: answerable=yes, asks_expected=yes, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 0.333, hard_realization=no
- Focus: answer_role=answer, question_focus=motivation, focus_match=yes

### Ours focus-match examples (quality-pass, focus_match=yes)

**Focus Example 1:**
- Story: three-dogs
- Question: How many times did the old man propose exchanging his dog for one of the boy's pigs?
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
- Story: three-dogs
- Question: Why did the princes decide to attack their comrade who had saved the king's daughters?
- Target answer: they were jealous of the youth .
- Predicted difficulty: Easy

**SelfRefine Example:**
- Story: three-dogs
- Question: Why did the princes take counsel together to get the better of the youth and win power and glory for themselves, and how did their jealousy manifest?
- Target answer: they were jealous of the youth .
- Predicted difficulty: Medium

### Failure cases

**Failure 1 (ICL):**
- Story: three-dogs
- Question: 
- Reason: empty question

**Failure 2 (SelfRefine):**
- Story: three-dogs
- Question: 
- Reason: empty question

**Failure 3 (SelfRefine):**
- Story: youth-who-wanted-to-win-daughter-of-mother-in-corner
- Question: Why did the mother eventually tell her son to get to work after he failed to find a suitable bride for himself?
- Reason: The question incorrectly assumes the son attempted to find a bride before being told to work, which is not supported by the story. The story does not mention the son failing to find a suitable bride.

## 9. Success Criteria

| Criterion | Status | Value |
|---|---|---|
| Ours quality-pass >= 60% | PASS | 80.0% |
| Ours strict quality-pass >= 40% | FAIL | 33.3% |
| Ours focus_match=yes >= 50% | PASS | 91.7% |
| Ours predicted Hard among quality-pass >= 20% | PASS | 25.0% |
| Ours hard_realization_pass >= 15% | FAIL | 0.0% |
| Ours bridge_required=yes >= 50% | PASS | 100.0% |
| Ours Hard hit > Direct/ICL/SelfRefine | PASS | Ours=0.25, Direct=0.17, ICL=0.14, SelfRefine=0.17 |
| No method copies source question | PASS | pass |

**Overall: SOME CRITERIA FAILED**
