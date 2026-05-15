# FairytaleQA Hard QG Pilot Report

Generated: 2026-05-10 22:33:48

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
| ICL | 13 | 15 | 86.7% |
| SelfRefine | 5 | 15 | 33.3% |
| Ours | 11 | 15 | 73.3% |

## 2. Quality Pass by Method

| Method | quality_pass | strict_quality_pass | Total | Pct (loose) | Pct (strict) |
|---|---:|---:|---:|---:|---:|
| Direct | 14 | 5 | 15 | 93.3% | 33.3% |
| ICL | 10 | 2 | 15 | 66.7% | 13.3% |
| SelfRefine | 5 | 1 | 15 | 33.3% | 6.7% |
| Ours | 10 | 5 | 15 | 66.7% | 33.3% |

## 3. Blind Difficulty Distribution (quality-pass only)

| Method | Easy | Medium | Hard | JudgeError | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 0 | 13 | 1 | 0 | 14 |
| ICL | 1 | 8 | 1 | 0 | 10 |
| SelfRefine | 0 | 3 | 2 | 0 | 5 |
| Ours | 0 | 9 | 1 | 0 | 10 |

### 3b. Difficulty Judge Status by Method

| Method | judge_ok | judge_error | Total | Error rate |
|---|---:|---:|---:|---:|
| Direct | 14 | 0 | 14 | 0.0% |
| ICL | 10 | 0 | 10 | 0.0% |
| SelfRefine | 5 | 0 | 5 | 0.0% |
| Ours | 10 | 0 | 10 | 0.0% |

## 4. Hard Hit Rate by Method

Denominator: quality-pass AND difficulty_judge_status=ok

| Method | Hard | judge-ok QP | Hard hit rate |
|---|---:|---:|---:|
| Direct | 1 | 14 | 7.1% |
| ICL | 1 | 10 | 10.0% |
| SelfRefine | 2 | 5 | 40.0% |
| Ours | 1 | 10 | 10.0% |

## 5. Evidence Dependency by Method (quality-pass, judge-ok only)

| Method | alone_sufficient=no | bridge_required=yes | removal in {ambiguous,unanswerable} | Total |
|---|---:|---:|---:|---:|
| Direct | 5 | 14 | 10 | 14 |
| ICL | 5 | 9 | 6 | 10 |
| SelfRefine | 2 | 5 | 2 | 5 |
| Ours | 6 | 10 | 7 | 10 |

### 5b. Target Evidence Coverage by Method (quality-pass, judge-ok only)

| Method | mean coverage | coverage>=0.67 | uses_all_target | Total |
|---|---:|---:|---:|---:|
| Direct | 0.375 | 0 | 0 | 14 |
| ICL | 0.342 | 0 | 0 | 10 |
| SelfRefine | 0.383 | 0 | 0 | 5 |
| Ours | 0.462 | 0 | 0 | 10 |

### 5c. Hard Realization Pass by Method

Hard realization = judge_ok AND num_judge_used>=3 AND uses_bridge in {yes,partial} AND coverage>=0.67 AND predicted=Hard

| Method | hard_realization_pass | quality-pass judge-ok | Rate |
|---|---:|---:|---:|
| Direct | 0 | 14 | 0.0% |
| ICL | 0 | 10 | 0.0% |
| SelfRefine | 0 | 5 | 0.0% |
| Ours | 0 | 10 | 0.0% |

## 6. Failure Reasons by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | other | 1 |
| ICL | not answerable | 2 |
| ICL | other | 1 |
| ICL | not fluent | 1 |
| ICL | wrong answer | 1 |
| SelfRefine | not answerable | 10 |
| Ours | not answerable | 4 |
| Ours | not fluent | 1 |

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
- Story: youth-who-wanted-to-win-daughter-of-mother-in-corner
- Question: How did the mother feel as the son continued to sing and dance all day and night, leading to her decision?
- Target answer: unhappy .
- Quality: answerable=yes, asks_expected=yes, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 0.667, hard_realization=no

### Best baseline examples (quality-pass)

**Direct Example:**
- Story: three-dogs
- Question: Why did the young swineherd agree to exchange his pigs for dogs three times?
- Target answer: three times .
- Predicted difficulty: Hard

**ICL Example:**
- Story: three-dogs
- Question: How many times did the youth exchange his pigs for dogs?
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
- Question: Why the old man proposed the young sw to exchange the dog for one of his pigs because .
- Reason: The question is not fully formed and does not clearly ask why the old man proposed the exchange three times. It lacks clarity and grammatical correctness.

**Failure 2 (SelfRefine):**
- Story: three-dogs
- Question: 
- Reason: empty question

**Failure 3 (Direct):**
- Story: silverwhite-lillwacker
- Question: Why did the courtier force the princess to promise that he, and none other, had rescued her?
- Reason: The question is answerable from the story, but the target answer does not fully match the story's content, as it mentions two princesses instead of one.

## 9. Success Criteria

| Criterion | Status | Value |
|---|---|---|
| Ours quality-pass >= 60% | PASS | 66.7% |
| Ours strict quality-pass >= 40% | FAIL | 33.3% |
| Ours predicted Hard among quality-pass >= 20% | FAIL | 10.0% |
| Ours hard_realization_pass >= 15% | FAIL | 0.0% |
| Ours bridge_required=yes >= 50% | PASS | 100.0% |
| Ours Hard hit > Direct/ICL/SelfRefine | FAIL | Ours=0.10, Direct=0.07, ICL=0.10, SelfRefine=0.40 |
| No method copies source question | PASS | pass |

**Overall: SOME CRITERIA FAILED**
