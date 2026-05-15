# FairytaleQA Hard QG Pilot Report

Generated: 2026-05-10 21:32:34

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
| Direct | 11 | 15 | 73.3% |
| ICL | 13 | 15 | 86.7% |
| SelfRefine | 3 | 15 | 20.0% |
| Ours | 13 | 15 | 86.7% |

## 2. Quality Pass by Method

| Method | quality_pass | Total | Pct |
|---|---:|---:|---:|
| Direct | 11 | 15 | 73.3% |
| ICL | 9 | 15 | 60.0% |
| SelfRefine | 2 | 15 | 13.3% |
| Ours | 12 | 15 | 80.0% |

## 3. Blind Difficulty Distribution (quality-pass only)

| Method | Easy | Medium | Hard | JudgeError | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 0 | 10 | 1 | 0 | 11 |
| ICL | 2 | 7 | 0 | 0 | 9 |
| SelfRefine | 0 | 2 | 0 | 0 | 2 |
| Ours | 0 | 11 | 1 | 0 | 12 |

### 3b. Difficulty Judge Status by Method

| Method | judge_ok | judge_error | Total | Error rate |
|---|---:|---:|---:|---:|
| Direct | 11 | 0 | 11 | 0.0% |
| ICL | 9 | 0 | 9 | 0.0% |
| SelfRefine | 2 | 0 | 2 | 0.0% |
| Ours | 12 | 0 | 12 | 0.0% |

## 4. Hard Hit Rate by Method

Denominator: quality-pass AND difficulty_judge_status=ok

| Method | Hard | judge-ok QP | Hard hit rate |
|---|---:|---:|---:|
| Direct | 1 | 11 | 9.1% |
| ICL | 0 | 9 | 0.0% |
| SelfRefine | 0 | 2 | 0.0% |
| Ours | 1 | 12 | 8.3% |

## 5. Evidence Dependency by Method (quality-pass, judge-ok only)

| Method | alone_sufficient=no | bridge_required=yes | removal in {ambiguous,unanswerable} | Total |
|---|---:|---:|---:|---:|
| Direct | 4 | 11 | 10 | 11 |
| ICL | 4 | 7 | 6 | 9 |
| SelfRefine | 0 | 2 | 2 | 2 |
| Ours | 6 | 12 | 10 | 12 |

## 6. Failure Reasons by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | not answerable | 4 |
| ICL | not answerable | 3 |
| ICL | wrong answer | 2 |
| ICL | not fluent | 1 |
| SelfRefine | not answerable | 12 |
| SelfRefine | wrong answer | 1 |
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
- Question: Why did the youth agree to exchange his pigs for the old man's dogs three times?
- Target answer: three times .
- Quality: answerable=yes, asks_expected=partial, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes

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
- Story: thomas-the-rhymer
- Question: Why did Thomas have to mount the elfin queen's steed according to the story?
- Target answer: he needed to pay for the price of kissing her .
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
- Story: three-dogs
- Question: 
- Reason: empty question

## 9. Success Criteria

| Criterion | Status | Value |
|---|---|---|
| Ours quality-pass >= 60% | PASS | 80.0% |
| Ours predicted Hard among quality-pass >= 20% | FAIL | 8.3% |
| Ours bridge_required=yes >= 50% | PASS | 100.0% |
| Ours Hard hit > Direct/ICL/SelfRefine | FAIL | Ours=0.08, Direct=0.09, ICL=0.00, SelfRefine=0.00 |
| No method copies source question | PASS | pass |

**Overall: SOME CRITERIA FAILED**
