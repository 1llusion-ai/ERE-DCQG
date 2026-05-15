# FairytaleQA Hard QG Pilot Report

Generated: 2026-05-10 20:21:35

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
| ICL | 14 | 15 | 93.3% |
| SelfRefine | 7 | 15 | 46.7% |
| Ours | 15 | 15 | 100.0% |

## 2. Quality Pass by Method

| Method | quality_pass | Total | Pct |
|---|---:|---:|---:|
| Direct | 12 | 15 | 80.0% |
| ICL | 11 | 15 | 73.3% |
| SelfRefine | 7 | 15 | 46.7% |
| Ours | 14 | 15 | 93.3% |

## 3. Blind Difficulty Distribution (quality-pass only)

| Method | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| Direct | 3 | 9 | 0 | 12 |
| ICL | 3 | 8 | 0 | 11 |
| SelfRefine | 1 | 6 | 0 | 7 |
| Ours | 3 | 11 | 0 | 14 |

## 4. Hard Hit Rate by Method

predicted Hard / quality-pass

| Method | Hard | quality-pass | Hard hit rate |
|---|---:|---:|---:|
| Direct | 0 | 12 | 0.0% |
| ICL | 0 | 11 | 0.0% |
| SelfRefine | 0 | 7 | 0.0% |
| Ours | 0 | 14 | 0.0% |

## 5. Evidence Dependency by Method (quality-pass only)

| Method | alone_sufficient=no | bridge_required=yes | removal in {ambiguous,unanswerable} | Total |
|---|---:|---:|---:|---:|
| Direct | 7 | 9 | 0 | 12 |
| ICL | 5 | 8 | 0 | 11 |
| SelfRefine | 3 | 6 | 0 | 7 |
| Ours | 8 | 11 | 0 | 14 |

## 6. Failure Reasons by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | not answerable | 2 |
| Direct | other | 1 |
| ICL | not answerable | 3 |
| ICL | wrong answer | 1 |
| SelfRefine | not answerable | 8 |
| Ours | wrong answer | 1 |

## 7. Copy/Reference Diagnostics

| Method | Total | Copies source | Copy rate |
|---|---:|---:|---:|
| Direct | 15 | 0 | 0.0% |
| ICL | 15 | 0 | 0.0% |
| SelfRefine | 15 | 0 | 0.0% |
| Ours | 15 | 0 | 0.0% |

## 8. Examples

### Best Ours examples (quality-pass, predicted Hard)

### Best baseline examples (quality-pass)

**Direct Example:**
- Story: three-dogs
- Question: How many times did the youth exchange his pigs for dogs?
- Target answer: three times .
- Predicted difficulty: Medium

**ICL Example:**
- Story: three-dogs
- Question: How many times did the youth exchange his pigs for dogs from the old man?
- Target answer: three times .
- Predicted difficulty: Medium

**SelfRefine Example:**
- Story: youth-who-wanted-to-win-daughter-of-mother-in-corner
- Question: Why did the mother eventually tell the youth to get to work, considering the increasing strain on her resources and the youth's growing needs?
- Target answer: she could not afford to care for him anymore .
- Predicted difficulty: Medium

### Failure cases

**Failure 1 (SelfRefine):**
- Story: three-dogs
- Question: 
- Reason: empty question

**Failure 2 (SelfRefine):**
- Story: three-dogs
- Question: 
- Reason: empty question

**Failure 3 (Direct):**
- Story: youth-who-wanted-to-win-daughter-of-mother-in-corner
- Question: 
- Reason: empty question

## 9. Success Criteria

| Criterion | Status | Value |
|---|---|---|
| Ours quality-pass >= 60% | PASS | 93.3% |
| Ours predicted Hard among quality-pass >= 20% | FAIL | 0.0% |
| Ours bridge_required=yes >= 50% | PASS | 78.6% |
| Ours Hard hit > Direct/ICL/SelfRefine | FAIL | Ours=0.00, Direct=0.00, ICL=0.00, SelfRefine=0.00 |
| No method copies source question | PASS | pass |

**Overall: SOME CRITERIA FAILED**
