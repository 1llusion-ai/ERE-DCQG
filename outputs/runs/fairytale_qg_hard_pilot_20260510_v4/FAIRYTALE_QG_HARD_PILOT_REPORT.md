# FairytaleQA Hard QG Pilot Report

Generated: 2026-05-10 20:35:20

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
| ICL | 15 | 15 | 100.0% |
| SelfRefine | 5 | 15 | 33.3% |
| Ours | 13 | 15 | 86.7% |

## 2. Quality Pass by Method

| Method | quality_pass | Total | Pct |
|---|---:|---:|---:|
| Direct | 11 | 15 | 73.3% |
| ICL | 12 | 15 | 80.0% |
| SelfRefine | 5 | 15 | 33.3% |
| Ours | 11 | 15 | 73.3% |

## 3. Blind Difficulty Distribution (quality-pass only)

| Method | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| Direct | 1 | 10 | 0 | 11 |
| ICL | 0 | 12 | 0 | 12 |
| SelfRefine | 0 | 5 | 0 | 5 |
| Ours | 0 | 11 | 0 | 11 |

## 4. Hard Hit Rate by Method

predicted Hard / quality-pass

| Method | Hard | quality-pass | Hard hit rate |
|---|---:|---:|---:|
| Direct | 0 | 11 | 0.0% |
| ICL | 0 | 12 | 0.0% |
| SelfRefine | 0 | 5 | 0.0% |
| Ours | 0 | 11 | 0.0% |

## 5. Evidence Dependency by Method (quality-pass only)

| Method | alone_sufficient=no | bridge_required=yes | removal in {ambiguous,unanswerable} | Total |
|---|---:|---:|---:|---:|
| Direct | 7 | 10 | 0 | 11 |
| ICL | 8 | 12 | 0 | 12 |
| SelfRefine | 1 | 5 | 0 | 5 |
| Ours | 9 | 11 | 0 | 11 |

## 6. Failure Reasons by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | not answerable | 3 |
| Direct | other | 1 |
| ICL | wrong answer | 1 |
| ICL | not fluent | 1 |
| ICL | not answerable | 1 |
| SelfRefine | not answerable | 10 |
| Ours | not answerable | 3 |
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
- Question: Why did the young swineherd agree to exchange his pigs for dogs three times?
- Target answer: three times .
- Predicted difficulty: Medium

**ICL Example:**
- Story: three-dogs
- Question: How many times did the youth exchange his pigs for dogs from the old man?
- Target answer: three times .
- Predicted difficulty: Medium

**SelfRefine Example:**
- Story: youth-who-wanted-to-win-daughter-of-mother-in-corner
- Question: Why did the mother eventually tell the youth to get to work, considering the increasing strain on her resources due to his growing appetite and clothing needs?
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
- Story: silverwhite-lillwacker
- Question: Why did the courtier force the princess to promise that he, and none other, had rescued her?
- Reason: The question is answerable from the story, but the expected answer does not fully match the target answer as the story only mentions one princess, not two. The target answer introduces an inconsistency by mentioning two princesses.

## 9. Success Criteria

| Criterion | Status | Value |
|---|---|---|
| Ours quality-pass >= 60% | PASS | 73.3% |
| Ours predicted Hard among quality-pass >= 20% | FAIL | 0.0% |
| Ours bridge_required=yes >= 50% | PASS | 100.0% |
| Ours Hard hit > Direct/ICL/SelfRefine | FAIL | Ours=0.00, Direct=0.00, ICL=0.00, SelfRefine=0.00 |
| No method copies source question | PASS | pass |

**Overall: SOME CRITERIA FAILED**
