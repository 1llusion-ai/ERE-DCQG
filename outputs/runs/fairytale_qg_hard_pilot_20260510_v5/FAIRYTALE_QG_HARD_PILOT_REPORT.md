# FairytaleQA Hard QG Pilot Report

Generated: 2026-05-10 20:46:59

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
| Direct | 14 | 15 | 93.3% |
| ICL | 12 | 15 | 80.0% |
| SelfRefine | 9 | 15 | 60.0% |
| Ours | 15 | 15 | 100.0% |

## 2. Quality Pass by Method

| Method | quality_pass | Total | Pct |
|---|---:|---:|---:|
| Direct | 12 | 15 | 80.0% |
| ICL | 9 | 15 | 60.0% |
| SelfRefine | 8 | 15 | 53.3% |
| Ours | 11 | 15 | 73.3% |

## 3. Blind Difficulty Distribution (quality-pass only)

| Method | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| Direct | 1 | 9 | 2 | 12 |
| ICL | 1 | 7 | 1 | 9 |
| SelfRefine | 0 | 7 | 1 | 8 |
| Ours | 3 | 8 | 0 | 11 |

## 4. Hard Hit Rate by Method

predicted Hard / quality-pass

| Method | Hard | quality-pass | Hard hit rate |
|---|---:|---:|---:|
| Direct | 2 | 12 | 16.7% |
| ICL | 1 | 9 | 11.1% |
| SelfRefine | 1 | 8 | 12.5% |
| Ours | 0 | 11 | 0.0% |

## 5. Evidence Dependency by Method (quality-pass only)

| Method | alone_sufficient=no | bridge_required=yes | removal in {ambiguous,unanswerable} | Total |
|---|---:|---:|---:|---:|
| Direct | 4 | 11 | 10 | 12 |
| ICL | 3 | 7 | 8 | 9 |
| SelfRefine | 4 | 8 | 6 | 8 |
| Ours | 5 | 8 | 7 | 11 |

## 6. Failure Reasons by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | other | 1 |
| Direct | wrong answer | 1 |
| Direct | not answerable | 1 |
| ICL | not answerable | 4 |
| ICL | wrong answer | 1 |
| ICL | not fluent | 1 |
| SelfRefine | not answerable | 7 |
| Ours | wrong answer | 2 |
| Ours | other | 1 |
| Ours | not answerable | 1 |

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
- Predicted difficulty: Hard

**ICL Example:**
- Story: three-dogs
- Question: How many times did the youth exchange his pigs for dogs from the old man?
- Target answer: three times .
- Predicted difficulty: Hard

**SelfRefine Example:**
- Story: three-dogs
- Question: Why did the young swineherd agree to exchange his pigs for dogs each time he saw the old man with a new, larger dog?
- Target answer: three times .
- Predicted difficulty: Hard

### Failure cases

**Failure 1 (Ours):**
- Story: three-dogs
- Question: What was the motivation behind the youth to agree the dog in exchange for one of his pigs?
- Reason: The question is not clear and does not match the target answer 'three times.' The question should ask about the frequency of the exchange, not the motivation. The fluency is also poor due to grammatical issues.

**Failure 2 (ICL):**
- Story: youth-who-wanted-to-win-daughter-of-mother-in-corner
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
| Ours bridge_required=yes >= 50% | PASS | 72.7% |
| Ours Hard hit > Direct/ICL/SelfRefine | FAIL | Ours=0.00, Direct=0.17, ICL=0.11, SelfRefine=0.12 |
| No method copies source question | PASS | pass |

**Overall: SOME CRITERIA FAILED**
