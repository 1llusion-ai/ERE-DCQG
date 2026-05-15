# FairytaleQA Hard QG Pilot Report

Generated: 2026-05-10 20:10:30

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
| SelfRefine | 5 | 15 | 33.3% |
| Ours | 13 | 15 | 86.7% |

## 2. Quality Pass by Method

| Method | quality_pass | Total | Pct |
|---|---:|---:|---:|
| Direct | 2 | 15 | 13.3% |
| ICL | 3 | 15 | 20.0% |
| SelfRefine | 1 | 15 | 6.7% |
| Ours | 2 | 15 | 13.3% |

## 3. Blind Difficulty Distribution (quality-pass only)

| Method | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| Direct | 2 | 0 | 0 | 2 |
| ICL | 3 | 0 | 0 | 3 |
| SelfRefine | 1 | 0 | 0 | 1 |
| Ours | 2 | 0 | 0 | 2 |

## 4. Hard Hit Rate by Method

predicted Hard / quality-pass

| Method | Hard | quality-pass | Hard hit rate |
|---|---:|---:|---:|
| Direct | 0 | 2 | 0.0% |
| ICL | 0 | 3 | 0.0% |
| SelfRefine | 0 | 1 | 0.0% |
| Ours | 0 | 2 | 0.0% |

## 5. Evidence Dependency by Method (quality-pass only)

| Method | alone_sufficient=no | bridge_required=yes | removal in {ambiguous,unanswerable} | Total |
|---|---:|---:|---:|---:|
| Direct | 0 | 0 | 0 | 2 |
| ICL | 0 | 0 | 0 | 3 |
| SelfRefine | 0 | 0 | 0 | 1 |
| Ours | 0 | 0 | 0 | 2 |

## 6. Failure Reasons by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | not answerable | 9 |
| Direct | other | 2 |
| Direct | wrong answer | 2 |
| ICL | not answerable | 8 |
| ICL | wrong answer | 3 |
| ICL | other | 1 |
| SelfRefine | not answerable | 12 |
| SelfRefine | other | 1 |
| SelfRefine | wrong answer | 1 |
| Ours | not answerable | 10 |
| Ours | wrong answer | 2 |
| Ours | other | 1 |

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
- Story: master-girl
- Question: Why did the king's son dip a lock of his hair into the second kettle?
- Target answer: the second kettle turned his hair into silver .
- Predicted difficulty: Easy

**ICL Example:**
- Story: three-dogs
- Question: How many times did the youth exchange his pigs for dogs from the old man?
- Target answer: three times .
- Predicted difficulty: Easy

**SelfRefine Example:**
- Story: master-girl
- Question: Why did the king's son think the second soup was the most expensive one he had tried?
- Target answer: the second kettle turned his hair into silver .
- Predicted difficulty: Easy

### Failure cases

**Failure 1 (Direct):**
- Story: three-dogs
- Question: 
- Reason: empty question

**Failure 2 (SelfRefine):**
- Story: three-dogs
- Question: Why did the youth agree to exchange his last pig for the third dog?
- Reason: The question can be answered from the story, but the expected answer 'three times' does not match the actual reason given in the story. The story indicates the youth agreed to exchange his last pig for the third dog because he wanted a dog to keep him company in the wilderness, not because of a specific number of exchanges.

**Failure 3 (Ours):**
- Story: three-dogs
- Question: 
- Reason: empty question

## 9. Success Criteria

| Criterion | Status | Value |
|---|---|---|
| Ours quality-pass >= 60% | FAIL | 13.3% |
| Ours predicted Hard among quality-pass >= 20% | FAIL | 0.0% |
| Ours bridge_required=yes >= 50% | FAIL | 0.0% |
| Ours Hard hit > Direct/ICL/SelfRefine | FAIL | Ours=0.00, Direct=0.00, ICL=0.00, SelfRefine=0.00 |
| No method copies source question | PASS | pass |

**Overall: SOME CRITERIA FAILED**
